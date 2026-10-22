"""
Append-only, tamper-evident audit log (SQLite).

Every prediction, grader verdict, sign-in, case view and export is recorded.
Three layers keep the record trustworthy:

  1. Triggers reject UPDATE and DELETE on predictions, verdicts and the log, so
     the application cannot rewrite history -- a revised opinion is a new row.
  2. Every entry is hash-chained to the one before it, and prediction/verdict
     entries carry the SHA-256 of the row they describe. verify_chain()
     recomputes all of it, so editing the file directly (dropping the triggers
     first) is detected. Export the head hash off-site periodically to make
     truncation detectable as well.
  3. One connection, guarded by a lock, in WAL mode: FastAPI serves requests
     from a thread pool, and an unguarded shared sqlite3 connection is not
     safe across threads.

No pixels are stored here. Images live encrypted in storage.ImageStore; this
log holds their hashes.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import statistics
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = Path("artifacts/audit.db")
GENESIS = "0" * 64

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    case_id           TEXT PRIMARY KEY,
    created_utc       TEXT NOT NULL,
    image_sha256      TEXT NOT NULL,
    filename          TEXT,
    decision          TEXT NOT NULL,
    probability       REAL,
    grade             INTEGER,
    gradable          INTEGER NOT NULL,
    quality_reasons   TEXT,
    operating_point   REAL,
    abstain_low       REAL,
    abstain_high      REAL,
    model_version     TEXT NOT NULL,
    preprocess_fp     TEXT NOT NULL,
    is_demo           INTEGER NOT NULL,
    latency_ms        REAL,
    payload_json      TEXT NOT NULL,
    patient_id        TEXT,
    eye               TEXT,
    encounter_id      TEXT,
    device            TEXT,
    uploaded_by       TEXT,
    source_format     TEXT,
    validated         INTEGER
);

CREATE TABLE IF NOT EXISTS verdicts (
    verdict_id    TEXT PRIMARY KEY,
    case_id       TEXT NOT NULL,
    created_utc   TEXT NOT NULL,
    grader        TEXT,
    verdict       TEXT NOT NULL,
    grade         INTEGER,
    agreed        INTEGER,
    note          TEXT,
    dme           INTEGER,
    findings      TEXT,
    FOREIGN KEY (case_id) REFERENCES predictions(case_id)
);

CREATE TABLE IF NOT EXISTS audit_log (
    seq             INTEGER PRIMARY KEY,
    created_utc     TEXT NOT NULL,
    actor           TEXT NOT NULL,
    action          TEXT NOT NULL,
    entity_id       TEXT,
    detail          TEXT NOT NULL,
    record_sha256   TEXT,
    prev_hash       TEXT NOT NULL,
    entry_hash      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pred_created ON predictions(created_utc);
CREATE INDEX IF NOT EXISTS idx_verdict_case ON verdicts(case_id);
"""

# Columns added after the first release. connect() adds any that an older
# database is missing; existing rows get NULL.
_MIGRATIONS = {
    "predictions": ["patient_id TEXT", "eye TEXT", "encounter_id TEXT",
                    "device TEXT", "uploaded_by TEXT", "source_format TEXT",
                    "validated INTEGER"],
    # Grader labels for the learning loop: maculopathy present (1/0, NULL if
    # not recorded) and other findings (JSON list of codes).
    "verdicts": ["dme INTEGER", "findings TEXT"],
}

_POST_MIGRATION = """
CREATE INDEX IF NOT EXISTS idx_pred_encounter ON predictions(encounter_id);
CREATE INDEX IF NOT EXISTS idx_pred_patient ON predictions(patient_id);
"""

_APPEND_ONLY = ("predictions", "verdicts", "audit_log")


class LockedConnection(sqlite3.Connection):
    """sqlite3 connection carrying the lock every caller must hold."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.lock = threading.RLock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def record_hash(row: dict) -> str:
    """
    Hash of a stored row. NULL columns are left out, so adding a column in a
    later migration does not invalidate rows written before it existed --
    while changing any recorded value, including blanking it, still does.
    """
    return _sha({k: v for k, v in row.items() if v is not None})


def connect(db_path: Path | str = DB_PATH) -> LockedConnection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path), check_same_thread=False,
                          factory=LockedConnection)
    con.row_factory = sqlite3.Row
    with con.lock:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        con.executescript(_SCHEMA)
        for table, cols in _MIGRATIONS.items():
            have = {r["name"] for r in con.execute(f"PRAGMA table_info({table})")}
            for col in cols:
                if col.split()[0] not in have:
                    con.execute(f"ALTER TABLE {table} ADD COLUMN {col}")
        con.executescript(_POST_MIGRATION)
        for t in _APPEND_ONLY:
            for op in ("UPDATE", "DELETE"):
                con.execute(
                    f"CREATE TRIGGER IF NOT EXISTS {t}_no_{op.lower()} "
                    f"BEFORE {op} ON {t} BEGIN "
                    f"SELECT RAISE(ABORT, '{t} is append-only'); END")
        _anchor_unchained(con)
        con.commit()
    return con


def _anchor_unchained(con) -> None:
    """
    Give every prediction and verdict that has no chain entry one now.

    Rows written before the hash chain existed would otherwise sit outside it,
    and an edit to them would go unnoticed. Anchoring proves nothing about
    what happened to a row BEFORE this moment -- the entry says so with
    legacy_anchor=True -- but from here on it is covered like any other.
    Idempotent: a row is anchored once. Caller holds con.lock.
    """
    preds = con.execute(
        """SELECT * FROM predictions WHERE case_id NOT IN
             (SELECT entity_id FROM audit_log WHERE action='prediction')
           ORDER BY created_utc, rowid""").fetchall()
    for r in preds:
        _append(con, "migration", "prediction", r["case_id"],
                {"decision": r["decision"], "legacy_anchor": True},
                record_hash(dict(r)))
    verdicts = con.execute(
        """SELECT * FROM verdicts WHERE verdict_id NOT IN
             (SELECT json_extract(detail, '$.verdict_id') FROM audit_log
               WHERE action='verdict')
           ORDER BY created_utc, rowid""").fetchall()
    for r in verdicts:
        _append(con, "migration", "verdict", r["case_id"],
                {"verdict_id": r["verdict_id"], "verdict": r["verdict"],
                 "legacy_anchor": True},
                record_hash(dict(r)))


# -- hash chain ----------------------------------------------------------------

def _append(con, actor: str, action: str, entity_id: str | None, detail: dict,
            record_sha256: str | None = None) -> None:
    """Caller holds con.lock and commits (so the record and its entry land together)."""
    last = con.execute("SELECT seq, entry_hash FROM audit_log "
                       "ORDER BY seq DESC LIMIT 1").fetchone()
    seq = (last["seq"] + 1) if last else 1
    prev = last["entry_hash"] if last else GENESIS
    entry = {"seq": seq, "created_utc": _now(), "actor": actor, "action": action,
             "entity_id": entity_id, "detail": json.dumps(detail, sort_keys=True),
             "record_sha256": record_sha256, "prev_hash": prev}
    entry["entry_hash"] = _sha(entry)
    con.execute("INSERT INTO audit_log (seq, created_utc, actor, action, entity_id, "
                "detail, record_sha256, prev_hash, entry_hash) "
                "VALUES (:seq,:created_utc,:actor,:action,:entity_id,:detail,"
                ":record_sha256,:prev_hash,:entry_hash)", entry)


def log_event(con, actor: str, action: str, entity_id: str | None = None,
              detail: dict | None = None) -> None:
    """Sign-ins, views, exports, account changes."""
    with con.lock:
        _append(con, actor, action, entity_id, detail or {})
        con.commit()


def last_event_detail(con, action: str, entity_id: str) -> dict | None:
    with con.lock:
        r = con.execute("SELECT detail FROM audit_log WHERE action=? AND entity_id=? "
                        "ORDER BY seq DESC LIMIT 1", (action, entity_id)).fetchone()
    return json.loads(r["detail"]) if r else None


def verify_chain(con) -> dict:
    """Recompute every link and every referenced record. ok=False lists why."""
    problems, prev, n = [], GENESIS, 0
    with con.lock:
        entries = con.execute("SELECT * FROM audit_log ORDER BY seq").fetchall()
        for e in entries:
            n += 1
            e = dict(e)
            if e["prev_hash"] != prev:
                problems.append(f"entry {e['seq']}: chain broken (previous hash mismatch)")
            body = {k: e[k] for k in ("seq", "created_utc", "actor", "action",
                                      "entity_id", "detail", "record_sha256", "prev_hash")}
            if _sha(body) != e["entry_hash"]:
                problems.append(f"entry {e['seq']}: entry altered")
            if e["action"] in ("prediction", "verdict"):
                table, key = (("predictions", "case_id") if e["action"] == "prediction"
                              else ("verdicts", "verdict_id"))
                rec_id = (e["entity_id"] if e["action"] == "prediction"
                          else json.loads(e["detail"]).get("verdict_id"))
                row = con.execute(f"SELECT * FROM {table} WHERE {key}=?",
                                  (rec_id,)).fetchone()
                if row is None:
                    problems.append(f"entry {e['seq']}: {table} row {rec_id} missing")
                elif record_hash(dict(row)) != e["record_sha256"]:
                    problems.append(f"entry {e['seq']}: {table} row {rec_id} altered")
            prev = e["entry_hash"]
    return {"ok": not problems, "entries": n, "head_hash": prev, "problems": problems}


# -- records -------------------------------------------------------------------

def log_prediction(con, pred, filename: str | None = None, *, patient_id=None,
                   eye=None, encounter_id=None, device=None, uploaded_by=None,
                   source_format=None) -> str:
    case_id = uuid.uuid4().hex[:12]
    d = pred.to_dict()
    row = {
        "case_id": case_id, "created_utc": _now(), "image_sha256": d["image_sha256"],
        "filename": filename, "decision": d["decision"], "probability": d["probability"],
        "grade": d["grade"], "gradable": int(d["quality"].get("gradable", False)),
        "quality_reasons": "; ".join(d["quality"].get("reasons", [])),
        "operating_point": d["operating_point"], "abstain_low": d["abstain_low"],
        "abstain_high": d["abstain_high"], "model_version": d["model_version"],
        "preprocess_fp": d["preprocess_fingerprint"], "is_demo": int(d["is_demo"]),
        "latency_ms": d["latency_ms"], "payload_json": json.dumps(d),
        "patient_id": patient_id, "eye": eye, "encounter_id": encounter_id,
        "device": device, "uploaded_by": uploaded_by, "source_format": source_format,
        "validated": int(bool(d.get("validated"))),
    }
    cols = ", ".join(row)
    with con.lock:
        con.execute(f"INSERT INTO predictions ({cols}) VALUES "
                    f"({', '.join(':' + k for k in row)})", row)
        stored = dict(con.execute("SELECT * FROM predictions WHERE case_id=?",
                                  (case_id,)).fetchone())
        _append(con, uploaded_by or "system", "prediction", case_id,
                {"decision": row["decision"], "encounter_id": encounter_id, "eye": eye},
                record_hash(stored))
        con.commit()
    return case_id


# Other findings a grader can record. They do not change the DR decision; they
# build the labelled data a future model for these conditions would need, and
# the monitoring view counts them.
FINDINGS = {
    "glaucoma_suspect": "Glaucoma suspected (optic disc)",
    "amd": "Age-related macular degeneration",
    "rvo": "Retinal vein occlusion",
    "hypertensive": "Hypertensive retinopathy",
    "other": "Other finding (described in the note)",
}


def log_verdict(con, case_id: str, verdict: str, grade=None, grader=None,
                note=None, dme: bool | None = None,
                findings: list[str] | None = None) -> str:
    """
    Record a human decision. Never overwrites: a revised opinion is a new row,
    so the sequence of clinical reasoning stays reconstructable.
    """
    with con.lock:
        row = con.execute("SELECT decision FROM predictions WHERE case_id = ?",
                          (case_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown case_id: {case_id}")

        agreed = None
        if verdict in ("REFER", "ROUTINE") and row["decision"] in ("REFER", "ROUTINE"):
            agreed = int(verdict == row["decision"])

        vid = uuid.uuid4().hex[:12]
        bad = [f for f in (findings or []) if f not in FINDINGS]
        if bad:
            raise ValueError(f"Unknown finding(s): {bad}")
        con.execute(
            """INSERT INTO verdicts (verdict_id, case_id, created_utc, grader,
                 verdict, grade, agreed, note, dme, findings)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (vid, case_id, _now(), grader, verdict, grade, agreed, note,
             None if dme is None else int(bool(dme)),
             json.dumps(sorted(set(findings))) if findings else None),
        )
        stored = dict(con.execute("SELECT * FROM verdicts WHERE verdict_id=?",
                                  (vid,)).fetchone())
        _append(con, grader or "unattributed", "verdict", case_id,
                {"verdict_id": vid, "verdict": verdict}, record_hash(stored))
        con.commit()
    return vid


_LATEST_VERDICT = """(SELECT v.{col} FROM verdicts v WHERE v.case_id = p.case_id
                       ORDER BY v.created_utc DESC, v.rowid DESC LIMIT 1)"""

_CASE_COLUMNS = """p.case_id, p.created_utc, p.filename, p.decision, p.probability,
    p.grade, p.gradable, p.is_demo, p.validated, p.patient_id, p.eye,
    p.encounter_id, p.device, p.uploaded_by, p.model_version"""


def list_cases(con, limit: int = 200) -> list[dict]:
    """
    Worklist, ordered clinically rather than chronologically: cases needing a
    human first, then likely referrals, then the rest. A queue sorted by upload
    time makes the model's triage useless.
    """
    with con.lock:
        rows = con.execute(
            f"""SELECT {_CASE_COLUMNS},
                      {_LATEST_VERDICT.format(col='verdict')} AS latest_verdict,
                      {_LATEST_VERDICT.format(col='grader')} AS latest_grader
                 FROM predictions p
                ORDER BY CASE p.decision
                           WHEN 'MANUAL_REVIEW' THEN 0
                           WHEN 'REFER'         THEN 1
                           ELSE 2 END,
                         p.probability DESC NULLS LAST,
                         p.created_utc DESC
                LIMIT ?""", (limit,)).fetchall()
    return [dict(r) for r in rows]


def case_detail(con, case_id: str) -> dict | None:
    with con.lock:
        row = con.execute("SELECT * FROM predictions WHERE case_id = ?",
                          (case_id,)).fetchone()
        if row is None:
            return None
        out = dict(row)
        out["payload"] = json.loads(out.pop("payload_json"))
        out["verdicts"] = [dict(v) for v in con.execute(
            "SELECT * FROM verdicts WHERE case_id = ? ORDER BY created_utc, rowid",
            (case_id,))]
    return out


def encounter_cases(con, encounter_id: str) -> list[dict]:
    """
    Every image in one screening visit, each with its latest human verdict and
    its maculopathy result (read from the stored payload; None if not assessed).
    """
    with con.lock:
        rows = con.execute(
            f"""SELECT {_CASE_COLUMNS}, p.payload_json,
                      {_LATEST_VERDICT.format(col='verdict')} AS latest_verdict,
                      {_LATEST_VERDICT.format(col='grader')} AS latest_grader
                 FROM predictions p WHERE p.encounter_id = ?
                ORDER BY p.created_utc""", (encounter_id,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["dme"] = json.loads(d.pop("payload_json") or "{}").get("dme")
        out.append(d)
    return out


def list_visits(con, limit: int = 200) -> list[dict]:
    """
    One row per screening visit (encounter), most recent first, each with all
    of its images. The worklist a clinician works through is patients, not
    images: one visit is typically 2-4 images.
    """
    with con.lock:
        heads = con.execute(
            """SELECT encounter_id, MAX(patient_id) AS patient_id,
                      MIN(created_utc) AS first_utc, MAX(created_utc) AS last_utc,
                      COUNT(*) AS n_images
                 FROM predictions WHERE encounter_id IS NOT NULL
                GROUP BY encounter_id ORDER BY last_utc DESC LIMIT ?""",
            (limit,)).fetchall()
    out = []
    for h in heads:
        visit = dict(h)
        visit["cases"] = encounter_cases(con, h["encounter_id"])
        out.append(visit)
    return out


def _latest_agreement(con, where: str = "", args: tuple = ()) -> tuple[int, int]:
    """(agreed, disagreed) counting each case ONCE, by its latest verdict."""
    row = con.execute(
        f"""SELECT SUM(a = 1), SUM(a = 0) FROM (
              SELECT {_LATEST_VERDICT.format(col='agreed')} AS a
                FROM predictions p {where}) WHERE a IS NOT NULL""", args).fetchone()
    return int(row[0] or 0), int(row[1] or 0)


def summary(con) -> dict:
    """Operational rollup for the dashboard strip."""
    with con.lock:
        q = lambda sql, *a: con.execute(sql, a).fetchone()[0]
        agreed, disagreed = _latest_agreement(con)
        latencies = [r[0] for r in con.execute(
            "SELECT latency_ms FROM predictions WHERE latency_ms IS NOT NULL")]
        return {
            "total_cases": q("SELECT COUNT(*) FROM predictions"),
            "refer": q("SELECT COUNT(*) FROM predictions WHERE decision='REFER'"),
            "routine": q("SELECT COUNT(*) FROM predictions WHERE decision='ROUTINE'"),
            "manual_review": q("SELECT COUNT(*) FROM predictions WHERE decision='MANUAL_REVIEW'"),
            "ungradable": q("SELECT COUNT(*) FROM predictions WHERE gradable=0"),
            "reviewed": q("SELECT COUNT(DISTINCT case_id) FROM verdicts"),
            "agreement_rate": (agreed / (agreed + disagreed)) if (agreed + disagreed) else None,
            "median_latency_ms": statistics.median(latencies) if latencies else 0.0,
        }


# -- post-deployment monitoring -------------------------------------------------
#
# Offline metrics cannot see drift: a new camera, a new photographer, a changed
# population. These are the signals that can, computed from live operation.

RECENT_DAYS = 28
MIN_N = 30                    # below this, a rate is too noisy to alert on
AGREEMENT_FLOOR = 0.85        # grader/model agreement on auto-decided cases
UNGRADABLE_RISE = 0.10        # absolute rise vs baseline
REFER_RATE_SHIFT = 0.10       # absolute change vs baseline
DEVICE_UNGRADABLE_CEILING = 0.25


def _rates(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    reviewed = [r for r in rows if r["agreed"] is not None]
    probs = [r["probability"] for r in rows if r["probability"] is not None]
    return {
        "n": n,
        "refer_rate": sum(r["decision"] == "REFER" for r in rows) / n,
        "manual_rate": sum(r["decision"] == "MANUAL_REVIEW" for r in rows) / n,
        "ungradable_rate": sum(not r["gradable"] for r in rows) / n,
        "mean_probability": (sum(probs) / len(probs)) if probs else None,
        "reviewed": len(reviewed),
        "agreement_rate": (sum(r["agreed"] for r in reviewed) / len(reviewed))
                          if reviewed else None,
    }


def monitoring(con, days: int = 90, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    since = (now - timedelta(days=days)).isoformat(timespec="seconds")
    with con.lock:
        rows = [dict(r) for r in con.execute(
            f"""SELECT p.created_utc, p.decision, p.gradable, p.probability,
                       p.device, p.is_demo, p.validated,
                       {_LATEST_VERDICT.format(col='agreed')} AS agreed
                  FROM predictions p WHERE p.created_utc >= ?
                 ORDER BY p.created_utc""", (since,))]

    weekly: dict[str, list] = {}
    by_device: dict[str, list] = {}
    for r in rows:
        y, w, _ = datetime.fromisoformat(r["created_utc"]).isocalendar()
        weekly.setdefault(f"{y}-W{w:02d}", []).append(r)
        by_device.setdefault(r["device"] or "unknown", []).append(r)

    cut = (now - timedelta(days=RECENT_DAYS)).isoformat(timespec="seconds")
    recent = _rates([r for r in rows if r["created_utc"] >= cut])
    baseline = _rates([r for r in rows if r["created_utc"] < cut])

    alerts = []
    if recent.get("reviewed", 0) >= 20 and recent["agreement_rate"] < AGREEMENT_FLOOR:
        alerts.append(f"Grader agreement over the last {RECENT_DAYS} days is "
                      f"{recent['agreement_rate']:.0%}, below {AGREEMENT_FLOOR:.0%}. "
                      f"Possible drift: re-validate before continuing.")
    if recent.get("n", 0) >= MIN_N and baseline.get("n", 0) >= MIN_N:
        rise = recent["ungradable_rate"] - baseline["ungradable_rate"]
        if rise > UNGRADABLE_RISE:
            alerts.append(f"Ungradable rate rose {rise:+.0%} versus baseline "
                          f"({baseline['ungradable_rate']:.0%} -> "
                          f"{recent['ungradable_rate']:.0%}). Check camera and capture.")
        shift = recent["refer_rate"] - baseline["refer_rate"]
        if abs(shift) > REFER_RATE_SHIFT:
            alerts.append(f"Referral rate changed {shift:+.0%} versus baseline. A "
                          f"population or input shift can do this; investigate.")
    for dev, drows in by_device.items():
        rr = _rates(drows)
        if rr["n"] >= 20 and rr["ungradable_rate"] > DEVICE_UNGRADABLE_CEILING:
            alerts.append(f"Device '{dev}': {rr['ungradable_rate']:.0%} of captures "
                          f"ungradable (n={rr['n']}).")
    unvalidated = sum(1 for r in rows if not r["is_demo"] and r["validated"] == 0)
    if unvalidated:
        alerts.append(f"{unvalidated} prediction(s) in this window came from a model "
                      f"that did not pass the acceptance gate.")

    return {
        "window_days": days, "recent_days": RECENT_DAYS,
        "recent": recent, "baseline": baseline,
        "weekly": [{"week": k, **_rates(v)} for k, v in sorted(weekly.items())],
        "by_device": [{"device": k, **_rates(v)} for k, v in sorted(by_device.items())],
        "alerts": alerts,
    }
