from __future__ import annotations
 
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
 
DB_PATH = Path("artifacts/audit.db")
 
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
    payload_json      TEXT NOT NULL
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
    FOREIGN KEY (case_id) REFERENCES predictions(case_id)
);
 
CREATE INDEX IF NOT EXISTS idx_pred_created ON predictions(created_utc);
CREATE INDEX IF NOT EXISTS idx_verdict_case ON verdicts(case_id);
"""
 
 
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
 
 
def connect(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path), check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(_SCHEMA)
    return con
 
 
def log_prediction(con: sqlite3.Connection, pred, filename: str | None = None) -> str:
    case_id = uuid.uuid4().hex[:12]
    d = pred.to_dict()
    con.execute(
        """INSERT INTO predictions (case_id, created_utc, image_sha256, filename,
             decision, probability, grade, gradable, quality_reasons,
             operating_point, abstain_low, abstain_high, model_version,
             preprocess_fp, is_demo, latency_ms, payload_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (case_id, _now(), d["image_sha256"], filename, d["decision"],
         d["probability"], d["grade"], int(d["quality"].get("gradable", False)),
         "; ".join(d["quality"].get("reasons", [])), d["operating_point"],
         d["abstain_low"], d["abstain_high"], d["model_version"],
         d["preprocess_fingerprint"], int(d["is_demo"]), d["latency_ms"],
         json.dumps(d)),
    )
    con.commit()
    return case_id
 
 
def log_verdict(con, case_id: str, verdict: str, grade=None, grader=None,
                note=None) -> str:
    """
    Record a human decision. Never overwrites: a revised opinion is a new row,
    so the sequence of clinical reasoning stays reconstructable.
    """
    row = con.execute("SELECT decision FROM predictions WHERE case_id = ?",
                      (case_id,)).fetchone()
    if row is None:
        raise KeyError(f"Unknown case_id: {case_id}")
 
    agreed = None
    if verdict in ("REFER", "ROUTINE") and row["decision"] in ("REFER", "ROUTINE"):
        agreed = int(verdict == row["decision"])
 
    vid = uuid.uuid4().hex[:12]
    con.execute(
        """INSERT INTO verdicts (verdict_id, case_id, created_utc, grader,
             verdict, grade, agreed, note) VALUES (?,?,?,?,?,?,?,?)""",
        (vid, case_id, _now(), grader, verdict, grade, agreed, note),
    )
    con.commit()
    return vid
 
 
def list_cases(con, limit: int = 200) -> list[dict]:
    """
    Worklist, ordered clinically rather than chronologically: cases needing a
    human first, then likely referrals, then the rest. A queue sorted by upload
    time makes the model's triage useless.
    """
    rows = con.execute(
        """SELECT p.*,
                  (SELECT v.verdict FROM verdicts v WHERE v.case_id = p.case_id
                    ORDER BY v.created_utc DESC LIMIT 1) AS latest_verdict,
                  (SELECT v.grader FROM verdicts v WHERE v.case_id = p.case_id
                    ORDER BY v.created_utc DESC LIMIT 1) AS latest_grader
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
    row = con.execute("SELECT * FROM predictions WHERE case_id = ?",
                      (case_id,)).fetchone()
    if row is None:
        return None
    out = dict(row)
    out["payload"] = json.loads(out.pop("payload_json"))
    out["verdicts"] = [dict(v) for v in con.execute(
        "SELECT * FROM verdicts WHERE case_id = ? ORDER BY created_utc", (case_id,))]
    return out
 
 
def summary(con) -> dict:
    """Operational rollup for the dashboard strip."""
    q = lambda sql, *a: con.execute(sql, a).fetchone()[0]
    total = q("SELECT COUNT(*) FROM predictions")
    reviewed = q("SELECT COUNT(DISTINCT case_id) FROM verdicts")
    agreed = q("SELECT COUNT(*) FROM verdicts WHERE agreed = 1")
    disagreed = q("SELECT COUNT(*) FROM verdicts WHERE agreed = 0")
    return {
        "total_cases": total,
        "refer": q("SELECT COUNT(*) FROM predictions WHERE decision='REFER'"),
        "routine": q("SELECT COUNT(*) FROM predictions WHERE decision='ROUTINE'"),
        "manual_review": q("SELECT COUNT(*) FROM predictions WHERE decision='MANUAL_REVIEW'"),
        "ungradable": q("SELECT COUNT(*) FROM predictions WHERE gradable=0"),
        "reviewed": reviewed,
        "agreement_rate": (agreed / (agreed + disagreed)) if (agreed + disagreed) else None,
        "median_latency_ms": q(
            "SELECT AVG(latency_ms) FROM predictions") or 0.0,
    }
 