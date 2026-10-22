"""
Export what doctors decided into a pseudonymised training snapshot.

    python learning/snapshot.py                       # -> snapshots/<UTC date-time>/
    python learning/snapshot.py --out D:\\research\\snapshots

This is step 1 of the learning loop (see learning/README.md). Every image
whose latest grader verdict exists becomes one labelled row:

  verdict          REFER / ROUTINE / UNGRADABLE   (the grader's, never the model's)
  gradable         0 if the grader said UNGRADABLE -> labels for the quality model
  referable        1 for REFER, 0 for ROUTINE      -> labels for the DR model
  grade, dme       optional ICDR grade / maculopathy, when the grader gave them
  findings         optional other findings (glaucoma suspect, AMD, ...)
  model_*          what the model said at the time, for disagreement analysis

Pseudonymisation. Patient and visit identifiers are replaced by keyed
HMAC-SHA256 codes. The key (secrets/pseudonym.key) is separate from the image
key, created on first use, and never written into a snapshot: without it a
snapshot cannot be linked back to a patient by its codes. This is
PSEUDONYMISATION, not anonymisation -- the images themselves are personal
data under GDPR, so snapshots stay on hospital storage (encrypted disk), are
covered by the study's data-protection assessment, and are gitignored.

Locked test set. Each patient code is assigned permanently to "train" or
"locked_test" (default 20%) by the code's own value. The same patient lands in
the same set in every future snapshot, so a model trained on one snapshot is
always tested on patients it has never seen -- the basis of every "the new
version is better on our own patients" claim.

Dates are reduced to the month. Names, birth dates and DICOM headers are not
stored by the platform at all.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PSEUDONYM_KEY = Path("secrets/pseudonym.key")
LOCKED_TEST_FRACTION = 0.20


def load_pseudonym_key(path: Path = PSEUDONYM_KEY) -> bytes:
    env = os.environ.get("DR_PSEUDONYM_KEY")
    if env:
        return env.encode()
    if path.exists():
        return path.read_bytes().strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_hex(32).encode()
    path.write_bytes(key)
    print(f"  Created {path}. Back it up with the image key: it keeps patient codes "
          f"(and so the locked test set) stable across snapshots.")
    return key


def code(key: bytes, kind: str, value: str) -> str:
    """Keyed, irreversible code: same input -> same code, for this key only."""
    return hmac.new(key, f"{kind}:{value}".encode(), hashlib.sha256).hexdigest()[:16]


def split_of(patient_code: str, locked_fraction: float = LOCKED_TEST_FRACTION) -> str:
    """Permanent assignment from the code itself: no state to lose or tamper with."""
    return "locked_test" if int(patient_code[:8], 16) / 0xFFFFFFFF < locked_fraction else "train"


def export(con, store, out_dir: Path, key: bytes, actor: str = "snapshot",
           locked_fraction: float = LOCKED_TEST_FRACTION) -> dict:
    import cv2
    import numpy as np
    import pandas as pd
    import audit

    with con.lock:
        rows = [dict(r) for r in con.execute(
            """SELECT p.case_id, p.patient_id, p.eye, p.encounter_id, p.device,
                      p.created_utc, p.decision, p.probability, p.gradable AS model_gradable,
                      p.model_version, p.payload_json,
                      v.verdict, v.grade, v.dme, v.findings, v.grader
                 FROM predictions p
                 JOIN verdicts v ON v.verdict_id = (
                      SELECT v2.verdict_id FROM verdicts v2 WHERE v2.case_id = p.case_id
                       ORDER BY v2.created_utc DESC, v2.rowid DESC LIMIT 1)
                WHERE p.patient_id IS NOT NULL
                ORDER BY p.created_utc""")]
    if not rows:
        raise SystemExit("  No graded images yet: nothing to export.")

    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=False)
    out, missing, graders = [], 0, set()
    for r in rows:
        raw = store.get(r["case_id"], "raw")
        if raw is None:
            missing += 1                       # removed by the retention policy
            continue
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            missing += 1
            continue
        pc = code(key, "patient", r["patient_id"])
        ic = code(key, "case", r["case_id"])
        name = f"{pc}_{ic}.png"               # quality_model.patient_of() -> patient code
        cv2.imwrite(str(img_dir / name), img)  # re-encoded pixels: no metadata survives
        graders.add(r["grader"])
        dme_model = json.loads(r["payload_json"] or "{}").get("dme") or {}
        v = r["verdict"]
        out.append({
            "filename": f"images/{name}", "patient": pc,
            "visit": code(key, "visit", r["encounter_id"] or ""),
            "eye": r["eye"], "month": (r["created_utc"] or "")[:7], "device": r["device"],
            "verdict": v, "gradable": int(v != "UNGRADABLE"),
            "referable": {"REFER": 1, "ROUTINE": 0}.get(v),
            "grade": r["grade"], "dme": r["dme"],
            "findings": ";".join(json.loads(r["findings"])) if r["findings"] else "",
            "model_decision": r["decision"], "model_probability": r["probability"],
            "model_gradable": r["model_gradable"], "model_version": r["model_version"],
            "model_dme_suspected": dme_model.get("suspected"),
            "split": split_of(pc, locked_fraction),
        })
    df = pd.DataFrame(out)
    df.to_csv(out_dir / "labels.csv", index=False)

    def counts(col):
        return {str(k): int(n) for k, n in Counter(df[col].dropna()).items()}
    card = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "images": int(len(df)), "patients": int(df["patient"].nunique()),
        "visits": int(df["visit"].nunique()), "skipped_no_image": missing,
        "graders": len(graders), "months": sorted(df["month"].unique().tolist()),
        "verdicts": counts("verdict"), "split_images": counts("split"),
        "split_patients": {s: int(df.loc[df.split == s, "patient"].nunique())
                           for s in ("train", "locked_test")},
        "with_grade": int(df["grade"].notna().sum()), "with_dme": int(df["dme"].notna().sum()),
        "findings": dict(Counter(f for x in df["findings"] for f in x.split(";") if f)),
        "locked_test_fraction": locked_fraction,
        "pseudonymisation": "HMAC-SHA256 with secrets/pseudonym.key (not included)",
        "files_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(img_dir.iterdir())},
    }
    (out_dir / "dataset_card.json").write_text(json.dumps(card, indent=2))
    audit.log_event(con, actor, "export_snapshot", out_dir.name,
                    {k: card[k] for k in ("images", "patients", "verdicts", "split_images")})
    return card


def main(argv=None):
    ap = argparse.ArgumentParser(description="Export graded cases to a training snapshot.")
    ap.add_argument("--out", default=str(ROOT / "snapshots"))
    ap.add_argument("--locked-fraction", type=float, default=LOCKED_TEST_FRACTION,
                    help="share of PATIENTS permanently held out for testing")
    ap.add_argument("--actor", default=os.environ.get("USERNAME") or os.environ.get("USER") or "snapshot",
                    help="recorded in the audit log as who exported")
    args = ap.parse_args(argv)
    os.chdir(ROOT)

    import audit
    from storage import ImageStore
    con = audit.connect()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) / stamp
    card = export(con, ImageStore(), out, load_pseudonym_key(), args.actor, args.locked_fraction)
    print(f"  Snapshot {out}")
    print(f"  {card['images']} images, {card['patients']} patients, {card['graders']} grader(s)")
    print(f"  verdicts {card['verdicts']}; split {card['split_images']}")
    print("  Contains decrypted images: keep it on encrypted hospital storage.")


if __name__ == "__main__":
    main()
