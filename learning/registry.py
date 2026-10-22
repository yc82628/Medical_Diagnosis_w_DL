"""
Model registry: every version kept, nothing promoted without a named approver.

    python learning/registry.py list quality
    python learning/registry.py promote quality 20261022-153000-1a2b3c4d --approved-by "Dr A. Example" --reason "..."
    python learning/registry.py rollback quality --approved-by "..." --reason "..."

The learning loop never changes the served model by itself. A retraining run
(e.g. learning/retrain_quality.py) REGISTERS a candidate with its evaluation
report. A person then promotes it:

  - promotion refuses a candidate whose report says it is worse than the
    current model on the site's locked test set, unless --override-reason is
    given (recorded);
  - the current model is archived first, so rollback is always possible;
  - the checkpoint and its evaluation file are installed together, so the
    model card and the service's acceptance checks see matching evidence;
  - every registration, promotion and rollback is written to the
    hash-chained audit log with the approver's name.

The server loads models at start-up: restart it after a promotion.

Layout: checkpoints/registry/<kind>/<version>/{model.pt, eval.json, meta.json}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REGISTRY = Path("checkpoints/registry")
# kind -> (served checkpoint, its evaluation file)
KINDS = {
    "quality": (Path("checkpoints/quality_model.pt"), Path("artifacts/quality_model_eval.json")),
    "dr": (Path("checkpoints/best_model.pt"), Path("artifacts/evaluation.json")),
    "dme": (Path("checkpoints/dme_model.pt"), Path("artifacts/dme_eval.json")),
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _meta(d: Path) -> dict:
    return json.loads((d / "meta.json").read_text())


def _write_meta(d: Path, m: dict) -> None:
    (d / "meta.json").write_text(json.dumps(m, indent=2))


def _log(con, actor: str, action: str, kind: str, version: str, detail: dict) -> None:
    if con is not None:
        import audit
        audit.log_event(con, actor, action, f"{kind}:{version}", detail)


def versions(kind: str, root: Path = REGISTRY) -> list[dict]:
    base = root / kind
    if not base.exists():
        return []
    out = []
    for d in sorted(base.iterdir()):
        if (d / "meta.json").exists():
            out.append({"version": d.name, "dir": d, **_meta(d)})
    return out


def register(kind: str, checkpoint: Path, eval_file: Path | None, *, source: str,
             summary: dict | None = None, comparison: dict | None = None,
             root: Path = REGISTRY, con=None, actor: str = "training") -> str:
    """Store a model version as a CANDIDATE. Returns its version id."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {list(KINDS)}")
    sha = _sha(checkpoint)
    version = f"{_now()}-{sha[:8]}"
    d = root / kind / version
    d.mkdir(parents=True)
    shutil.copy2(checkpoint, d / "model.pt")
    if eval_file and Path(eval_file).exists():
        shutil.copy2(eval_file, d / "eval.json")
    _write_meta(d, {"kind": kind, "status": "candidate", "sha256": sha, "source": source,
                    "registered_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "summary": summary or {}, "comparison": comparison or {}, "history": []})
    _log(con, actor, "model_registered", kind, version,
         {"sha256": sha, "source": source, "comparison": comparison or {}})
    return version


def _archive_current(kind: str, root: Path, con, actor: str, ck: Path, ev: Path) -> str | None:
    """Make sure the model being replaced is in the registry, marked retired."""
    if not ck.exists():
        return None
    sha = _sha(ck)
    for v in versions(kind, root):
        if v["sha256"] == sha:
            return v["version"]
    return register(kind, ck, ev if ev.exists() else None, source="archived on replacement",
                    root=root, con=con, actor=actor)


def promote(kind: str, version: str, approved_by: str, reason: str, *,
            override_reason: str | None = None, root: Path = REGISTRY, con=None,
            served: dict | None = None) -> dict:
    if not approved_by.strip() or not reason.strip():
        raise SystemExit("  Promotion needs --approved-by (a named person) and --reason.")
    d = root / kind / version
    if not (d / "meta.json").exists():
        raise SystemExit(f"  No version {version} for {kind}.")
    m = _meta(d)
    comp = m.get("comparison") or {}
    if comp and comp.get("candidate_not_worse") is False and not override_reason:
        raise SystemExit("  The evaluation report says this candidate is WORSE than the current "
                         "model on the locked test set:\n    " + "; ".join(comp.get("why", []))
                         + "\n  Promote anyway only with --override-reason (it is recorded).")
    ck, ev = (served or {}).get("checkpoint", KINDS[kind][0]), (served or {}).get("eval", KINDS[kind][1])
    previous = _archive_current(kind, root, con, approved_by, ck, ev)
    ck.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(d / "model.pt", ck)
    if (d / "eval.json").exists():
        ev.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(d / "eval.json", ev)
    if _sha(ck) != m["sha256"]:
        raise SystemExit("  Installed checkpoint does not match the registered hash.")
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for v in versions(kind, root):
        vm = _meta(v["dir"])
        if v["version"] == version:
            vm["status"] = "active"
            vm["history"].append({"utc": stamp, "event": "promoted", "by": approved_by,
                                  "reason": reason, "override": override_reason,
                                  "replaced": previous})
        elif vm["status"] == "active" or v["version"] == previous:
            vm["status"] = "retired"
            vm["history"].append({"utc": stamp, "event": "retired", "by": approved_by,
                                  "replaced_by": version})
        _write_meta(v["dir"], vm)
    _log(con, approved_by, "model_promoted", kind, version,
         {"reason": reason, "override": override_reason, "replaced": previous})
    return {"promoted": version, "replaced": previous}


def rollback(kind: str, approved_by: str, reason: str, *, root: Path = REGISTRY,
             con=None, served: dict | None = None) -> dict:
    """Re-promote the version that was active before the current one."""
    vs = versions(kind, root)
    active = next((v for v in vs if v["status"] == "active"), None)
    if active is None:
        raise SystemExit("  No active registered version to roll back from.")
    prev = next((h.get("replaced") for h in reversed(_meta(active["dir"])["history"])
                 if h["event"] == "promoted"), None)
    if not prev:
        raise SystemExit("  The active version replaced nothing: nothing to roll back to.")
    r = promote(kind, prev, approved_by, f"ROLLBACK: {reason}", root=root, con=con,
                served=served, override_reason="rollback")
    _log(con, approved_by, "model_rolled_back", kind, prev, {"from": active["version"]})
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description="Model registry: list, promote, roll back.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("list", "promote", "rollback"):
        p = sub.add_parser(name)
        p.add_argument("kind", choices=list(KINDS))
        if name == "promote":
            p.add_argument("version")
        if name != "list":
            p.add_argument("--approved-by", required=True)
            p.add_argument("--reason", required=True)
        if name == "promote":
            p.add_argument("--override-reason")
    args = ap.parse_args(argv)
    os.chdir(ROOT)

    if args.cmd == "list":
        vs = versions(args.kind)
        if not vs:
            print(f"  No registered {args.kind} models yet.")
        for v in vs:
            c = v.get("comparison") or {}
            verdict = ("" if not c else "  better/equal on locked test" if c.get("candidate_not_worse")
                       else "  WORSE on locked test")
            print(f"  {v['version']}  {v['status']:<9}  {v['source']}{verdict}")
        return
    import audit
    con = audit.connect()
    if args.cmd == "promote":
        r = promote(args.kind, args.version, args.approved_by, args.reason,
                    override_reason=args.override_reason, con=con)
    else:
        r = rollback(args.kind, args.approved_by, args.reason, con=con)
    print(f"  {args.kind}: {r['promoted']} is now active (replaced {r['replaced']}).")
    print("  Restart the server to load it.")


if __name__ == "__main__":
    main()
