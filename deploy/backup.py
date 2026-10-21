"""
Back up, verify and restore the platform's data.

    python deploy/backup.py                       # back up to backups/<timestamp>/
    python deploy/backup.py --to E:\\retina-backups --keep 14
    python deploy/backup.py verify backups/20261021-0200
    python deploy/backup.py restore backups/20261021-0200    (server stopped)

What a backup holds:
  audit.db        the audit log, accounts and case records -- copied with
                  SQLite's online backup, so it is consistent while the
                  server is running
  images/         the stored images, still ENCRYPTED (they stay unreadable
                  without the key)
  checkpoints/    the trained models
  artifacts/*.json  evaluation evidence for the model card and gate
  manifest.json   SHA-256 of every file, plus the audit-chain head hash

What it deliberately does NOT hold: secrets/data.key. A backup and its key
together are every patient image in readable form, so keep them apart: the
key in the hospital's secret store or a safe, the backups on a separate drive.
Verification still proves the key on this machine opens the backed-up images.

Schedule it daily, e.g. Windows Task Scheduler:
    .venv\\Scripts\\python.exe deploy\\backup.py --to E:\\retina-backups --keep 14
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from contextlib import closing
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DB = Path("artifacts/audit.db")
IMAGES = Path("artifacts/images")
CHECKPOINTS = Path("checkpoints")
EVIDENCE = Path("artifacts")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _files(base: Path) -> list[Path]:
    return sorted(p for p in base.rglob("*") if p.is_file())


def backup(root: Path, dest_parent: Path, keep: int | None = None) -> Path:
    import audit
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    dest = dest_parent / stamp
    dest.mkdir(parents=True, exist_ok=False)

    src_db = root / DB
    if not src_db.exists():
        raise SystemExit(f"  No database at {src_db}: nothing to back up.")
    # closing(), not a bare `with`: sqlite3's context manager only commits, it
    # does not close, and on Windows an open handle locks the file.
    with closing(sqlite3.connect(str(src_db))) as src, \
            closing(sqlite3.connect(str(dest / "audit.db"))) as dst:
        src.backup(dst)                     # consistent snapshot, server may be running

    if (root / IMAGES).exists():
        shutil.copytree(root / IMAGES, dest / "images")
    if (root / CHECKPOINTS).exists():
        (dest / "checkpoints").mkdir()
        for p in (root / CHECKPOINTS).glob("*.pt"):
            shutil.copy2(p, dest / "checkpoints" / p.name)
    (dest / "evidence").mkdir()
    for p in (root / EVIDENCE).glob("*.json"):
        shutil.copy2(p, dest / "evidence" / p.name)

    con = audit.connect(dest / "audit.db")
    chain = audit.verify_chain(con)
    con.close()
    manifest = {
        "created_utc": stamp, "source": str(root),
        "audit_chain": {k: chain[k] for k in ("ok", "entries", "head_hash")},
        "files": {p.relative_to(dest).as_posix(): _sha(p) for p in _files(dest)},
        "note": "secrets/data.key is NOT included; images are encrypted with it.",
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if keep:
        olds = sorted(d for d in dest_parent.iterdir()
                      if d.is_dir() and (d / "manifest.json").exists())
        for old in olds[:-keep]:
            shutil.rmtree(old)
    return dest


def verify(dest: Path, key: bytes | None = None) -> dict:
    """Hashes, audit chain, and that the current key decrypts the images."""
    import audit
    from storage import ImageStore, load_key
    m = json.loads((dest / "manifest.json").read_text())
    problems = []
    for rel, digest in m["files"].items():
        p = dest / rel
        if not p.exists():
            problems.append(f"missing: {rel}")
        elif _sha(p) != digest:
            problems.append(f"changed since backup: {rel}")
    con = audit.connect(dest / "audit.db")
    chain = audit.verify_chain(con)
    con.close()
    if not chain["ok"]:
        problems += [f"audit chain: {p}" for p in chain["problems"][:10]]
    encs = sorted((dest / "images").rglob("*.enc")) if (dest / "images").exists() else []
    decrypted = 0
    if encs:
        store = ImageStore(dest / "images", key or load_key(create=False))
        for p in encs[:5]:
            try:
                if store.get(p.stem, p.parent.name):
                    decrypted += 1
            except RuntimeError:
                problems.append(f"cannot decrypt {p.parent.name}/{p.name} with this key")
    return {"ok": not problems, "problems": problems, "files": len(m["files"]),
            "audit_entries": chain["entries"], "images": len(encs),
            "sample_decrypted": decrypted}


def _server_running(port: int) -> bool:
    import socket
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def restore(dest: Path, root: Path, port: int = 8443) -> Path:
    if _server_running(port):
        raise SystemExit(f"  The server is running on port {port}. Stop it before restoring.")
    v = verify(dest)
    if not v["ok"]:
        raise SystemExit("  Backup failed verification; not restoring:\n    "
                         + "\n    ".join(v["problems"]))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    aside = root / f"artifacts.before-restore-{stamp}"
    if (root / "artifacts").exists():
        shutil.move(str(root / "artifacts"), str(aside))     # nothing is deleted
    (root / "artifacts").mkdir()
    shutil.copy2(dest / "audit.db", root / DB)
    if (dest / "images").exists():
        shutil.copytree(dest / "images", root / IMAGES)
    for p in (dest / "evidence").glob("*.json"):
        shutil.copy2(p, root / EVIDENCE / p.name)
    if (dest / "checkpoints").exists():
        (root / CHECKPOINTS).mkdir(exist_ok=True)
        for p in (dest / "checkpoints").glob("*.pt"):
            shutil.copy2(p, root / CHECKPOINTS / p.name)
    return aside


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("action", nargs="?", default="backup", choices=["backup", "verify", "restore"])
    ap.add_argument("path", nargs="?", help="backup folder (verify / restore)")
    ap.add_argument("--to", default=str(ROOT / "backups"), help="where backups go")
    ap.add_argument("--keep", type=int, help="keep only the newest N backups")
    ap.add_argument("--port", type=int, default=8443)
    args = ap.parse_args(argv)

    import os
    os.chdir(ROOT)
    if args.action == "backup":
        dest = backup(ROOT, Path(args.to), args.keep)
        v = verify(dest)
        print(f"  Backup written to {dest}")
        print(f"  {v['files']} files, {v['audit_entries']} audit entries, {v['images']} "
              f"encrypted images; {v['sample_decrypted']} sample image(s) decrypted OK")
        if not v["ok"]:
            print("  ! VERIFICATION FAILED:\n    " + "\n    ".join(v["problems"]))
            return 1
        print("  Verified. The encryption key is not in the backup -- keep it separately.")
        return 0
    if not args.path:
        ap.error(f"{args.action} needs the backup folder")
    if args.action == "verify":
        v = verify(Path(args.path))
        print(json.dumps(v, indent=2))
        return 0 if v["ok"] else 1
    aside = restore(Path(args.path), ROOT, args.port)
    print(f"  Restored from {args.path}. The previous data is kept in {aside}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
