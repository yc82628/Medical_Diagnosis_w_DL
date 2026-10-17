"""
Encrypted image store.

Retinal photographs are special-category health data (GDPR Art. 9): a fundus
image is biometric and can identify a person on its own. Nothing here writes
pixels to disk in plaintext. Every image -- the full-resolution upload Grad-CAM
needs, the review preview, the cached Grad-CAM overlay -- is encrypted with
Fernet (AES-128-CBC + HMAC-SHA256) under a key that never lives next to the data.

Key resolution, first match wins:
    DR_DATA_KEY        the key itself (urlsafe base64, 32 bytes) -- production:
                       inject it from the hospital's secret store
    DR_DATA_KEY_FILE   path to a file holding the key (default secrets/data.key)
    otherwise          a key is generated at DR_DATA_KEY_FILE, with a warning.
                       Convenient for development; in production a lost key
                       means every stored image is unreadable, so back it up.

Maintenance:
    python storage.py encrypt-legacy      encrypt images written by older versions
    python storage.py purge --days 90     delete images older than N days
                                          (the audit log keeps its hashes)
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

KINDS = ("raw", "preview", "gradcam")
DEFAULT_ROOT = Path("artifacts/images")
DEFAULT_KEY_FILE = Path("secrets/data.key")


def load_key(create: bool = True) -> bytes:
    env = os.environ.get("DR_DATA_KEY")
    if env:
        return env.encode()
    path = Path(os.environ.get("DR_DATA_KEY_FILE") or DEFAULT_KEY_FILE)
    if path.exists():
        return path.read_bytes().strip()
    if not create:
        raise FileNotFoundError(f"No data key at {path} and DR_DATA_KEY is not set.")
    path.parent.mkdir(parents=True, exist_ok=True)
    key = Fernet.generate_key()
    path.write_bytes(key)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass                                    # Windows: ACLs, not mode bits
    print(f"\n  ! Generated a new image-encryption key at {path.resolve()}."
          f"\n    Back it up: without it, stored images cannot be decrypted."
          f"\n    In production, supply DR_DATA_KEY from a secret store instead.\n")
    return key


class ImageStore:
    def __init__(self, root: Path | str = DEFAULT_ROOT, key: bytes | None = None):
        self.root = Path(root)
        self.fernet = Fernet(key or load_key())
        for k in KINDS:
            (self.root / k).mkdir(parents=True, exist_ok=True)

    def _path(self, case_id: str, kind: str) -> Path:
        if kind not in KINDS:
            raise ValueError(f"Unknown image kind {kind!r}")
        # case ids are generated hex; refuse anything else so a crafted id can
        # never become a path outside the store.
        if not case_id.isalnum():
            raise ValueError("Invalid case id")
        return self.root / kind / f"{case_id}.enc"

    def put(self, case_id: str, kind: str, data: bytes) -> None:
        p = self._path(case_id, kind)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(self.fernet.encrypt(data))
        os.replace(tmp, p)                      # never a half-written file

    def get(self, case_id: str, kind: str) -> bytes | None:
        p = self._path(case_id, kind)
        if not p.exists():
            return None
        try:
            return self.fernet.decrypt(p.read_bytes())
        except InvalidToken:
            raise RuntimeError(
                f"{p} could not be decrypted: wrong DR_DATA_KEY, or the file was "
                f"altered.") from None

    def exists(self, case_id: str, kind: str) -> bool:
        return self._path(case_id, kind).exists()

    def purge_older_than(self, days: float) -> int:
        """Retention: delete stored images older than `days`. Returns the count."""
        cutoff = time.time() - days * 86400
        n = 0
        for k in KINDS:
            for p in (self.root / k).glob("*.enc"):
                if p.stat().st_mtime < cutoff:
                    p.unlink()
                    n += 1
        return n

    def encrypt_legacy(self, previews_dir="artifacts/previews",
                       raw_dir="artifacts/raw") -> int:
        """
        Older versions wrote plaintext JPEG previews and PNG originals. Encrypt
        each into the store, then delete the plaintext.
        """
        n = 0
        for folder, kind, pattern in ((raw_dir, "raw", "*.png"),
                                      (previews_dir, "preview", "*.jpg")):
            for p in sorted(Path(folder).glob(pattern)):
                stem = p.stem
                k = "gradcam" if stem.endswith("_gradcam") else kind
                case_id = stem.removesuffix("_gradcam")
                if not case_id.isalnum():
                    continue
                self.put(case_id, k, p.read_bytes())
                p.unlink()
                n += 1
        return n


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Encrypted image store maintenance.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("encrypt-legacy", help="encrypt plaintext images from older versions")
    pg = sub.add_parser("purge", help="delete stored images older than N days")
    pg.add_argument("--days", type=float, required=True)
    args = ap.parse_args()

    store = ImageStore()
    if args.cmd == "encrypt-legacy":
        print(f"  Encrypted and removed {store.encrypt_legacy()} plaintext image(s).")
    else:
        print(f"  Deleted {store.purge_older_than(args.days)} image(s) older than "
              f"{args.days:g} days. Audit records are kept.")
