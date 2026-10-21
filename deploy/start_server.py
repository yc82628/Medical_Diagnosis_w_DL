"""
Start the platform for the hospital network, with the settings deploy/setup.py
wrote to deploy/server.env: HTTPS, the staff-network allow-list, the
inactivity sign-out.

    .venv\\Scripts\\python deploy\\start_server.py      (or deploy\\start_server.bat)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / "deploy" / "server.env"
REQUIRED = ("DR_HOST", "DR_PORT", "DR_SSL_CERT", "DR_SSL_KEY")


def load_env(path: Path = ENV_FILE) -> dict:
    if not path.exists():
        raise SystemExit(f"  {path} not found. Run deploy/setup.py first.")
    out = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    missing = [k for k in REQUIRED if not out.get(k)]
    if missing:
        raise SystemExit(f"  {path} is missing {missing}. Re-run deploy/setup.py.")
    return out


def main() -> int:
    env = dict(os.environ)
    for k, v in load_env().items():
        env.setdefault(k, v)                 # a real environment variable wins
    for k in ("DR_SSL_CERT", "DR_SSL_KEY"):
        if not (ROOT / env[k]).exists():
            raise SystemExit(f"  {env[k]} not found. Re-run deploy/setup.py.")
    if not env.get("DR_ALLOWED_NETWORKS"):
        print("  ! DR_ALLOWED_NETWORKS is empty: ANY device that can reach this "
              "machine can open the sign-in page.")
    cmd = [sys.executable, "service.py", "--host", env["DR_HOST"], "--port", env["DR_PORT"],
           "--ssl-certfile", env["DR_SSL_CERT"], "--ssl-keyfile", env["DR_SSL_KEY"],
           "--i-understand-this-is-not-a-medical-device"]
    return subprocess.call(cmd, cwd=ROOT, env=env)


if __name__ == "__main__":
    sys.exit(main())
