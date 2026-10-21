"""
First-run setup for serving the platform on a hospital network.

    .venv\\Scripts\\python deploy\\setup.py

Checks the GPU and models, creates the image-encryption key and the first
admin account, asks which network ranges staff devices use, creates the HTTPS
certificates, and writes deploy/server.env for start_server.py. Safe to re-run:
existing keys, accounts and the local CA are kept.

Scripted install (no prompts):
    set DR_SETUP_ADMIN_PASSWORD=...        (read once, never stored)
    python deploy\\setup.py --non-interactive --admin-user it.admin ^
        --networks 192.168.10.0/24 --name retina.ward3.local
"""
from __future__ import annotations

import argparse
import getpass
import ipaddress
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

ENV_FILE = ROOT / "deploy" / "server.env"
DEFAULT_PORT = 8443


def say(ok: bool | None, msg: str, hint: str = "") -> None:
    mark = {True: " OK ", False: "FAIL", None: "WARN"}[ok]
    print(f"  [{mark}] {msg}")
    if hint:
        for line in hint.splitlines():
            print(f"         {line}")


def check_gpu() -> None:
    try:
        import torch
    except ImportError:
        say(False, "PyTorch is not installed", "pip install -r requirements-gpu.txt, then "
            "pip install -r requirements.txt -r requirements-serve.txt")
        raise SystemExit(1)
    if torch.cuda.is_available():
        say(True, f"GPU: {torch.cuda.get_device_name(0)} (torch {torch.__version__})")
    else:
        say(None, f"No CUDA GPU visible (torch {torch.__version__})",
            "The service still works on the CPU, a few seconds per image instead of\n"
            "a fraction of one. On an NVIDIA laptop, see SETUP.md 'Common errors'.")


def check_models() -> None:
    from predictor import build_predictor
    p, note = build_predictor()
    if p.is_demo:
        say(None, "No validated model loaded: the service will run in DEMO mode", note)
    elif not p.validated:
        say(None, "Model loaded WITHOUT passing the acceptance gate", note)
    else:
        say(True, f"Model {p.model_version} passed the acceptance gate")
    try:
        from quality_model import default_quality_note
        say(True, default_quality_note())
    except Exception as e:                                  # noqa: BLE001
        say(None, f"Quality model: {e}")


def check_key() -> None:
    from storage import load_key
    if os.environ.get("DR_DATA_KEY"):
        say(True, "Image-encryption key supplied by DR_DATA_KEY")
        return
    path = Path(os.environ.get("DR_DATA_KEY_FILE") or "secrets/data.key")
    existed = path.exists()
    load_key(create=True)
    say(True, f"Image-encryption key {'found' if existed else 'CREATED'} at {path}",
        "" if existed else "BACK IT UP NOW, somewhere other than this laptop.\n"
                           "Without it, every stored image is unreadable.")


def ensure_admin(args) -> None:
    import audit
    import auth
    con = audit.connect()
    auth.init_schema(con)
    if auth.count_users(con):
        say(True, f"{auth.count_users(con)} account(s) already exist; none created")
        return
    user = args.admin_user or ("" if args.non_interactive else
                               input("  First admin username: ").strip())
    if not user:
        raise SystemExit("  An admin username is required (--admin-user).")
    if args.non_interactive:
        pw = os.environ.get("DR_SETUP_ADMIN_PASSWORD", "")
        if not pw:
            raise SystemExit("  --non-interactive needs DR_SETUP_ADMIN_PASSWORD.")
    else:
        pw = getpass.getpass(f"  Password for {user} (min {auth.MIN_PASSWORD_LENGTH} chars): ")
        if pw != getpass.getpass("  Repeat password: "):
            raise SystemExit("  Passwords do not match.")
    try:
        auth.create_user(con, user, pw, "admin")
    except ValueError as e:
        raise SystemExit(f"  {e}")
    audit.log_event(con, "setup", "user_created", user, {"role": "admin"})
    say(True, f"Created admin account {user!r}",
        "Create staff accounts with:  python auth.py create-user --username NAME "
        "--role grader")


def choose_networks(args, ips: list[str]) -> str:
    if args.networks is not None:
        spec = args.networks
    elif args.non_interactive:
        raise SystemExit("  --non-interactive needs --networks (e.g. 192.168.10.0/24).")
    else:
        guess = str(ipaddress.ip_network(f"{ips[0]}/24", strict=False)) if ips else ""
        print("\n  Which network ranges do STAFF devices use? Only these can open the site.")
        print("  Ask hospital IT for the staff Wi-Fi / VLAN range. Do NOT include guest Wi-Fi.")
        ans = input(f"  Staff networks [{guess}]: ").strip()
        spec = ans or guess
    from service import parse_networks
    nets = parse_networks(spec)              # SystemExit on a typo
    if not nets and not args.allow_any_network:
        raise SystemExit("  Refusing to serve with no network restriction. Give the staff "
                         "range, or --allow-any-network for a closed test network.")
    for n in nets:
        if n.num_addresses > 2 ** 24:
            say(None, f"{n} is very broad ({n.num_addresses:,} addresses)",
                "Check it does not also cover guest or public networks.")
    covered = [ip for ip in ips if any(ipaddress.ip_address(ip) in n for n in nets)]
    if nets and ips and not covered:
        say(None, f"This laptop's address ({', '.join(ips)}) is outside {spec}",
            "Staff on that range can still connect if they can route to it; check with IT.")
    return ",".join(str(n) for n in nets)


def main(argv=None):
    ap = argparse.ArgumentParser(description="First-run setup for a hospital network.")
    ap.add_argument("--networks", help="staff network ranges, e.g. 192.168.10.0/24")
    ap.add_argument("--allow-any-network", action="store_true",
                    help="permit no network restriction (closed test networks only)")
    ap.add_argument("--name", action="append", default=[],
                    help="extra host name staff will use (repeatable)")
    ap.add_argument("--ip", action="append", default=[], help="extra IP for the certificate")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--idle-minutes", type=float, default=15,
                    help="sign out after this long without activity (0 = never)")
    ap.add_argument("--admin-user")
    ap.add_argument("--non-interactive", action="store_true")
    ap.add_argument("--skip-model-check", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    print("=" * 66)
    print("  Retinal Screening -- hospital network setup")
    print("  RESEARCH SOFTWARE -- NOT A MEDICAL DEVICE")
    print("=" * 66)

    print("\n[1/5] Hardware and models")
    check_gpu()
    if not args.skip_model_check:
        check_models()

    print("\n[2/5] Image encryption key")
    check_key()

    print("\n[3/5] Admin account")
    ensure_admin(args)

    print("\n[4/5] Staff network")
    from make_certs import lan_ips, make
    ips = lan_ips()
    networks = choose_networks(args, ips)
    say(True, f"Allowed networks: {networks or 'ANY (test network only)'}")

    print("\n[5/5] HTTPS certificates")
    host = socket.gethostname()
    r = make([host, f"{host}.local", *args.name], ips + args.ip)
    say(True, f"{'New local CA created' if r['new_ca'] else 'Existing local CA reused'}; "
              f"server certificate valid until {r['expires']}",
        f"for {', '.join(r['names'] + r['ips'])}")

    certs = Path("deploy/certs")
    ENV_FILE.write_text(
        "# Written by deploy/setup.py. Read by deploy/start_server.py.\n"
        "# A real environment variable overrides anything here.\n"
        "DR_HOST=0.0.0.0\n"
        f"DR_PORT={args.port}\n"
        f"DR_ALLOWED_NETWORKS={networks}\n"
        f"DR_IDLE_MINUTES={args.idle_minutes:g}\n"
        f"DR_SSL_CERT={(certs / 'server.crt').as_posix()}\n"
        f"DR_SSL_KEY={(certs / 'server.key').as_posix()}\n",
        encoding="utf-8")

    print("\n" + "=" * 66)
    print("  Setup complete. Start the server with:   deploy\\start_server.bat")
    print("\n  Staff open:")
    for target in [f"{host}.local", *args.name, *ips]:
        print(f"      https://{target}:{args.port}")
    print("\n  Before staff use it (details in DEPLOY.md):")
    if r["new_ca"]:
        print("    1. Install deploy\\certs\\ca.crt on every staff device.")
        print("    2. Move deploy\\certs\\ca.key to offline storage.")
    print("    -  Ask IT to run deploy\\install_windows.ps1 as administrator")
    print("       (firewall rule for the staff network, start on boot, no sleep).")
    print("    -  Back up secrets\\data.key off this laptop; schedule deploy\\backup.py.")
    print("=" * 66)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
