"""
Accounts, roles and sessions.

Every verdict in the audit log is attributed to the signed-in account, not to
a free-text name field -- an audit trail anyone can sign as is not an audit
trail.

Roles
    photographer   uploads images, sees the worklist; cannot record verdicts
    grader         everything a photographer can, plus verdicts, Grad-CAM,
                   FHIR export and monitoring
    admin          everything, plus audit-chain verification

Passwords are PBKDF2-HMAC-SHA256 (600k iterations, per-user salt). Sessions are
random 256-bit tokens; only their SHA-256 is stored. Five failed sign-ins lock
an account for 15 minutes.

Hospital single sign-on: put the service behind the hospital's identity proxy
(e.g. oauth2-proxy, Azure App Proxy, Shibboleth) and set DR_SSO_HEADER to the
header it sets (e.g. X-Forwarded-User). The named user must still exist here
and be active -- the role comes from this table, never from the header. Only
enable this when the service is reachable solely through that proxy;
otherwise anyone can send the header.

Account management (no web UI, by design: smaller attack surface):
    python auth.py create-user --username alice --role grader
    python auth.py list-users
    python auth.py disable-user --username alice
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

ROLES = ("photographer", "grader", "admin")
MIN_PASSWORD_LENGTH = 12
PBKDF2_ITERATIONS = 600_000
MAX_FAILED_ATTEMPTS = 5
LOCKOUT = timedelta(minutes=15)
SESSION_LIFETIME = timedelta(hours=8)       # one clinical shift
# Shared ward PCs get walked away from. A session unused for this long ends,
# whatever its 8-hour lifetime says. DR_IDLE_MINUTES overrides; 0 disables.
IDLE_DEFAULT_MINUTES = 15
_TOUCH_EVERY = timedelta(seconds=60)        # write last-seen at most once a minute

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username         TEXT PRIMARY KEY,
    password_hash    TEXT NOT NULL,
    role             TEXT NOT NULL,
    active           INTEGER NOT NULL DEFAULT 1,
    created_utc      TEXT NOT NULL,
    failed_attempts  INTEGER NOT NULL DEFAULT 0,
    locked_until     TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    token_sha256  TEXT PRIMARY KEY,
    username      TEXT NOT NULL,
    created_utc   TEXT NOT NULL,
    expires_utc   TEXT NOT NULL,
    revoked       INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (username) REFERENCES users(username)
);
"""


class AuthError(Exception):
    """Sign-in refused. The message is safe to show the user."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(d: datetime) -> str:
    return d.isoformat(timespec="seconds")


def init_schema(con) -> None:
    with con.lock:
        con.executescript(_SCHEMA)
        cols = {r["name"] for r in con.execute("PRAGMA table_info(sessions)")}
        if "last_seen_utc" not in cols:
            con.execute("ALTER TABLE sessions ADD COLUMN last_seen_utc TEXT")
        con.commit()


def idle_limit() -> timedelta | None:
    import os
    raw = os.environ.get("DR_IDLE_MINUTES", "").strip()
    minutes = float(raw) if raw else IDLE_DEFAULT_MINUTES
    return timedelta(minutes=minutes) if minutes > 0 else None


# -- passwords ---------------------------------------------------------------

def hash_password(password: str, iterations: int = PBKDF2_ITERATIONS) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    b64 = lambda b: base64.b64encode(b).decode()
    return f"pbkdf2_sha256${iterations}${b64(salt)}${b64(dk)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, it, salt, dk = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        calc = hashlib.pbkdf2_hmac("sha256", password.encode(),
                                   base64.b64decode(salt), int(it))
        return hmac.compare_digest(calc, base64.b64decode(dk))
    except (ValueError, TypeError):
        return False


# -- users -------------------------------------------------------------------

def create_user(con, username: str, password: str, role: str,
                iterations: int = PBKDF2_ITERATIONS) -> None:
    username = username.strip()
    if not username or not username.replace(".", "").replace("-", "").replace("_", "").isalnum():
        raise ValueError("Username may contain letters, digits, '.', '-' and '_' only.")
    if role not in ROLES:
        raise ValueError(f"Role must be one of {ROLES}.")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    with con.lock:
        if con.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
            raise ValueError(f"User {username!r} already exists.")
        con.execute("INSERT INTO users (username, password_hash, role, created_utc) "
                    "VALUES (?,?,?,?)",
                    (username, hash_password(password, iterations), role, _iso(_now())))
        con.commit()


def set_active(con, username: str, active: bool) -> None:
    with con.lock:
        cur = con.execute("UPDATE users SET active=? WHERE username=?",
                          (int(active), username))
        if not active:
            con.execute("UPDATE sessions SET revoked=1 WHERE username=?", (username,))
        con.commit()
    if cur.rowcount == 0:
        raise KeyError(username)


def list_users(con) -> list[dict]:
    with con.lock:
        return [dict(r) for r in con.execute(
            "SELECT username, role, active, created_utc, locked_until FROM users "
            "ORDER BY username")]


def get_user(con, username: str) -> dict | None:
    with con.lock:
        r = con.execute("SELECT username, role, active FROM users WHERE username=?",
                        (username,)).fetchone()
    return dict(r) if r else None


def count_users(con) -> int:
    with con.lock:
        return con.execute("SELECT COUNT(*) FROM users").fetchone()[0]


def authenticate(con, username: str, password: str) -> dict:
    """Returns the user, or raises AuthError. Same message for unknown user and
    wrong password, so the endpoint does not reveal which accounts exist."""
    generic = "Incorrect username or password."
    with con.lock:
        row = con.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if row is None:
            # Burn comparable time so timing does not reveal unknown users.
            verify_password(password, hash_password("x", 10_000))
            raise AuthError(generic)
        if row["locked_until"] and datetime.fromisoformat(row["locked_until"]) > _now():
            raise AuthError("Account temporarily locked after repeated failed "
                            "sign-ins. Try again later or contact an administrator.")
        if not verify_password(password, row["password_hash"]):
            n = row["failed_attempts"] + 1
            locked = _iso(_now() + LOCKOUT) if n >= MAX_FAILED_ATTEMPTS else None
            con.execute("UPDATE users SET failed_attempts=?, locked_until=? "
                        "WHERE username=?",
                        (0 if locked else n, locked, username))
            con.commit()
            raise AuthError(generic)
        if not row["active"]:
            raise AuthError("This account is disabled.")
        con.execute("UPDATE users SET failed_attempts=0, locked_until=NULL "
                    "WHERE username=?", (username,))
        con.commit()
        return {"username": row["username"], "role": row["role"]}


# -- sessions ----------------------------------------------------------------

def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(con, username: str) -> str:
    token = secrets.token_urlsafe(32)
    now = _now()
    with con.lock:
        con.execute("INSERT INTO sessions (token_sha256, username, created_utc, "
                    "expires_utc) VALUES (?,?,?,?)",
                    (_token_hash(token), username, _iso(now), _iso(now + SESSION_LIFETIME)))
        con.commit()
    return token


def resolve_session(con, token: str | None, idle: timedelta | None = None,
                    now: datetime | None = None) -> dict | None:
    """
    The signed-in user, or None. A session ends when revoked, expired, its
    user is disabled, or -- with `idle` -- after `idle` without a request.
    `idle` defaults to idle_limit() (DR_IDLE_MINUTES, 15 minutes).
    """
    if not token:
        return None
    idle = idle_limit() if idle is None else idle
    now = now or _now()
    th = _token_hash(token)
    with con.lock:
        r = con.execute(
            """SELECT u.username, u.role, u.active, s.created_utc, s.expires_utc,
                      s.revoked, s.last_seen_utc
                 FROM sessions s JOIN users u ON u.username = s.username
                WHERE s.token_sha256 = ?""", (th,)).fetchone()
        if (r is None or r["revoked"] or not r["active"]
                or datetime.fromisoformat(r["expires_utc"]) <= now):
            return None
        seen = datetime.fromisoformat(r["last_seen_utc"] or r["created_utc"])
        if idle and now - seen > idle:
            con.execute("UPDATE sessions SET revoked=1 WHERE token_sha256=?", (th,))
            con.commit()
            return None
        if now - seen >= _TOUCH_EVERY:
            con.execute("UPDATE sessions SET last_seen_utc=? WHERE token_sha256=?",
                        (_iso(now), th))
            con.commit()
    return {"username": r["username"], "role": r["role"]}


def revoke_session(con, token: str | None) -> None:
    if not token:
        return
    with con.lock:
        con.execute("UPDATE sessions SET revoked=1 WHERE token_sha256=?",
                    (_token_hash(token),))
        con.commit()


if __name__ == "__main__":
    import argparse
    import getpass

    import audit

    ap = argparse.ArgumentParser(description="Manage platform accounts.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create-user")
    c.add_argument("--username", required=True)
    c.add_argument("--role", required=True, choices=ROLES)
    sub.add_parser("list-users")
    d = sub.add_parser("disable-user")
    d.add_argument("--username", required=True)
    e = sub.add_parser("enable-user")
    e.add_argument("--username", required=True)
    args = ap.parse_args()

    con = audit.connect()
    init_schema(con)
    if args.cmd == "create-user":
        pw = getpass.getpass(f"Password for {args.username} (min {MIN_PASSWORD_LENGTH} chars): ")
        if pw != getpass.getpass("Repeat password: "):
            raise SystemExit("  Passwords do not match.")
        try:
            create_user(con, args.username, pw, args.role)
        except ValueError as err:
            raise SystemExit(f"  {err}")
        audit.log_event(con, "cli", "user_created",
                        args.username, {"role": args.role})
        print(f"  Created {args.role} account {args.username!r}.")
    elif args.cmd == "list-users":
        for u in list_users(con):
            state = "active" if u["active"] else "disabled"
            print(f"  {u['username']:<20} {u['role']:<13} {state}")
    else:
        active = args.cmd == "enable-user"
        try:
            set_active(con, args.username, active)
        except KeyError:
            raise SystemExit(f"  No user {args.username!r}.")
        audit.log_event(con, "cli", "user_enabled" if active else "user_disabled",
                        args.username, {})
        print(f"  {'Enabled' if active else 'Disabled'} {args.username!r}.")
