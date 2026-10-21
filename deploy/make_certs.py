"""
HTTPS certificates for running the platform on a hospital network.

A hospital Wi-Fi has no public domain name, so public certificate authorities
cannot issue a certificate for it. This creates a small private one instead:

  deploy/certs/ca.crt       the "Retinal Screening local CA" -- install it ONCE
                            on each staff device (or push it via hospital IT's
                            device management). Not secret.
  deploy/certs/ca.key       the CA's private key. SECRET: whoever holds it can
                            impersonate the server. Move it off the server to
                            offline storage after setup; it is only needed to
                            issue a new server certificate.
  deploy/certs/server.crt   the server certificate, valid for this machine's
  deploy/certs/server.key   names and LAN addresses. server.key is secret.

Re-running reuses an existing CA, so staff devices keep trusting the server
when its certificate is renewed or its IP address changes.

If hospital IT already runs a certificate authority, use a certificate from
them instead and skip this script (see DEPLOY.md).

    python deploy/make_certs.py                       # names + IPs detected
    python deploy/make_certs.py --name retina.ward3.local --ip 192.168.10.20
"""
from __future__ import annotations

import argparse
import datetime as dt
import ipaddress
import os
import socket
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

CERT_DIR = Path(__file__).resolve().parent / "certs"
CA_NAME = "Retinal Screening local CA"
CA_YEARS = 10
# Browsers (Safari in particular) reject server certificates valid for longer
# than 825 days.
SERVER_DAYS = 825


def lan_ips() -> list[str]:
    """This machine's LAN addresses (best effort, no extra dependencies)."""
    ips = set()
    try:                                   # the interface used for the default route
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith(("127.", "169.254.")))


def _write_private(path: Path, key) -> None:
    path.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                       serialization.PrivateFormat.PKCS8,
                                       serialization.NoEncryption()))
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _load_or_make_ca(cert_dir: Path, now):
    crt, key = cert_dir / "ca.crt", cert_dir / "ca.key"
    if crt.exists() and key.exists():
        return (x509.load_pem_x509_certificate(crt.read_bytes()),
                serialization.load_pem_private_key(key.read_bytes(), None), False)
    if crt.exists() and not key.exists():
        raise SystemExit(
            f"  {crt} exists but its private key does not (moved offline?).\n"
            f"  Put ca.key back in {cert_dir} to issue a new server certificate.\n"
            f"  Creating a new CA instead would mean re-installing ca.crt on every "
            f"staff device.")
    ca_key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, CA_NAME),
                      x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Retinal Screening")])
    ca = (x509.CertificateBuilder()
          .subject_name(name).issuer_name(name)
          .public_key(ca_key.public_key())
          .serial_number(x509.random_serial_number())
          .not_valid_before(now - dt.timedelta(minutes=5))
          .not_valid_after(now + dt.timedelta(days=365 * CA_YEARS))
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
          .add_extension(x509.KeyUsage(digital_signature=True, key_cert_sign=True,
                                       crl_sign=True, content_commitment=False,
                                       key_encipherment=False, data_encipherment=False,
                                       key_agreement=False, encipher_only=False,
                                       decipher_only=False), critical=True)
          .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()),
                         critical=False)
          .sign(ca_key, hashes.SHA256()))
    cert_dir.mkdir(parents=True, exist_ok=True)
    crt.write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    _write_private(key, ca_key)
    return ca, ca_key, True


def make(names: list[str], ips: list[str], cert_dir: Path = CERT_DIR) -> dict:
    now = dt.datetime.now(dt.timezone.utc)
    ca, ca_key, new_ca = _load_or_make_ca(cert_dir, now)

    names = sorted({n.strip().lower() for n in names if n and n.strip()})
    addrs = sorted({ipaddress.ip_address(i) for i in ips if i}, key=str)
    if not names and not addrs:
        raise SystemExit("  No host name or IP address to put in the certificate.")
    sans = [x509.DNSName(n) for n in names] + [x509.IPAddress(a) for a in addrs]

    key = ec.generate_private_key(ec.SECP256R1())
    cn = names[0] if names else str(addrs[0])
    cert = (x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)]))
            .issuer_name(ca.subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(minutes=5))
            .not_valid_after(now + dt.timedelta(days=SERVER_DAYS))
            .add_extension(x509.SubjectAlternativeName(sans), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
                           critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(
                ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256()))
    (cert_dir / "server.crt").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    _write_private(cert_dir / "server.key", key)
    return {"cert_dir": cert_dir, "new_ca": new_ca, "names": names,
            "ips": [str(a) for a in addrs],
            "expires": cert.not_valid_after_utc.date().isoformat()}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Create HTTPS certificates for the LAN.")
    ap.add_argument("--name", action="append", default=[],
                    help="extra host name staff will type (repeatable)")
    ap.add_argument("--ip", action="append", default=[],
                    help="extra IP address (repeatable); detected ones are added")
    ap.add_argument("--no-detect", action="store_true",
                    help="use only --name / --ip, do not detect this machine's")
    ap.add_argument("--out", default=str(CERT_DIR))
    args = ap.parse_args(argv)

    names, ips = list(args.name), list(args.ip)
    if not args.no_detect:
        host = socket.gethostname()
        names += [host, f"{host}.local"]
        ips += lan_ips()
    r = make(names, ips, Path(args.out))
    print(f"  {'Created a NEW local CA' if r['new_ca'] else 'Reused the existing local CA'}"
          f" in {r['cert_dir']}")
    print(f"  Server certificate for: {', '.join(r['names'] + r['ips'])}")
    print(f"  Valid until {r['expires']}")
    if r["new_ca"]:
        print("\n  Install deploy/certs/ca.crt on every staff device (see DEPLOY.md).")
        print("  Then move deploy/certs/ca.key to offline storage.")
    return r


if __name__ == "__main__":
    main()
