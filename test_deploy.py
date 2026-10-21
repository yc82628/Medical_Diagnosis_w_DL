"""Deployment kit: LAN certificates, backup / verify / restore, server.env parsing."""
import sys
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.asymmetric import ec

sys.path.insert(0, str(Path(__file__).parent / "deploy"))

import audit  # noqa: E402
import backup as bk  # noqa: E402
import make_certs  # noqa: E402
import start_server  # noqa: E402
from predictor import DemoPredictor  # noqa: E402
from storage import ImageStore  # noqa: E402
from test_predictor import _fake_fundus  # noqa: E402


# -- certificates ---------------------------------------------------------------

def _load(p):
    return x509.load_pem_x509_certificate(Path(p).read_bytes())


def test_server_certificate_is_signed_by_the_local_ca(tmp_path):
    r = make_certs.make(["retina.ward3.local"], ["192.168.10.20"], tmp_path)
    ca, srv = _load(tmp_path / "ca.crt"), _load(tmp_path / "server.crt")
    srv.verify_directly_issued_by(ca)                    # raises if not
    san = srv.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert "retina.ward3.local" in san.get_values_for_type(x509.DNSName)
    assert [str(i) for i in san.get_values_for_type(x509.IPAddress)] == ["192.168.10.20"]
    assert ca.extensions.get_extension_for_class(x509.BasicConstraints).value.ca
    assert not srv.extensions.get_extension_for_class(x509.BasicConstraints).value.ca
    days = (srv.not_valid_after_utc - srv.not_valid_before_utc).days
    assert days <= 826, "browsers reject server certificates valid > 825 days"
    assert r["new_ca"] and isinstance(srv.public_key(), ec.EllipticCurvePublicKey)


def test_reissuing_keeps_the_ca_so_devices_stay_trusted(tmp_path):
    make_certs.make(["a.local"], [], tmp_path)
    ca1 = (tmp_path / "ca.crt").read_bytes()
    r = make_certs.make(["a.local"], ["10.0.0.5"], tmp_path)   # new IP address
    assert not r["new_ca"] and (tmp_path / "ca.crt").read_bytes() == ca1
    _load(tmp_path / "server.crt").verify_directly_issued_by(_load(tmp_path / "ca.crt"))


def test_missing_ca_key_is_refused_not_silently_replaced(tmp_path):
    make_certs.make(["a.local"], [], tmp_path)
    (tmp_path / "ca.key").unlink()                       # moved offline, as advised
    with pytest.raises(SystemExit, match="ca.key"):
        make_certs.make(["a.local"], [], tmp_path)


# -- backup / restore -------------------------------------------------------------

def _install(root: Path, key: bytes):
    """A minimal installation: one prediction in the DB, its encrypted images."""
    (root / "artifacts").mkdir(parents=True)
    (root / "checkpoints").mkdir()
    (root / "checkpoints" / "best_model.pt").write_bytes(b"weights")
    (root / "artifacts" / "evaluation.json").write_text("{}")
    con = audit.connect(root / "artifacts" / "audit.db")
    cid = audit.log_prediction(con, DemoPredictor().predict(_fake_fundus()),
                               patient_id="P1", eye="left")
    con.close()
    ImageStore(root / "artifacts" / "images", key).put(cid, "raw", b"\x89PNG pixels")
    return cid


def test_backup_verify_restore_round_trip(tmp_path, monkeypatch):
    key = Fernet.generate_key()
    monkeypatch.setenv("DR_DATA_KEY", key.decode())
    root = tmp_path / "install"
    cid = _install(root, key)

    dest = bk.backup(root, tmp_path / "backups")
    assert not (dest / "secrets").exists(), "the key must never be in a backup"
    v = bk.verify(dest)
    assert v["ok"] and v["images"] == 1 and v["sample_decrypted"] == 1
    assert v["audit_entries"] == 1

    # Disaster: the live data is lost. Restore brings it back; old data is kept aside.
    import shutil
    shutil.rmtree(root / "artifacts")
    (root / "artifacts").mkdir()
    aside = bk.restore(dest, root, port=1)               # nothing listens on port 1
    assert aside.exists()
    con = audit.connect(root / "artifacts" / "audit.db")
    assert audit.case_detail(con, cid)["patient_id"] == "P1"
    assert audit.verify_chain(con)["ok"]
    assert ImageStore(root / "artifacts" / "images", key).get(cid, "raw") == b"\x89PNG pixels"


def test_verify_detects_tampering_and_a_wrong_key(tmp_path, monkeypatch):
    key = Fernet.generate_key()
    root = tmp_path / "install"
    _install(root, key)
    dest = bk.backup(root, tmp_path / "backups")
    assert bk.verify(dest, key)["ok"]
    assert not bk.verify(dest, Fernet.generate_key())["ok"]      # wrong key
    (dest / "checkpoints" / "best_model.pt").write_bytes(b"tampered")
    assert any("changed" in p for p in bk.verify(dest, key)["problems"])


def test_keep_prunes_old_backups(tmp_path, monkeypatch):
    key = Fernet.generate_key()
    root = tmp_path / "install"
    _install(root, key)
    import time
    for _ in range(3):
        bk.backup(root, tmp_path / "b", keep=2)
        time.sleep(1.1)                                  # distinct timestamps
    assert len([d for d in (tmp_path / "b").iterdir() if d.is_dir()]) == 2


def test_restore_refuses_while_server_runs(tmp_path, monkeypatch):
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    try:
        with pytest.raises(SystemExit, match="Stop it"):
            bk.restore(tmp_path, tmp_path, port=s.getsockname()[1])
    finally:
        s.close()


# -- server.env -------------------------------------------------------------------

def test_server_env_requires_its_settings(tmp_path):
    p = tmp_path / "server.env"
    p.write_text("DR_HOST=0.0.0.0\nDR_PORT=8443  # comment\n")
    with pytest.raises(SystemExit, match="missing"):
        start_server.load_env(p)
    p.write_text("DR_HOST=0.0.0.0\nDR_PORT=8443\nDR_SSL_CERT=a\nDR_SSL_KEY=b\n"
                 "DR_ALLOWED_NETWORKS=192.168.10.0/24\n")
    assert start_server.load_env(p)["DR_PORT"] == "8443"
