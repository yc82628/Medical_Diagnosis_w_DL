"""
Grading platform API.

    python auth.py create-user --username you --role admin     (first run)
    python service.py           then open http://127.0.0.1:8000

Runs with or without a trained model. Without one -- or with one that fails
the acceptance gate (config.AcceptanceConfig) -- it starts in DEMO mode and
says so on every response and across the top of the interface: a demo that
can be mistaken for a working diagnostic is worse than no demo.

Every endpoint except the interface itself and sign-in requires an account.

Endpoints                                          roles
    GET  /                       interface         public
    GET  /api/health             liveness          public
    POST /api/login | /api/logout | GET /api/me    public / signed in
    GET  /api/system             provenance        any
    POST /api/predict            upload + decide   any (multipart: file, patient_id,
                                                        eye, encounter_id?, device?)
    GET  /api/cases              worklist          any
    GET  /api/cases/{id}         case + trail      any (logged)
    GET  /api/cases/{id}/image   preview           any
    GET  /api/cases/{id}/gradcam attention map     grader, admin
    POST /api/cases/{id}/verdict record verdict    grader, admin
    GET  /api/visits             visit worklist    any
    GET  /api/encounters/{id}    visit decision    any
    GET  /api/encounters/{id}/fhir  FHIR R4 report grader, admin (logged)
    GET  /api/summary            rollup            any
    GET  /api/monitoring         drift signals     grader, admin
    GET  /api/audit/verify       chain check       admin
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel

import audit
import auth
import encounters
import fhir
from dicom_io import DicomError, is_dicom, read_dicom
from predictor import build_predictor
from quality_model import default_quality_gate, default_quality_note
from storage import ImageStore

APP_DIR = Path(__file__).parent
WEB_DIR = APP_DIR / "webapp"
SESSION_COOKIE = "dr_session"
MAX_UPLOAD_MB = float(os.environ.get("DR_MAX_UPLOAD_MB", "40"))
# A uniform attention map puts ~0.33 of its mass in this region (the corners
# and rim of a square image). Clearly more than that means the model is looking
# at the camera's black surround, not the retina: the shortcut signature.
BORDER_MASS_WARN = 0.40

ANY_ROLE = auth.ROLES
CLINICAL = ("grader", "admin")


class VerdictIn(BaseModel):
    verdict: str                 # REFER | ROUTINE | UNGRADABLE
    grade: int | None = None
    note: str | None = None


class LoginIn(BaseModel):
    username: str
    password: str


def _missing_ui_page(reason: str, detail: str) -> str:
    """A diagnostic page, not a shrug."""
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Interface not loaded</title><style>
body{{background:#101419;color:#E6EAF0;font:14px/1.6 system-ui,sans-serif;
margin:0;padding:48px 28px;display:flex;justify-content:center}}
main{{max-width:640px}}
h1{{font-size:19px;margin:0 0 6px;color:#E9A445}}
p{{color:#8C97A8;margin:0 0 18px}}
pre{{background:#171C24;border:1px solid #2A323E;border-radius:8px;
padding:14px 16px;overflow-x:auto;font:12.5px ui-monospace,monospace;color:#E6EAF0}}
code{{color:#E9A445}} b{{color:#E6EAF0;font-weight:600}}
</style></head><body><main>
<h1>The interface file was not found</h1>
<p>The API is running correctly &mdash; this is the web page that is missing,
not the server.</p>
<p><b>Looked for:</b></p><pre>{detail}</pre>
<p>{reason}</p>
<p><b>Fix:</b> create a <code>webapp</code> folder next to
<code>service.py</code> and save <code>index.html</code> inside it, then restart
with <code>python service.py</code>.</p>
</main></body></html>"""


def _decode_upload(raw: bytes, filename: str | None):
    """(BGR image, DICOM metadata or {}, source format). Raises HTTPException."""
    if is_dicom(raw) or (filename or "").lower().endswith(".dcm"):
        try:
            img, meta = read_dicom(raw)
        except DicomError as e:
            raise HTTPException(400, str(e))
        return img, meta, "dicom"
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise HTTPException(400, f"Could not decode '{filename}' as an image. "
                                 f"Supported: DICOM, PNG, JPEG, TIFF.")
    return arr, {}, "image"


def _json_safe(o):
    """NaN/inf -> None. evaluation.json legitimately holds NaN (e.g. NPV with no
    negatives); Python parses it, but a JSON response cannot carry it."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_json_safe(v) for v in o]
    return o


def _quality_gate_info() -> dict:
    g = default_quality_gate()
    if g is None:
        return {"source": "classical"}
    return {"source": "learned", "model": g.version, "threshold": g.threshold}


def create_app(db_path=audit.DB_PATH, image_root="artifacts/images",
               evaluation_path="artifacts/evaluation.json",
               data_key: bytes | None = None, predictor=None,
               load_note: str | None = None, cookie_secure: bool | None = None,
               sso_header: str | None = None) -> FastAPI:
    """
    Build the app. Tests pass temporary paths, a key and a stub predictor;
    production uses the defaults plus environment (DR_DATA_KEY, DR_SSO_HEADER,
    DR_COOKIE_SECURE).
    """
    if predictor is None:
        predictor, load_note = build_predictor()
        # Load the quality model now, not on the first upload: a corrupt
        # checkpoint should stop startup, and the note says which gate runs.
        load_note = f"{load_note} {default_quality_note()}"
    load_note = load_note or ""
    con = audit.connect(db_path)
    auth.init_schema(con)
    store = ImageStore(image_root, data_key)
    if cookie_secure is None:
        cookie_secure = os.environ.get("DR_COOKIE_SECURE") == "1"
    sso_header = sso_header if sso_header is not None else os.environ.get("DR_SSO_HEADER")

    app = FastAPI(title="DR Screening Platform", version="0.2.0")
    app.state.con, app.state.store, app.state.predictor = con, store, predictor

    def mode() -> str:
        if predictor.is_demo:
            return "DEMO"
        return "MODEL" if predictor.validated else "UNVALIDATED"

    # -- middleware ---------------------------------------------------------

    @app.middleware("http")
    async def guard(request: Request, call_next):
        # Refuse oversized uploads before reading the body.
        if request.url.path == "/api/predict":
            length = request.headers.get("content-length")
            if length and length.isdigit() and int(length) > MAX_UPLOAD_MB * 1024 * 1024:
                return JSONResponse({"detail": f"Upload larger than {MAX_UPLOAD_MB:g} MB."},
                                    status_code=413)
        resp = await call_next(request)
        # The interface is one self-contained file: no third-party scripts,
        # fonts or frames are needed, so none are allowed.
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' blob:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'")
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/api/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    # -- auth dependencies --------------------------------------------------

    def current_user(request: Request) -> dict:
        if sso_header:
            name = request.headers.get(sso_header)
            if name:
                u = auth.get_user(con, name.strip())
                if u and u["active"]:
                    return {"username": u["username"], "role": u["role"]}
                raise HTTPException(403, f"Signed in as {name!r} by single sign-on, "
                                         f"but that account is not enabled here.")
        user = auth.resolve_session(con, request.cookies.get(SESSION_COOKIE))
        if user is None:
            raise HTTPException(401, "Sign in required.")
        return user

    def require(*roles):
        def dep(user: dict = Depends(current_user)) -> dict:
            if user["role"] not in roles:
                raise HTTPException(403, f"Your role ({user['role']}) cannot do this.")
            return user
        return dep

    # -- interface + sign-in ------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index():
        f = WEB_DIR / "index.html"
        if not f.exists():
            reason = ("The <code>webapp</code> folder exists but contains no "
                      "<code>index.html</code>." if WEB_DIR.exists() else
                      "There is no <code>webapp</code> folder at all.")
            return HTMLResponse(_missing_ui_page(reason, str(f.resolve())), status_code=500)
        html = f.read_text(encoding="utf-8")
        if "<style>" not in html or len(html) < 2000:
            return HTMLResponse(_missing_ui_page(
                f"The file exists but looks incomplete ({len(html)} bytes). "
                f"Re-download <code>webapp/index.html</code>.", str(f.resolve())),
                status_code=500)
        return HTMLResponse(html)

    @app.get("/api/health")
    def health():
        """Liveness for container orchestration. Reveals nothing about the data."""
        return {"status": "ok"}

    @app.post("/api/login")
    def login(body: LoginIn, response: Response):
        try:
            user = auth.authenticate(con, body.username.strip(), body.password)
        except auth.AuthError as e:
            audit.log_event(con, body.username.strip()[:64] or "?", "login_failed")
            raise HTTPException(401, str(e))
        token = auth.create_session(con, user["username"])
        response.set_cookie(SESSION_COOKIE, token, httponly=True, samesite="strict",
                            secure=cookie_secure,
                            max_age=int(auth.SESSION_LIFETIME.total_seconds()))
        audit.log_event(con, user["username"], "login")
        return user

    @app.post("/api/logout")
    def logout(request: Request, response: Response):
        auth.revoke_session(con, request.cookies.get(SESSION_COOKIE))
        response.delete_cookie(SESSION_COOKIE)
        return {"signed_out": True}

    @app.get("/api/me")
    def me(user: dict = Depends(current_user)):
        return user

    # -- system -------------------------------------------------------------

    @app.get("/api/system")
    def system(user: dict = Depends(current_user)):
        """Provenance. Anyone looking at an output can see exactly what produced it."""
        ev = Path(evaluation_path)
        evaluation = _json_safe(json.loads(ev.read_text())) if ev.exists() else None
        return {
            "mode": mode(),
            "validated": bool(predictor.validated),
            "acceptance_failures": list(getattr(predictor, "acceptance_failures", [])),
            "gradcam_available": not predictor.is_demo,
            "note": load_note,
            "model_version": predictor.model_version,
            "preprocess_fingerprint": predictor.pc.fingerprint(),
            "quality_gate": _quality_gate_info(),
            "image_size": predictor.pc.image_size,
            "operating_point": predictor.operating_point,
            "abstain_low": predictor.abstain_low,
            "abstain_high": predictor.abstain_high,
            "has_evaluation": evaluation is not None,
            "evaluation": evaluation,
            "max_upload_mb": MAX_UPLOAD_MB,
            "intended_use": {
                "decision": "Referral triage for diabetic retinopathy screening",
                "target": "Referable DR, ICDR grade 2 or worse",
                "user": "Trained grader or clinician within a screening programme",
                "not_for": [
                    "Diagnosis or treatment planning",
                    "Detection of glaucoma, AMD, or any non-DR pathology",
                    "Use on patients without a diabetes diagnosis",
                    "Autonomous use without human review",
                ],
                "status": "RESEARCH SOFTWARE -- NOT A MEDICAL DEVICE",
            },
        }

    # -- predict ------------------------------------------------------------
    # Plain `def`, not `async def`: FastAPI runs it in a worker thread, so a
    # slow model call does not freeze every other request.

    @app.post("/api/predict")
    def predict(file: UploadFile = File(...), patient_id: str | None = Form(None),
                eye: str | None = Form(None), encounter_id: str | None = Form(None),
                device: str | None = Form(None),
                user: dict = Depends(require(*ANY_ROLE))):
        limit = int(MAX_UPLOAD_MB * 1024 * 1024)
        raw = file.file.read(limit + 1)
        if not raw:
            raise HTTPException(400, "Empty file.")
        if len(raw) > limit:
            raise HTTPException(413, f"Upload larger than {MAX_UPLOAD_MB:g} MB.")

        arr, meta, fmt = _decode_upload(raw, file.filename)

        # Form fields win over DICOM tags only when the tags are absent; a
        # mismatch between the two is a wrong-patient risk, so it is refused.
        pid = (patient_id or "").strip() or None
        if meta.get("patient_id") and pid and pid != meta["patient_id"]:
            raise HTTPException(400, "Patient ID entered does not match the DICOM file's "
                                     "PatientID. Check you have the right patient.")
        pid = pid or meta.get("patient_id")
        side = encounters.normalise_eye(eye) if eye else None
        if eye and side is None:
            raise HTTPException(400, "Eye must be 'left' or 'right'.")
        if meta.get("eye") and side and side != meta["eye"]:
            raise HTTPException(400, "Eye entered does not match the DICOM laterality tag.")
        side = side or meta.get("eye")
        if not pid:
            raise HTTPException(400, "Patient ID is required: a result that cannot be "
                                     "linked to a patient cannot be acted on.")
        if not side:
            raise HTTPException(400, "Eye (left or right) is required.")
        if len(pid) > 64 or len(encounter_id or "") > 128 or len(device or "") > 128:
            raise HTTPException(400, "Identifier too long.")
        enc = ((encounter_id or "").strip() or meta.get("encounter_id")
               # Local date: a visit at 00:30 belongs to today in the clinic,
               # not to yesterday in UTC.
               or f"{pid}:{datetime.now().date().isoformat()}")
        dev = (device or "").strip() or meta.get("device")

        pred = predictor.predict(arr)
        case_id = audit.log_prediction(
            con, pred, file.filename, patient_id=pid, eye=side, encounter_id=enc,
            device=dev, uploaded_by=user["username"], source_format=fmt)

        h, w = arr.shape[:2]
        s = 720 / max(h, w)
        preview = cv2.resize(arr, (int(w * s), int(h * s)),
                             interpolation=cv2.INTER_AREA) if s < 1 else arr
        ok_p, jpg = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 88])
        # Full resolution too: Grad-CAM must run on the pixels the model saw.
        ok_r, png = cv2.imencode(".png", arr)
        if ok_p:
            store.put(case_id, "preview", jpg.tobytes())
        if ok_r:
            store.put(case_id, "raw", png.tobytes())

        out = pred.to_dict()
        out.update(case_id=case_id, filename=file.filename, patient_id=pid, eye=side,
                   encounter_id=enc, device=dev)
        return out

    # -- cases --------------------------------------------------------------

    @app.get("/api/cases")
    def cases(limit: int = 200, user: dict = Depends(current_user)):
        return {"cases": audit.list_cases(con, max(1, min(limit, 1000)))}

    @app.get("/api/cases/{case_id}")
    def case(case_id: str, user: dict = Depends(current_user)):
        d = audit.case_detail(con, case_id)
        if d is None:
            raise HTTPException(404, f"No case {case_id}")
        audit.log_event(con, user["username"], "view_case", case_id)
        return d

    @app.get("/api/cases/{case_id}/image")
    def case_image(case_id: str, user: dict = Depends(current_user)):
        if not case_id.isalnum():
            raise HTTPException(404, "No such case.")
        data = store.get(case_id, "preview")
        if data is None:
            raise HTTPException(404, "No preview stored for this case (it may have "
                                     "been removed by the retention policy).")
        return Response(data, media_type="image/jpeg")

    @app.get("/api/cases/{case_id}/gradcam")
    def case_gradcam(case_id: str, user: dict = Depends(require(*CLINICAL))):
        """
        Grad-CAM overlay, computed on demand. 409 rather than an image when an
        explanation would be misleading (demo mode; ungradable image).
        X-Border-Mass reports attention on the black surround.
        """
        detail = audit.case_detail(con, case_id)
        if detail is None:
            raise HTTPException(404, f"No case {case_id}")
        if predictor.is_demo:
            raise HTTPException(
                409, "Grad-CAM is unavailable in demo mode: there is no trained model "
                     "to explain.")
        if not detail["gradable"]:
            raise HTTPException(
                409, "This image failed the quality gate, so the model never graded "
                     "it. There is no decision to explain.")

        cached = store.get(case_id, "gradcam")
        if cached is not None:
            prior = audit.last_event_detail(con, "gradcam", case_id) or {}
            return _gradcam_response(cached, prior.get("border_mass"))

        raw = store.get(case_id, "raw")
        if raw is None:
            raise HTTPException(404, "Original image for this case is no longer stored.")
        try:
            from gradcam import explain
        except ImportError:
            raise HTTPException(503, "Grad-CAM requires PyTorch, which is not installed.")
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        try:
            with predictor.lock:
                result = explain(predictor.model, img, predictor.pc, device=predictor.device)
        except RuntimeError as e:
            raise HTTPException(422, f"Could not produce a reliable explanation: {e}")
        ok, jpg = cv2.imencode(".jpg", result["overlay"], [cv2.IMWRITE_JPEG_QUALITY, 88])
        border = float(result["stats"]["border_mass"])
        if ok:
            store.put(case_id, "gradcam", jpg.tobytes())
        audit.log_event(con, user["username"], "gradcam", case_id,
                        {"border_mass": round(border, 4),
                         "border_warning": border > BORDER_MASS_WARN})
        return _gradcam_response(jpg.tobytes(), border)

    def _gradcam_response(data: bytes, border_mass: float | None) -> Response:
        headers = {}
        if border_mass is not None:
            headers["X-Border-Mass"] = f"{border_mass:.3f}"
            headers["X-Border-Warning"] = "1" if border_mass > BORDER_MASS_WARN else "0"
        return Response(data, media_type="image/jpeg", headers=headers)

    @app.post("/api/cases/{case_id}/verdict")
    def verdict(case_id: str, body: VerdictIn, user: dict = Depends(require(*CLINICAL))):
        if body.verdict not in ("REFER", "ROUTINE", "UNGRADABLE"):
            raise HTTPException(400, "verdict must be REFER, ROUTINE or UNGRADABLE")
        if body.grade is not None and body.grade not in range(5):
            raise HTTPException(400, "grade must be an ICDR grade 0-4")
        try:
            vid = audit.log_verdict(con, case_id, body.verdict, body.grade,
                                    user["username"], (body.note or "")[:2000] or None)
        except KeyError as e:
            raise HTTPException(404, str(e))
        return {"verdict_id": vid, "case_id": case_id, "recorded": True,
                "grader": user["username"]}

    # -- encounters ---------------------------------------------------------

    def _encounter(encounter_id: str):
        rows = audit.encounter_cases(con, encounter_id)
        if not rows:
            raise HTTPException(404, f"No visit {encounter_id}")
        return rows, encounters.combine(rows)

    @app.get("/api/visits")
    def visits(limit: int = 200, user: dict = Depends(current_user)):
        """
        Worklist of visits, each with its patient-level decision. Visits that
        still need a grader come first, then by clinical priority (a visit
        needing human review, then referrals, then incomplete, then routine),
        then most recent.
        """
        order = {"MANUAL_REVIEW": 0, "REFER": 1, "INCOMPLETE": 2, "ROUTINE": 3}
        rows = []
        for v in audit.list_visits(con, max(1, min(limit, 1000))):
            combined = encounters.combine(v["cases"])
            pending = sum(1 for c in v["cases"] if not c["latest_verdict"])
            rows.append({
                "encounter_id": v["encounter_id"], "patient_id": v["patient_id"],
                "first_utc": v["first_utc"], "last_utc": v["last_utc"],
                "n_images": v["n_images"], "pending_images": pending,
                "needs_review": pending > 0, **combined,
            })
        rows.sort(key=lambda r: r["last_utc"], reverse=True)
        rows.sort(key=lambda r: (not r["needs_review"], order.get(r["decision"], 9)))
        return {"visits": rows}

    @app.get("/api/encounters/{encounter_id:path}/fhir")
    def encounter_fhir(encounter_id: str, user: dict = Depends(require(*CLINICAL))):
        rows, combined = _encounter(encounter_id)
        try:
            report = fhir.encounter_report(encounter_id, rows, combined)
        except ValueError as e:
            raise HTTPException(409, str(e))
        audit.log_event(con, user["username"], "export_fhir", encounter_id,
                        {"status": report["status"]})
        return JSONResponse(report, media_type="application/fhir+json")

    @app.get("/api/encounters/{encounter_id:path}")
    def encounter(encounter_id: str, user: dict = Depends(current_user)):
        rows, combined = _encounter(encounter_id)
        return {"encounter_id": encounter_id,
                "patient_id": next((r["patient_id"] for r in rows if r["patient_id"]), None),
                **combined, "cases": rows}

    # -- operations ---------------------------------------------------------

    @app.get("/api/summary")
    def summary(user: dict = Depends(current_user)):
        return audit.summary(con)

    @app.get("/api/monitoring")
    def monitoring(days: int = 90, user: dict = Depends(require(*CLINICAL))):
        return audit.monitoring(con, max(7, min(days, 730)))

    @app.get("/api/audit/verify")
    def verify(user: dict = Depends(require("admin"))):
        result = audit.verify_chain(con)
        audit.log_event(con, user["username"], "verify_chain", None,
                        {"ok": result["ok"], "entries": result["entries"]})
        return result

    return app


_APP = None


def __getattr__(name):
    """`uvicorn service:app` works; importing the module for tests does not build one."""
    global _APP
    if name == "app":
        if _APP is None:
            _APP = create_app()
        return _APP
    raise AttributeError(name)


if __name__ == "__main__":
    import argparse
    import uvicorn

    ap = argparse.ArgumentParser(description="Run the DR screening platform.")
    ap.add_argument("--host", default="127.0.0.1",
                    help="Interface to bind. Default 127.0.0.1 = this machine only.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ssl-certfile", help="TLS certificate (PEM). Required off-localhost.")
    ap.add_argument("--ssl-keyfile", help="TLS private key (PEM).")
    ap.add_argument("--behind-tls-proxy", action="store_true",
                    help="A TLS-terminating reverse proxy sits in front (the Docker "
                         "deployment). Satisfies the TLS requirement and marks "
                         "cookies Secure.")
    ap.add_argument("--i-understand-this-is-not-a-medical-device",
                    action="store_true", dest="acknowledged",
                    help="Required in order to bind to a non-local address.")
    args = ap.parse_args()

    LOCAL = {"127.0.0.1", "localhost", "::1"}
    tls = bool(args.ssl_certfile and args.ssl_keyfile)
    secure_transport = tls or args.behind_tls_proxy

    print("=" * 66)
    print("  DR Screening Platform")
    print("  RESEARCH SOFTWARE -- NOT A MEDICAL DEVICE")
    print("=" * 66)

    if args.host not in LOCAL:
        problems = []
        if not args.acknowledged:
            problems.append(
                "  - Exposing this beyond one machine changes the legal picture:\n"
                "    retinal photographs are special-category health data (GDPR\n"
                "    Art. 9), and offering software for clinical use engages medical\n"
                "    device regulation. See INTENDED_USE.md section 7. Re-run with\n"
                "    --i-understand-this-is-not-a-medical-device once considered.")
        if not secure_transport:
            problems.append(
                "  - No TLS. Passwords, session cookies and patient images would\n"
                "    cross the network in clear text. Pass --ssl-certfile and\n"
                "    --ssl-keyfile, or run behind a TLS reverse proxy and pass\n"
                "    --behind-tls-proxy.")
        if problems:
            raise SystemExit(f"\n  Refusing to bind to {args.host}:\n\n"
                             + "\n\n".join(problems) + "\n")
        via = "TLS" if tls else "a TLS reverse proxy"
        print(f"\n  ! Bound to {args.host} via {via} -- reachable beyond this machine.")
    else:
        print("  Network: this machine only (127.0.0.1). Not reachable externally.")

    if secure_transport:
        os.environ.setdefault("DR_COOKIE_SECURE", "1")

    application = create_app()
    if auth.count_users(application.state.con) == 0:
        print("\n  No accounts exist yet. Create the first one in another terminal:")
        print("      python auth.py create-user --username <name> --role admin\n")

    if args.host in ("0.0.0.0", "::"):
        print(f"  Listening on all interfaces, port {args.port}. Open it through "
              f"the hospital's reverse proxy address, not this one.")
    else:
        print(f"  Open http{'s' if tls else ''}://{args.host}:{args.port}")
    print("=" * 66)
    uvicorn.run(application, host=args.host, port=args.port, log_level="warning",
                ssl_certfile=args.ssl_certfile, ssl_keyfile=args.ssl_keyfile)
