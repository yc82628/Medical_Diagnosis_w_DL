"""
Grading platform API.

    python service.py           then open http://127.0.0.1:8000

Runs with or without a trained model. Without one it starts in DEMO mode and
says so on every response and across the top of the interface -- a demo that
can be mistaken for a working diagnostic is worse than no demo.

Endpoints
    GET  /                      grader review interface
    GET  /api/system            model version, operating point, mode, scope
    POST /api/predict           upload an image, get a decision (logged)
    GET  /api/cases             worklist, ordered by clinical priority
    GET  /api/cases/{id}        one case with its full audit trail
    GET  /api/cases/{id}/image  the stored preview
    POST /api/cases/{id}/verdict  record a grader decision
    GET  /api/summary           operational rollup
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

import audit
from predictor import build_predictor

APP_DIR = Path(__file__).parent
WEB_DIR = APP_DIR / "webapp"
PREVIEW_DIR = Path("artifacts/previews")
RAW_DIR = Path("artifacts/raw")           # full-resolution uploads, for Grad-CAM

app = FastAPI(title="DR Screening Platform", version="0.1.0")

PREDICTOR, LOAD_NOTE = build_predictor()
CON = audit.connect()
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
print(f"\n  {LOAD_NOTE}\n")


# ---------------------------------------------------------------------------

class VerdictIn(BaseModel):
    verdict: str                 # REFER | ROUTINE | UNGRADABLE
    grade: int | None = None
    grader: str | None = None
    note: str | None = None


def _missing_ui_page(reason: str, detail: str) -> str:
    """
    A diagnostic page, not a shrug.

    The previous version returned an unstyled one-line <h1>, which renders as
    large serif text on white and looks indistinguishable from "the app is
    broken". If the interface cannot load, the page that replaces it should say
    exactly what is wrong and how to fix it.
    """
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
<code>service.py</code> and save <code>index.html</code> inside it:</p>
<pre>Medical_Diagnosis_w_DL\\
├── service.py
└── webapp\\
    └── index.html</pre>
<p>Then restart with <code>python service.py</code>. The API works meanwhile
&mdash; try <a href="/api/system" style="color:#E9A445">/api/system</a>.</p>
</main></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    f = WEB_DIR / "index.html"
    if not f.exists():
        nearby = []
        if WEB_DIR.exists():
            nearby = [p.name for p in WEB_DIR.iterdir()][:10]
            reason = (f"The <code>webapp</code> folder exists but contains no "
                      f"<code>index.html</code>. It contains: {nearby or 'nothing'}.")
        else:
            reason = ("There is no <code>webapp</code> folder at all. It is a "
                      "subfolder, so it is easy to miss when saving files "
                      "individually.")
        return HTMLResponse(_missing_ui_page(reason, str(f.resolve())),
                            status_code=500)

    html = f.read_text(encoding="utf-8")
    if "<style>" not in html or len(html) < 2000:
        # A truncated or plain-text copy renders as unstyled serif text on white,
        # which looks like a broken app rather than a broken file.
        return HTMLResponse(_missing_ui_page(
            f"The file exists but looks incomplete &mdash; {len(html)} bytes, and "
            f"{'no &lt;style&gt; block' if '<style>' not in html else 'unexpectedly short'}. "
            f"It was probably saved partially, or saved as plain text. "
            f"Re-download <code>webapp/index.html</code>.",
            str(f.resolve())), status_code=500)
    return HTMLResponse(html)


@app.get("/api/system")
def system():
    """Provenance. Anyone looking at an output can see exactly what produced it."""
    ev = Path("artifacts/evaluation.json")
    evaluation = json.loads(ev.read_text()) if ev.exists() else None
    return {
        "mode": "DEMO" if PREDICTOR.is_demo else "MODEL",
        "gradcam_available": not PREDICTOR.is_demo,
        "note": LOAD_NOTE,
        "model_version": PREDICTOR.model_version,
        "preprocess_fingerprint": PREDICTOR.pc.fingerprint(),
        "image_size": PREDICTOR.pc.image_size,
        "operating_point": PREDICTOR.operating_point,
        "abstain_low": PREDICTOR.abstain_low,
        "abstain_high": PREDICTOR.abstain_high,
        "has_evaluation": evaluation is not None,
        "evaluation": evaluation,
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


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file.")

    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise HTTPException(
            400, f"Could not decode '{file.filename}' as an image. "
                 f"Supported: png, jpg, jpeg, tif.")

    pred = PREDICTOR.predict(arr)
    case_id = audit.log_prediction(CON, pred, file.filename)

    # Store a downscaled preview only -- enough to review, small enough to serve.
    h, w = arr.shape[:2]
    s = 720 / max(h, w)
    preview = cv2.resize(arr, (int(w * s), int(h * s)),
                         interpolation=cv2.INTER_AREA) if s < 1 else arr
    cv2.imwrite(str(PREVIEW_DIR / f"{case_id}.jpg"), preview,
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    # Keep the full-resolution image too. Grad-CAM must run on the same pixels
    # the model saw, and the preview above is downscaled for display.
    cv2.imwrite(str(RAW_DIR / f"{case_id}.png"), arr)

    out = pred.to_dict()
    out["case_id"] = case_id
    out["filename"] = file.filename
    return JSONResponse(out)


@app.get("/api/cases")
def cases(limit: int = 200):
    return {"cases": audit.list_cases(CON, limit)}


@app.get("/api/cases/{case_id}")
def case(case_id: str):
    d = audit.case_detail(CON, case_id)
    if d is None:
        raise HTTPException(404, f"No case {case_id}")
    return d


@app.get("/api/cases/{case_id}/image")
def case_image(case_id: str):
    f = PREVIEW_DIR / f"{case_id}.jpg"
    if not f.exists():
        raise HTTPException(404, "No preview stored for this case.")
    return FileResponse(f, media_type="image/jpeg")


@app.get("/api/cases/{case_id}/gradcam")
def case_gradcam(case_id: str):
    """
    Grad-CAM overlay for a case, computed on demand.

    Returns 409 rather than an image when an explanation would be misleading:
      - demo mode: there is no trained model to explain
      - ungradable images: the model never ran, so there is nothing to attribute
    An honest "not available, and here is why" beats a heatmap over noise.
    """
    detail = audit.case_detail(CON, case_id)
    if detail is None:
        raise HTTPException(404, f"No case {case_id}")

    if PREDICTOR.is_demo:
        raise HTTPException(
            409, "Grad-CAM is unavailable in demo mode: there is no trained "
                 "model to explain. The demo predictor is an image-statistics "
                 "heuristic. Train a model and restart to enable explanations.")

    if not detail["gradable"]:
        raise HTTPException(
            409, "This image failed the quality gate, so the model never graded "
                 "it. There is no decision to explain.")

    cached = PREVIEW_DIR / f"{case_id}_gradcam.jpg"
    if cached.exists():
        return FileResponse(cached, media_type="image/jpeg")

    raw = RAW_DIR / f"{case_id}.png"
    if not raw.exists():
        raise HTTPException(404, "Original image for this case is no longer stored.")

    try:
        from gradcam import explain
    except ImportError:
        raise HTTPException(503, "Grad-CAM requires PyTorch, which is not installed.")

    img = cv2.imread(str(raw), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(404, "Stored image could not be read.")

    try:
        result = explain(PREDICTOR.model, img, PREDICTOR.pc, device=PREDICTOR.device)
    except RuntimeError as e:
        # gradcam raises on degenerate maps (e.g. an unconverged model) rather
        # than returning a flat overlay that looks like a real explanation.
        raise HTTPException(422, f"Could not produce a reliable explanation: {e}")

    cv2.imwrite(str(cached), result["overlay"], [cv2.IMWRITE_JPEG_QUALITY, 88])
    return FileResponse(cached, media_type="image/jpeg")


@app.post("/api/cases/{case_id}/verdict")
def verdict(case_id: str, body: VerdictIn):
    if body.verdict not in ("REFER", "ROUTINE", "UNGRADABLE"):
        raise HTTPException(400, "verdict must be REFER, ROUTINE or UNGRADABLE")
    try:
        vid = audit.log_verdict(CON, case_id, body.verdict, body.grade,
                                body.grader, body.note)
    except KeyError as e:
        raise HTTPException(404, str(e))
    return {"verdict_id": vid, "case_id": case_id, "recorded": True}


@app.get("/api/summary")
def summary():
    return audit.summary(CON)


if __name__ == "__main__":
    import argparse
    import uvicorn

    ap = argparse.ArgumentParser(description="Run the DR screening platform.")
    ap.add_argument("--host", default="127.0.0.1",
                    help="Interface to bind. Default 127.0.0.1 = this machine only. "
                         "Anything else makes the service reachable over the network.")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--i-understand-this-is-not-a-medical-device",
                    action="store_true", dest="acknowledged",
                    help="Required in order to bind to a non-local address.")
    args = ap.parse_args()

    LOCAL = {"127.0.0.1", "localhost", "::1"}

    print("=" * 66)
    print("  DR Screening Platform")
    print("  RESEARCH SOFTWARE -- NOT A MEDICAL DEVICE")
    print("=" * 66)

    # Binding to 127.0.0.1 makes the service reachable only from this machine:
    # not the local network, not the internet. That is the right default for
    # research software, and it is also the configuration that stays furthest
    # from any question of "placing on the market" or "putting into service"
    # under EU MDR. Exposing it should be a deliberate act, never the result of
    # leaving a flag at its default.
    if args.host not in LOCAL:
        if not args.acknowledged:
            raise SystemExit(
                f"\n  Refusing to bind to {args.host}."
                f"\n"
                f"\n  That makes this reachable beyond this machine. What changes:"
                f"\n"
                f"\n    - Others could submit images, including real patient images."
                f"\n      Retinal photographs of identifiable people are special-"
                f"\n      category health data under GDPR Article 9. This project has"
                f"\n      no lawful basis, no consent flow, and no DPIA."
                f"\n    - Offering software for clinical use is what engages medical"
                f"\n      device regulation. Running it privately for research does not."
                f"\n    - Uploaded previews are written unencrypted to"
                f"\n      artifacts/previews/, and the audit database is unauthenticated."
                f"\n    - There is no login. Anyone who can reach the port can use it."
                f"\n"
                f"\n  If you have considered all of that, re-run with:"
                f"\n      python service.py --host {args.host} \\"
                f"\n          --i-understand-this-is-not-a-medical-device"
                f"\n"
            )
        print(f"\n  ! Bound to {args.host} -- reachable beyond this machine.")
        print("  ! Do not submit identifiable patient images to this instance.\n")
    else:
        print("  Network: this machine only (127.0.0.1). Not reachable externally.")

    ui = WEB_DIR / "index.html"
    if not ui.exists():
        print(f"\n  WARNING: {ui.resolve()} not found.")
        print("  The API will run, but the web interface will not load.")
        print("  Create a 'webapp' folder next to service.py and put index.html in it.\n")
    else:
        size = ui.stat().st_size
        if size < 2000:
            print(f"\n  WARNING: {ui.name} is only {size} bytes -- it looks truncated.")
            print("  Re-download webapp/index.html.\n")
        else:
            print(f"  Interface: {ui.name} ({size//1024} KB)")

    print(f"  Open http://{args.host}:{args.port}")
    print("=" * 66)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")