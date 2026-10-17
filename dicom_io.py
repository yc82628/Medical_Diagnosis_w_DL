"""
DICOM ingest.

Fundus cameras in a hospital export DICOM (Ophthalmic Photography IOD), not
JPEG. Reading it directly also brings the identifiers the decision needs:
patient ID, which eye, which visit, which camera -- typed by nobody, so not
mistyped by anybody.

Only what the platform needs is read. Patient name, birth date and the rest
of the demographic header are never extracted or stored.

Compressed transfer syntaxes (JPEG baseline, JPEG 2000) need a pixel-data
plugin, e.g. `pip install pylibjpeg pylibjpeg-openjpeg`; without one the
upload is refused with an explanation rather than mis-decoded.
"""
from __future__ import annotations

import io

import cv2
import numpy as np

from encounters import normalise_eye


class DicomError(ValueError):
    """The file is DICOM but cannot be used. Message is safe to show the user."""


def is_dicom(raw: bytes) -> bool:
    return len(raw) >= 132 and raw[128:132] == b"DICM"


def read_dicom(raw: bytes) -> tuple[np.ndarray, dict]:
    """Returns (BGR uint8 image, metadata)."""
    try:
        import pydicom
    except ImportError:
        raise DicomError("DICOM support needs pydicom: pip install pydicom") from None

    try:
        ds = pydicom.dcmread(io.BytesIO(raw))
    except Exception as e:
        raise DicomError(f"Not a readable DICOM file ({e.__class__.__name__}).") from None

    if int(getattr(ds, "NumberOfFrames", 1) or 1) > 1:
        raise DicomError("Multi-frame DICOM is not supported; export single images.")
    try:
        arr = ds.pixel_array          # pydicom converts YBR to RGB by default
    except Exception as e:
        raise DicomError(
            f"Could not decode the pixel data (transfer syntax "
            f"{getattr(ds.file_meta, 'TransferSyntaxUID', '?')}). Install a "
            f"decoder plugin: pip install pylibjpeg pylibjpeg-openjpeg. "
            f"({e.__class__.__name__})") from None

    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        bits = int(getattr(ds, "BitsStored", 16) or 16)
        arr = np.clip(arr.astype(np.float64) / (2 ** bits - 1) * 255, 0, 255).astype(np.uint8)

    pi = str(getattr(ds, "PhotometricInterpretation", "RGB")).upper()
    if arr.ndim == 2:
        if pi == "MONOCHROME1":
            arr = 255 - arr
        bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    else:
        raise DicomError(f"Unsupported pixel layout {arr.shape}.")

    lat = getattr(ds, "ImageLaterality", None) or getattr(ds, "Laterality", None)
    device = " ".join(str(x) for x in (getattr(ds, "Manufacturer", None),
                                       getattr(ds, "ManufacturerModelName", None)) if x)
    meta = {
        "patient_id": str(ds.PatientID) if getattr(ds, "PatientID", None) else None,
        "eye": normalise_eye(lat),
        "encounter_id": (str(ds.StudyInstanceUID)
                         if getattr(ds, "StudyInstanceUID", None) else None),
        "device": device or None,
    }
    return bgr, meta
