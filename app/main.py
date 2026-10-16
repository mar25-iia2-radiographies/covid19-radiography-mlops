"""API pédagogique de classification de radiographies avec masque associé.

Usage expérimental uniquement : aucune utilisation diagnostique.
"""
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.model import CLASS_NAMES, load_covid_model, predict_image
from app.preprocessing import make_batch, preprocess_image

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
ALLOWED_TYPES = {"image/png", "image/jpeg"}


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    model_version: str
    inference_ms: float = Field(ge=0)
    disclaimer: str = "Usage expérimental uniquement — ne constitue pas un diagnostic médical."


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_covid_model()
    yield
    app.state.model = None


app = FastAPI(
    title="COVID-19 Radiography — MLOps API",
    version="0.1.0",
    description="Prototype de classification de radiographies avec masque pulmonaire obligatoire. Usage non clinique.",
    lifespan=lifespan,
)


@app.get("/")
def home():
    return {"service": "COVID-19 Radiography API", "documentation": "/docs", "usage": "experimental"}


@app.get("/health")
def health():
    if getattr(app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {"status": "ok", "model_loaded": True}


@app.get("/model/info")
def model_info():
    if getattr(app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {"model_name": "VGG16", "model_version": "v1.0", "input_shape": [224, 224, 3], "classes": CLASS_NAMES, "requires_mask": True}


async def _read_image(file: UploadFile, label: str) -> bytes:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail=f"{label}: PNG ou JPEG uniquement")
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"{label}: taille supérieure à 10 Mio")
    if not raw:
        raise HTTPException(status_code=422, detail=f"{label}: fichier vide")
    # Vérifier que les octets représentent réellement une image décodable.
    decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if decoded is None or decoded.size > MAX_IMAGE_PIXELS:
        raise HTTPException(status_code=422, detail=f"{label}: image invalide ou trop grande")
    return raw


@app.post("/predict", response_model=PredictResponse)
async def predict(image: UploadFile = File(...), mask: UploadFile = File(...)):
    if getattr(app.state, "model", None) is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    image_bytes = await _read_image(image, "image")
    mask_bytes = await _read_image(mask, "mask")

    # Pour cette première version, réutiliser SANS MODIFICATION la fonction
    # preprocess_image(path, path) déjà validée par les golden tests.
    # Les fichiers temporaires sont automatiquement supprimés après inférence.
    with TemporaryDirectory(prefix="covid-api-") as directory:
        image_path = Path(directory) / "image.png"
        mask_path = Path(directory) / "mask.png"
        image_path.write_bytes(image_bytes)
        mask_path.write_bytes(mask_bytes)
        try:
            batch = make_batch(preprocess_image(image_path, mask_path))
        except (ValueError, cv2.error) as error:
            raise HTTPException(status_code=422, detail="Prétraitement impossible : image ou masque invalide") from error

    started = perf_counter()
    result = np.asarray(predict_image(app.state.model, batch), dtype=np.float64)
    elapsed_ms = (perf_counter() - started) * 1000
    if result.shape != (4,) or not np.all(np.isfinite(result)) or not np.isclose(result.sum(), 1.0, atol=1e-4):
        raise HTTPException(status_code=500, detail="Sortie du modèle invalide")
    best = int(np.argmax(result))
    return PredictResponse(
        predicted_class=CLASS_NAMES[best],
        confidence=float(result[best]),
        probabilities={name: float(value) for name, value in zip(CLASS_NAMES, result)},
        model_version="v1.0",
        inference_ms=round(elapsed_ms, 2),
    )
