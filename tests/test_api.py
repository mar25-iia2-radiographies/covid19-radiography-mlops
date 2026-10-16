"""Tests HTTP (modèle simulé : rapides et indépendants du fichier .keras)."""
from io import BytesIO

import cv2
import numpy as np
from fastapi.testclient import TestClient

import app.main as api_module


def _png(value=120):
    ok, buffer = cv2.imencode(".png", np.full((32, 32), value, dtype=np.uint8))
    assert ok
    return buffer.tobytes()


def test_health_and_info(monkeypatch):
    monkeypatch.setattr(api_module, "load_covid_model", lambda: object())
    with TestClient(api_module.app) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/model/info").json()["requires_mask"] is True


def test_predict_returns_four_classes(monkeypatch):
    monkeypatch.setattr(api_module, "load_covid_model", lambda: object())
    monkeypatch.setattr(api_module, "predict_image", lambda model, batch: np.array([0.8, 0.1, 0.08, 0.02]))
    with TestClient(api_module.app) as client:
        response = client.post("/predict", files={
            "image": ("image.png", BytesIO(_png()), "image/png"),
            "mask": ("mask.png", BytesIO(_png(255)), "image/png"),
        })
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["predicted_class"] == "COVID"
        assert len(data["probabilities"]) == 4
        assert data["confidence"] == 0.8


def test_reject_invalid_upload(monkeypatch):
    monkeypatch.setattr(api_module, "load_covid_model", lambda: object())
    with TestClient(api_module.app) as client:
        response = client.post("/predict", files={
            "image": ("bad.txt", BytesIO(b"not an image"), "text/plain"),
            "mask": ("mask.png", BytesIO(_png(255)), "image/png"),
        })
        assert response.status_code == 415
