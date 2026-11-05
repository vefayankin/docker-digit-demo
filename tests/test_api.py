"""Runs inside the `test` stage of the Dockerfile: a failing test stops the build."""
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

SAMPLE = json.loads((Path(__file__).parent.parent / "examples" / "digit.json").read_text())


def test_health():
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_model_quality_gate():
    with TestClient(app) as client:
        accuracy = client.get("/info").json()["model"]["accuracy"]
        assert accuracy >= 0.95, f"accuracy {accuracy} is below the 0.95 gate"


def test_predict_known_digit():
    with TestClient(app) as client:
        r = client.post("/predict", json=SAMPLE)
        assert r.status_code == 200
        assert r.json()["digit"] == 3


def test_sample_endpoint():
    with TestClient(app) as client:
        r = client.get("/sample/0")
        assert r.status_code == 200
        assert "predicted" in r.text


def test_rejects_bad_input():
    with TestClient(app) as client:
        assert client.post("/predict", json={"pixels": [0, 1, 2]}).status_code == 422
