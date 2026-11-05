"""FastAPI service that recognises handwritten digits, running inside a container."""
import json
import os
import platform
import socket
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
SHADES = " .:-=+*#%@"  # pixel value 0..16 -> character
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(MODEL_DIR / "model.joblib")
    state["test"] = joblib.load(MODEL_DIR / "test_set.joblib")
    state["metadata"] = json.loads((MODEL_DIR / "metadata.json").read_text())
    yield
    state.clear()


app = FastAPI(title="Digit recognizer (Dockerized)", lifespan=lifespan)


class Image(BaseModel):
    pixels: list[float] = Field(..., min_length=64, max_length=64,
                                description="8x8 grayscale image, row by row, values 0-16")


def to_ascii(pixels) -> str:
    rows = []
    for r in range(8):
        row = pixels[r * 8:(r + 1) * 8]
        rows.append("".join(SHADES[min(int(v * 9 / 16 + 0.5), 9)] * 2 for v in row))
    return "\n".join(rows)


def classify(pixels) -> tuple[int, float]:
    proba = state["model"].predict_proba([pixels])[0]
    best = int(proba.argmax())
    return best, float(proba[best])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/info")
def info() -> dict:
    """Inside a container the hostname is the container ID."""
    return {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "model": state["metadata"],
    }


@app.post("/predict")
def predict(image: Image) -> dict:
    digit, probability = classify(image.pixels)
    return {
        "digit": digit,
        "probability": round(probability, 4),
        "model_type": state["metadata"]["model_type"],
        "served_by": socket.gethostname(),
    }


@app.get("/sample/{index}", response_class=PlainTextResponse)
def sample(index: int) -> str:
    """Draw a real test image in the terminal and classify it."""
    X, y = state["test"]["X"], state["test"]["y"]
    if not 0 <= index < len(X):
        raise HTTPException(status_code=404, detail=f"index must be 0..{len(X) - 1}")
    digit, probability = classify(X[index])
    return (
        f"{to_ascii(X[index])}\n\n"
        f"true digit : {int(y[index])}\n"
        f"predicted  : {digit}  ({probability:.0%})\n"
        f"model      : {state['metadata']['model_type']}\n"
        f"container  : {socket.gethostname()}\n"
    )
