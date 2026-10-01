"""MathVision local inference API.

Thin HTTP layer over the PROTECTED pipeline in app.main
(run_pipeline / solve_image -> preprocessing -> segmentation ->
SymbolPredictor -> solve_expression). No ML logic duplicated here.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "mathvision_cnn.keras"
METRICS_PATH = ROOT / "models" / "metrics.json"

app = FastAPI(title="MathVision API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174",
                   "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def _predictor():
    from app.recognition.predictor import SymbolPredictor

    try:
        p = SymbolPredictor()
        p.predict_batch([np.zeros((28, 28), np.uint8)])  # warm up
        return p
    except Exception:
        return None


def _metrics() -> dict:
    try:
        if METRICS_PATH.exists():
            return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


@app.get("/api/health")
def health() -> dict:
    online = MODEL_PATH.exists() and _predictor() is not None
    return {"online": online, "model_exists": MODEL_PATH.exists()}


@app.get("/api/model-info")
def model_info() -> dict:
    m = _metrics()
    return {
        "name": "MathVision CNN",
        "task": "Handwritten symbol classification",
        "input": "28 × 28 × 1",
        "classes": "0–9 + − × ÷ = x",
        "inference": "LOCAL",
        "test_accuracy": m.get("test_accuracy"),
        "macro_f1": m.get("macro_f1"),
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    from app.main import solve_image

    raw = await file.read()
    if not raw:
        return JSONResponse({"error": "Empty file."}, status_code=400)
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Could not decode image (use PNG/JPG)."},
                            status_code=400)
    try:
        sol = solve_image(img, _predictor())
    except Exception:
        return JSONResponse({"error": "Could not process that image."}, status_code=422)

    if not sol["crops"]:
        return JSONResponse({
            "expression": "", "solution_kind": "error",
            "solution_message": "No mathematical symbols were detected.",
            "confidence": 0.0, "ms": sol["ms"], "num_symbols": 0,
            "symbols": [],
        })
    res = sol["result"]
    return {
        "expression": sol["expression"],
        "solution_kind": res.kind if res else "error",
        "solution_message": res.message if res else "Recognition unavailable.",
        "confidence": sol["confidence"],
        "ms": sol["ms"],
        "num_symbols": len(sol["crops"]),
        "symbols": [{"label": lab, "confidence": conf}
                    for lab, conf in sol["details"]],
    }
