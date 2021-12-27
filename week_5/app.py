"""
Practice FastAPI
"""

# author: Seungyun Baek

from fastapi import FastAPI
from infer import ColaONNXPredictor

from typing import List

app = FastAPI(title="MLOps Basics App")
predictor = ColaONNXPredictor("./models/model.onnx")


@app.get("/")
async def home() -> str:
    """
    Home page
    """
    return "<h2>This is a sample NLP Project</h2>"


@app.get("/predict")
async def get_prediction(text: str) -> str:
    result = predictor.predict(text)
    return str(result)
