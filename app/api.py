# app/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from src.predictor import load_predictor # Use the same cached loader

app = FastAPI()
predictor = load_predictor()

class URLRequest(BaseModel):
    url: str

@app.post("/analyze")
async def analyze_url(request: URLRequest):
    risk_score, _ = predictor.predict(request.url)
    return {"url": request.url, "risk_score": risk_score}