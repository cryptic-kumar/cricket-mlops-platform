from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .data import DataLayer, FEATURES
from .serving import ServingLayer

app = FastAPI(title='Cricket MLOps Serving API')

class PredictionRequest(BaseModel):
    client_id: str = Field(min_length=1, max_length=100)
    features: list[float]

_serving = None

def serving():
    global _serving
    if _serving is None:
        _serving = ServingLayer(DataLayer().reference_features())
    return _serving

@app.get('/health')
def health(): return {'status': 'ok'}

@app.get('/features')
def features(): return {'features': FEATURES}

@app.post('/predict')
def predict(req: PredictionRequest):
    result = serving().predict(req.client_id, req.features)
    if 'error' in result: raise HTTPException(status_code=429 if result['error']=='rate_limited' else 400, detail=result)
    return result
