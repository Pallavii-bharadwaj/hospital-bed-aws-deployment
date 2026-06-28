import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Hospital Bed Predictor")
model = joblib.load("model.pkl")

class Patient(BaseModel):
    age: float
    cci: float
    los_index: float
    prior_adm_12m: float
    crp: float
    creatinine: float
    has_carer: int
    imd: float
    site: int

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(p: Patient):
    row = pd.DataFrame([p.dict()])
    pred = float(model.predict(row)[0])
    return {"predicted_bed_days_30d": round(pred, 2)}
