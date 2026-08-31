from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

class HouseFeatures(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    location_score: float

app = FastAPI()

model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.post("/predict")
async def predict_price(house: HouseFeatures):
    features = np.array([[
        house.area,
        house.bedrooms,
        house.bathrooms,
        house.location_score
    ]])
    
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    
    return {
        "predicted_price": round(prediction, 2),
        "message": f"Dự đoán cho nhà {house.area}m2, {house.bedrooms} phòng ngủ"
    }