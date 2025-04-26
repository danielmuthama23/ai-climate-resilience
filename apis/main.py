from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from typing import Optional
import json
import os

# Initialize FastAPI app
app = FastAPI(
    title="Climate Resilience API",
    description="API for flood prediction and agricultural optimization in Africa",
    version="1.0.0"
)

# Load model and preprocessing components
MODEL_PATH = os.getenv("MODEL_PATH", "./models/trained_models/flood_model.h5")
SCALER_PATH = os.getenv("SCALER_PATH", "./data/processed/scaler_params.json")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH) as f:
        scaler_params = json.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load model/scaler: {str(e)}")

# Define request/response schemas
class SensorInput(BaseModel):
    rainfall: float
    soil_moisture: float
    latitude: float
    longitude: float
    timestamp: Optional[str] = None

class PredictionResponse(BaseModel):
    flood_risk: float
    confidence: float
    risk_category: str

def preprocess_input(data: SensorInput) -> np.ndarray:
    """Preprocess sensor input to match model requirements"""
    # Scale numerical features
    scaled_values = [
        (data.rainfall - scaler_params["mean"][0]) / scaler_params["scale"][0],
        (data.soil_moisture - scaler_params["mean"][1]) / scaler_params["scale"][1]
    ]
    
    # TODO: Add satellite data integration based on coordinates
    # For demo, use random satellite features
    satellite_features = np.random.randn(224*224*3).tolist()
    
    return np.array([satellite_features + scaled_values])

@app.post("/predict/flood", response_model=PredictionResponse)
async def predict_flood_risk(input_data: SensorInput):
    """Make flood prediction based on sensor data and location"""
    try:
        # Prepare input tensor
        processed_input = preprocess_input(input_data)
        
        # Make prediction
        prediction = model.predict(processed_input)
        
        # Interpret results
        risk = float(prediction[0][0])
        confidence = min(abs(risk * 2.5), 1.0)  # Simple confidence heuristic
        
        return {
            "flood_risk": risk,
            "confidence": confidence,
            "risk_category": "high" if risk > 0.7 else "medium" if risk > 0.4 else "low"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Service health endpoint"""
    return {
        "status": "OK",
        "model_version": "1.0",
        "ready": True
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)