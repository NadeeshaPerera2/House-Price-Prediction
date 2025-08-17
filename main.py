from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np

# Load model
bundle = joblib.load("house_model.pkl")
model = bundle["model"]
features = bundle["features"]
metrics = bundle["metrics"]

app = FastAPI(title="House Price Prediction API")

# Input schema
class HouseInput(BaseModel):
    sqft: float = Field(..., description="House square footage")
    bedrooms: int = Field(..., description="Number of bedrooms")
    bathrooms: int = Field(..., description="Number of bathrooms")
    age: int = Field(..., description="Age of the house in years")

# Output schema
class PredictionOutput(BaseModel):
    predicted_price: float

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "House Price API running"}

@app.post("/predict", response_model=PredictionOutput)
def predict(data: HouseInput):
    try:
        arr = np.array([[data.sqft, data.bedrooms, data.bathrooms, data.age]])
        pred = model.predict(arr)[0]
        return PredictionOutput(predicted_price=float(pred))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info")
def model_info():
    return {
        "model_type": "LinearRegression",
        "problem_type": "regression",
        "features": features,
        "metrics": metrics
    }
