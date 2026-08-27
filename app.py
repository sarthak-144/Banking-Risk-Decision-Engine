from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd

# 1. Initialize FastAPI app
app = FastAPI(title="Banking Risk & LTV Decision Engine", version="1.0")

# 2. Define the exact same PyTorch architecture used in training
class RiskAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(RiskAutoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 4), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, input_dim))
    def forward(self, x):
        return self.decoder(self.encoder(x))

# 3. Load Models into memory on startup
print("Loading models into memory...")
try:
    # Load XGBoost LTV Model
    ltv_model = xgb.XGBRegressor()
    ltv_model.load_model("models/xgboost_ltv.json")

    # Load Scaler and PyTorch Autoencoder
    risk_scaler = joblib.load("models/risk_scaler.pkl")
    risk_model = RiskAutoencoder(input_dim=3)
    risk_model.load_state_dict(torch.load("models/autoencoder.pth", weights_only=True))
    risk_model.eval() 
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models. Did you run Step 3? Details: {e}")

# 4. Define incoming JSON payload schemas using Pydantic
class LTVPayload(BaseModel):
    avg_transaction_size: float
    max_transaction_size: float

class RiskPayload(BaseModel):
    massive_transfer_count: float
    avg_balance_discrepancy: float
    max_balance_discrepancy: float

# 5. Define the Endpoints
@app.get("/")
def health_check():
    return {"status": "Decision Engine API is running."}

@app.post("/predict-ltv")
def predict_ltv(payload: LTVPayload):
    """Predicts Customer Lifetime Value based on transaction history."""
    # Convert JSON payload to DataFrame for XGBoost
    data = pd.DataFrame([payload.model_dump()])
    prediction = ltv_model.predict(data)[0]
    
    return {
        "predicted_ltv_score": float(prediction),
        "business_action": "Premium Offer Eligible" if prediction > 150000 else "Standard Tier"
    }

@app.post("/predict-fraud")
def predict_fraud(payload: RiskPayload):
    """Calculates anomaly score. High reconstruction error indicates fraud."""
    # Extract features and scale them
    features = np.array([[
        payload.massive_transfer_count,
        payload.avg_balance_discrepancy,
        payload.max_balance_discrepancy
    ]])
    scaled_features = risk_scaler.transform(features)
    
    # Convert to tensor and pass through Autoencoder
    tensor_data = torch.FloatTensor(scaled_features)
    with torch.no_grad():
        reconstructed = risk_model(tensor_data)
        # Calculate Mean Squared Error (Anomaly Score)
        mse = torch.mean((tensor_data - reconstructed) ** 2).item()
    
    # In a real system, threshold is determined by a validation set. 
    # We'll use an arbitrary strict threshold here based on our training loss (~0.27).
    is_fraud = bool(mse > 1.5)
    
    return {
        "anomaly_score": float(mse),
        "is_fraud_detected": is_fraud,
        "action": "Block Transaction" if is_fraud else "Approve"
    }