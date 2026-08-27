import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sqlalchemy import create_engine
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import numpy as np

os.makedirs("models", exist_ok=True)

print("Extracting features from PostgreSQL...")
engine = create_engine('postgresql://sjain:admin123@localhost:5432/risk_db')
df_risk = pd.read_sql("SELECT * FROM account_risk_features", engine)
df_ltv = pd.read_sql("SELECT * FROM customer_ltv_features", engine)

# ==========================================
# MODEL 1: COMMERCIAL LTV (XGBoost)
# ==========================================
print("\n--- Training Commercial LTV Model (XGBoost) ---")
df_ltv['target_ltv'] = (df_ltv['total_volume'] * 0.1) + (df_ltv['total_transactions'] * 50)

X_ltv = df_ltv[['avg_transaction_size', 'max_transaction_size']]
y_ltv = df_ltv['target_ltv']
X_train_ltv, X_test_ltv, y_train_ltv, y_test_ltv = train_test_split(X_ltv, y_ltv, test_size=0.2, random_state=42)

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
xgb_model.fit(X_train_ltv, y_train_ltv)

# --- NEW: Evaluate the XGBoost Model ---
predictions = xgb_model.predict(X_test_ltv)
rmse = np.sqrt(mean_squared_error(y_test_ltv, predictions))
r2 = r2_score(y_test_ltv, predictions)

print(f"XGBoost Evaluation Metrics:")
print(f" - Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f" - R-squared (R2): {r2:.4f} (1.0 is perfect)")

xgb_model.save_model("models/xgboost_ltv.json")
print("LTV Model saved to models/xgboost_ltv.json")


# ==========================================
# MODEL 2: RISK ANOMALY DETECTION (Autoencoder)
# ==========================================
print("\n--- Training Risk Anomaly Model (PyTorch Autoencoder) ---")
features = ['massive_transfer_count', 'avg_balance_discrepancy', 'max_balance_discrepancy']
normal_accounts = df_risk[df_risk['historical_fraud_flags'] == 0][features]

scaler = StandardScaler()
X_normal_scaled = scaler.fit_transform(normal_accounts)
joblib.dump(scaler, "models/risk_scaler.pkl")

class RiskAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(RiskAutoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 4), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, input_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))

tensor_data = torch.FloatTensor(X_normal_scaled)
autoencoder = RiskAutoencoder(input_dim=len(features))
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.01)

print("Training PyTorch Autoencoder (minimizing reconstruction error)...")
epochs = 20
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = autoencoder(tensor_data)
    loss = criterion(outputs, tensor_data)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss (MSE): {loss.item():.4f}")

torch.save(autoencoder.state_dict(), "models/autoencoder.pth")
print("Risk Model saved to models/autoencoder.pth")
print("\nStep 3 Complete! Models are ready for deployment.")