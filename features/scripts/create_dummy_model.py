# scripts/create_dummy_model.py
import joblib
from sklearn.linear_model import LinearRegression
import numpy as np

# Fake training data
X = np.random.rand(100, 3)
y = np.random.rand(100)

model = LinearRegression()
model.fit(X, y)

# Save model to correct location
joblib.dump(model, "models/stock_model.pkl")
print("✅ Model saved to models/stock_model.pkl")
