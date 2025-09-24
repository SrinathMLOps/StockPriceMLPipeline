# 📁 models/train_model.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import mlflow
import mlflow.sklearn
import os

def train():
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("stock_price_prediction")
    
    with mlflow.start_run():
        # Load and prepare data
        df = pd.read_csv('../dags/data/stock_raw.csv')
        
        # Feature engineering
        df['ma_3'] = df['close'].astype(float).rolling(window=3).mean()
        df['pct_change_1d'] = df['close'].pct_change()
        df['volume'] = df['volume'].astype(float)
        df = df.dropna()
        
        X = df[['ma_3', 'pct_change_1d', 'volume']]
        y = df['close'].astype(float)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)
        
        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log model
        mlflow.sklearn.log_model(model, "stock_model")
        
        print(f"✅ Accuracy: {acc:.4f}")
        print(f"✅ Precision: {precision:.4f}")
        print(f"✅ Recall: {recall:.4f}")
        print(f"✅ F1 Score: {f1:.4f}")
        
        # Save model locally
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, 'stock_model.pkl')
        print("📦 Model saved to stock_model.pkl")
        print("🔬 Experiment logged to MLflow")

if __name__ == "__main__":
    train()
