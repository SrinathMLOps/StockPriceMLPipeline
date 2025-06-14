# 📁 models/train_model.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train():
    df = pd.read_csv('data/features.csv')

    df['target'] = (df['pct_change'].shift(-1) > 0).astype(int)  # 1 if price goes up next hour
    df.dropna(inplace=True)

    X = df[['ma_3', 'pct_change', 'volatility']]
    y = df['target']

    X_train, X_te...

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.2f}")

    joblib.dump(model, 'models/stock_model.pkl')
    print("📦 Model saved to models/stock_model.pkl")

if __name__ == "__main__":
    train()
