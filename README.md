# StockPriceMLPipeline
# 🧠 Stock Price Prediction Pipeline

A fully containerized end-to-end machine learning pipeline that automates **stock data ingestion**, **data validation**, **feature engineering**, and **real-time prediction** using:

- Apache Airflow for orchestration
- FastAPI for ML model inference
- Docker Compose for deployment
- Scikit-learn for training a regression model

This project demonstrates real-world **MLOps** practices by connecting a live data source to a deployed prediction service.

---

## 🔧 Tech Stack

- **Apache Airflow** – DAG orchestration and data pipeline automation
- **FastAPI** – Lightweight web server for serving ML predictions
- **Docker Compose** – Service containerization and orchestration
- **Scikit-learn** – Model training and regression
- **Pandas & NumPy** – Data preprocessing
- **PostgreSQL & Redis** – Infrastructure support for Airflow

---

## 🚀 Features

- ⏰ Hourly data ingestion via the [TwelveData API](https://twelvedata.com)
- 🧼 Built-in data validation checks
- 🧪 Feature engineering: 3-period moving average, percent change, volume
- 📦 Trained regression model using Linear Regression
- 🔮 Real-time predictions via FastAPI `/predict` endpoint
- 🐳 Fully containerized using Docker Compose

---

## 🛠️ Getting Started

### 1️⃣ Train and Save the Model

Before running the system, generate a `.pkl` model file using:

```bash
python features/scripts/create_real_model.py


This script uses previously fetched stock data to train a regression model and save it as:

models/stock_model.pkl




2️⃣ Start All Services
Use Docker Compose to build and run all containers:

docker compose up --build

This launches:

Airflow (webserver & scheduler)

FastAPI (model API)

PostgreSQL and Redis

3️⃣ Access Services

| Tool        | URL                                            |
| ----------- | ---------------------------------------------- |
| Airflow UI  | [http://localhost:8080](http://localhost:8080) |
| FastAPI API | [http://localhost:8000](http://localhost:8000) |


📈 API Usage
🔮 Predict Endpoint
Send a POST request to /predict with these 3 inputs:

| Parameter       | Type  | Description                            |
| --------------- | ----- | -------------------------------------- |
| `ma_3`          | float | 3-period moving average of stock price |
| `pct_change_1d` | float | Percent change in price since last day |
| `volume`        | float | Trading volume                         |

✅ Example Request

POST http://localhost:8000/predict?ma_3=1.2&pct_change_1d=0.5&volume=10000

✅ Example Response
{
  "prediction": 147.23
}


🧠 Model Training Script
📁 features/scripts/create_real_model.py

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
import os

# Load raw stock data
df = pd.read_csv("dags/data/stock_raw.csv")

# Feature engineering
df['ma_3'] = df['close'].astype(float).rolling(window=3).mean()
df['pct_change_1d'] = df['close'].pct_change()
df['volume'] = df['volume'].astype(float)
df = df.dropna()

X = df[['ma_3', 'pct_change_1d', 'volume']]
y = df['close'].astype(float)

model = LinearRegression()
model.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/stock_model.pkl")
print("✅ Model trained and saved to models/stock_model.pkl")

🧩 Project Structure

.
├── dags/
│   └── stock_pipeline.py               # Airflow DAG
├── serving/
│   ├── main.py                         # FastAPI app
│   ├── Dockerfile
│   └── requirements.txt
├── models/
│   └── stock_model.pkl                 # Trained ML model
├── features/
│   ├── create_features.py
│   └── scripts/
│       └── create_real_model.py        # Model training script
├── docker-compose.yml
├── requirements.txt
└── README.md

📄 Summary for the Project
Project: Stock Price Prediction Pipeline using MLOps
Description: Developed an end-to-end ML pipeline that automatically pulls stock data hourly, validates it, extracts features, trains a regression model, and serves predictions via a FastAPI endpoint. Used Docker Compose to orchestrate services including Airflow, Redis, and PostgreSQL.
Skills Demonstrated: MLOps, Airflow, FastAPI, Docker, Model Deployment, Regression, Data Engineering, REST APIs

📜 License
This project is licensed under the MIT License.


---

Let me know if you'd like help with:
- Custom GitHub cover images
- Deploying this to Render, Azure, or Hugging Face Spaces
- Automating `.env` and `.pkl` creation with shell scripts

It's fully production-ready — amazing work! 🚀

