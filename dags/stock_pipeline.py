# 📁 dags/stock_pipeline.py

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import requests
import pandas as pd
import os

# ✅ Task 1: Pull stock data from API
def pull_stock_data():
    url = f"https://api.twelvedata.com/time_series?symbol=AAPL&interval=1h&apikey={os.getenv('TWELVE_API_KEY', 'demo')}"
    
    response = requests.get(url)
    data = response.json()

    # ✅ Check if API returned 'values'
    if 'values' not in data:
        raise ValueError(f"API Error: 'values' not found in response: {data}")

    df = pd.DataFrame(data['values'])

    # ✅ Create output folder if it doesn't exist
    output_dir = "/opt/airflow/dags/data"
    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(os.path.join(output_dir, "stock_raw.csv"), index=False)
    print("✅ Stock data saved to CSV")

# ✅ Task 2: Validate data
def validate_stock_data():
    df = pd.read_csv('/opt/airflow/dags/data/stock_raw.csv')
    assert 'datetime' in df.columns and 'close' in df.columns, "Missing required columns"
    assert not df.isnull().values.any(), "Data contains nulls"
    print("✅ Data validation passed")

# ✅ Task 3: Feature engineering
def run_feature_engineering():
    from features.create_features import create_features
    create_features('/opt/airflow/dags/data/stock_raw.csv', '/opt/airflow/dags/data/features.csv')

# ✅ DAG configuration
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'stock_data_pipeline',
    default_args=default_args,
    description='Fetch, validate, and process stock data',
    schedule_interval='@hourly',
    catchup=False
)

# ✅ Task bindings
t1 = PythonOperator(
    task_id='pull_stock_data',
    python_callable=pull_stock_data,
    dag=dag
)

t2 = PythonOperator(
    task_id='validate_stock_data',
    python_callable=validate_stock_data,
    dag=dag
)

t3 = PythonOperator(
    task_id='run_feature_engineering',
    python_callable=run_feature_engineering,
    dag=dag
)

# ✅ DAG flow
t1 >> t2 >> t3
