# 📁 features/create_features.py

import pandas as pd

def create_features(input_path, output_path):
    df = pd.read_csv(input_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime')

    df['close'] = df['close'].astype(float)
    df['ma_3'] = df['close'].rolling(window=3).mean()
    df['pct_change'] = df['close'].pct_change()
    df['volatility'] = df['close'].rolling(window=3).std()

    df.dropna(inplace=True)
    df.to_csv(...

if __name__ == "__main__":
    create_features('data/stock_raw.csv', 'data/features.csv')
