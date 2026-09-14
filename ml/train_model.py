"""
Trains the stock price prediction model.

Adjust TICKERS to match the companies you expect users to query most —
the app lets users request any ticker, so training on just one company
(the old behavior) makes predictions for everything else meaningless.
"""
from pathlib import Path

import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

TICKERS = ["TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
MODEL_PATH = Path(__file__).resolve().parent / "stock_predictor.pkl"


def build_features(ticker: str) -> pd.DataFrame:
    data = yf.download(ticker, period="2y", interval="1d")
    if data.empty:
        return pd.DataFrame()
    data['Prev_Close'] = data['Close'].shift(1)
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data = data.dropna()
    return data[['Prev_Close', 'MA5', 'Close']]


def main():
    frames = [build_features(t) for t in TICKERS]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise RuntimeError("No training data was downloaded for any ticker.")

    combined = pd.concat(frames, ignore_index=True)
    X = combined[['Prev_Close', 'MA5']]
    y = combined['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(f"Trained on {len(combined)} rows across {len(frames)} ticker(s).")
    print(f"Test MAE: {mean_absolute_error(y_test, preds):.2f}")
    print(f"Test R^2: {r2_score(y_test, preds):.3f}")

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()