import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

ticker = "TCS.NS"
data = yf.download(ticker, period="1y", interval="1d")

data['Prev_Close'] = data['Close'].shift(1)
data['MA5'] = data['Close'].rolling(window=5).mean()
data = data.dropna()

X = data[['Prev_Close', 'MA5']]
y = data['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

joblib.dump(model, 'ml/stock_predictor.pkl')
print("Model training done and saved as stock_predictor.pkl")
