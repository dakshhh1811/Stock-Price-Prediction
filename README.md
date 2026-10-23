Stock Price Prediction

A Django web application that predicts stock prices using machine learning and adds a news-based GenAI layer. Users enter an NSE stock ticker, view predicted vs historical prices, see news sentiment, and ask questions about the stock using a Retrieval-Augmented Generation (RAG) assistant.

Key Features
Predicted vs actual price chart for any ticker (data from Yahoo Finance via yfinance)
Two models: a Random Forest regressor (Scikit-learn) and an LSTM (RNN) time-series model (TensorFlow)
Model evaluation using MAE, RMSE and R²
News sentiment using VADER and Hugging Face Transformers (FinBERT) on headlines from NewsAPI
RAG assistant (LangChain + FAISS vector database + LLM) that answers questions about a stock using recent news
Interactive price charts with Plotly
Django backend with SQLite and REST API endpoints
Docker support for easy deployment
Tech Stack

Python, Django, Scikit-learn, TensorFlow, LSTM, Hugging Face Transformers, LangChain, FAISS, LLMs, yfinance, Pandas, NumPy, Matplotlib, Plotly, SQLite, REST APIs, Docker, Git

How It Works
yfinance data -> preprocessing -> feature engineering -> model training -> Django backend -> charts + sentiment + RAG chat
                                                                              ^
                              NewsAPI headlines -> sentiment (VADER / FinBERT) + FAISS index

Features used by the models: previous day's close and 5-day moving average (Random Forest); a 60-day window of closing prices (LSTM).

Models

Model	Library	Role
Random Forest Regressor	Scikit-learn	Main model used by the web app (trained on TCS, Infosys, Reliance, HDFC Bank, ICICI Bank)
LSTM (RNN)	TensorFlow / Keras	Time-series model for comparison
Project Structure
Stock-Price-Prediction/
├── Stock_price_prediction/       # Django project (settings, root urls.py, wsgi/asgi)
├── predictor/                    # Django app (views, templates, news sentiment utils)
│   ├── views.py                  # Prediction, chart, dashboard and RAG views
│   ├── utils.py                  # News fetching + sentiment (VADER)
│   ├── api.py                    # REST API endpoints (JSON)
│   └── templates/predictor/      # HTML templates
├── ml/                           # Machine learning code
│   ├── train_model.py            # Trains the Random Forest model (saves stock_predictor.pkl)
│   ├── train_lstm.py             # Trains the LSTM model (saves stock_lstm.keras)
│   ├── evaluate.py               # MAE, RMSE and R² for both models
│   ├── sentiment_finbert.py      # FinBERT news sentiment (Hugging Face)
│   └── rag.py                    # RAG pipeline (LangChain + FAISS + LLM)
├── Dockerfile                    # Docker setup
├── .envexample                   # Example environment variables
├── manage.py                     # Django management script
├── requirements.txt              # Dependencies
└── README.md                     # This file
1. Installation
A. Clone the Repository
bash
git clone https://github.com/dakshhh1811/Stock-Price-Prediction.git
cd Stock-Price-Prediction
B. Create a Virtual Environment
bash
# Create a new environment
python -m venv venv

# Activate the environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
C. Install Dependencies
bash
pip install -r requirements.txt
D. Set Up API Keys

Copy .envexample to .env and add your keys:

NEWS_API_KEY=your_newsapi_key_here
OPENAI_API_KEY=your_llm_key_here
NEWS_API_KEY (from newsapi.org) is needed for news sentiment and the RAG assistant.
OPENAI_API_KEY (or the key for the LLM you use) is needed only for the RAG assistant.

Without a NEWS_API_KEY, the app still works and shows "News sentiment unavailable".

2. How to Run the Program
A. Train the Models
bash
# Random Forest (Scikit-learn)
python ml/train_model.py

# LSTM (TensorFlow)
python ml/train_lstm.py

Each script downloads 2 years of daily data with yfinance, trains the model, prints the test metrics and saves the model file.

B. Evaluate the Models
bash
python ml/evaluate.py

Prints MAE, RMSE and R² for the Random Forest and LSTM models on the test set.

C. Start the Web App
bash
python manage.py migrate
python manage.py runserver

Open http://127.0.0.1:8000/predictor/predict/TCS.NS/ in your browser.

URL	What it shows
/predictor/predict/<ticker>/	Predicted price + actual vs predicted chart + news sentiment
/predictor/chart/<ticker>/	Interactive Plotly price chart (last 1 month)
/predictor/dashboard/<ticker>/	News sentiment dashboard with top headlines
/predictor/ask/<ticker>/	RAG assistant: ask questions about the stock
/predictor/api/predict/<ticker>/	REST API: prediction as JSON

Example tickers: TCS.NS, INFY.NS, RELIANCE.NS, HDFCBANK.NS, ICICIBANK.NS

3. News Sentiment and RAG Assistant

Sentiment: recent headlines for the company are fetched from NewsAPI. Each headline is scored with VADER (fast baseline) and FinBERT (finance-specific Transformer from Hugging Face). The average score is shown as Positive, Neutral or Negative.

RAG assistant:

News articles are split into chunks and converted to embeddings.
The embeddings are stored in a FAISS vector database.
When you ask a question, LangChain retrieves the most relevant chunks.
An LLM answers using only those chunks, with prompt instructions to avoid making things up.
4. Run with Docker
bash
# Build the image
docker build -t stock-predictor .

# Run the app
docker run -p 8000:8000 --env-file .env stock-predictor

Then open http://127.0.0.1:8000/predictor/predict/TCS.NS/.

5. Documentation
Features: previous close and 5-day moving average (Random Forest); 60-day price window (LSTM).
Evaluation metrics: MAE, RMSE, R².
Data source: Yahoo Finance (yfinance), 2 years of daily data.
Disclaimer: this project is for learning purposes only and is not financial advice.
