import io
import base64
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import yfinance as yf
from django.conf import settings
from django.shortcuts import render, redirect
from .utils import fetch_news_sentiment


# Load once, from an absolute path so it doesn't depend on cwd.
+MODEL_PATH = settings.BASE_DIR / 'ml' / 'stock_predictor.pkl'
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None

def redirect_to_company(request):
    company = request.GET.get('company', 'TCS.NS')
    return redirect('stock_predict', company=company)


def stock_predict(request, company="TCS.NS"):
    

    company = request.GET.get('company', company).upper()

    if model is None:
       return render(request, "predictor/prediction.html", {
            "company": company,
            "predicted_price": "Model not available",
            "chart": None,
            "news_sentiment": "No news available"
        })
    # Download recent stock data
    data = yf.download(company, period="1mo", interval="1d")

    if data.empty or len(data) < 10:
        return render(request, "predictor/prediction.html", {
            "company": company,
            "predicted_price": "Insufficient data",
            "chart": None,
            "news_sentiment": "No news available"
        })

    # Feature engineering
    data['Prev_Close'] = data['Close'].shift(1)
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data = data.dropna()

    # Prepare latest features
    latest = data.iloc[-1]
    X_new = latest[['Prev_Close', 'MA5']].values.reshape(1, -1)

    # Predict price
    predicted_price = model.predict(X_new)[0]

    # Predict for all rows for chart
    X_all = data[['Prev_Close', 'MA5']]
    predicted_all = model.predict(X_all)

    # Plot actual vs predicted price chart
    plt.figure(figsize=(10,5))
    plt.plot(data.index, data['Close'], label='Actual Price')
    plt.plot(data.index, predicted_all, label='Predicted Price')
    plt.legend()
    plt.title(f"{company} Actual vs Predicted Prices (Last 1 Month)")
    plt.xlabel("Date")
    plt.ylabel("Price")

    # Save plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    chart_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Get news sentiment (dummy for now)
    sentiment_data = fetch_news_sentiment(company)
    news_summary = sentiment_data.get('summary', 'No news available')

    context = {
        "company": company,
        "predicted_price": round(float(predicted_price), 2),
        "chart": chart_base64,
        "news_sentiment": news_summary,
        "prediction_date": datetime.today().strftime("%Y-%m-%d"),
        "today_date": datetime.today().strftime("%Y-%m-%d"),
    }

    return render(request, "predictor/prediction.html", context)





def stock_chart(request, company="TCS.NS"):
    company = request.GET.get('company', company).upper()
    stock_data = yf.download(company, period="1mo", interval="1d")

    if stock_data.empty:
        return render(request, 'predictor/stock_chart.html', {'plot_div': None, 'company': company})

    # Plot
    trace1 = go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines+markers', name='Stock Price')
 

    layout = go.Layout(
        title=f'{company} - Price',
        yaxis=dict(title='Stock Price'),
        xaxis=dict(title='Date'),
        legend=dict(x=0, y=1.2)
    )

    fig = go.Figure(data=[trace1], layout=layout)
    chart_div = fig.to_html(full_html=False)

    return render(request, 'predictor/stock_chart.html', {'plot_div': chart_div, 'company': company})

def stock_dashboard(request, company):
    sentiment_data = fetch_news_sentiment(company)
    return render(request, 'predictor/stock_dashboard.html', {
        'sentiment_data': sentiment_data,
        'company': company,
    })


def stock_plot(request):
    # Dummy data for the plot (you can replace this with real stock data)
    x = [1, 2, 3, 4, 5]
    y = [100, 120, 115, 130, 125]

    # Create the plot
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, marker='o', color='blue')
    plt.title('Stock Price Over Time')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.grid(True)

    # Save the plot to a bytes buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    # Encode the image to base64 string
    plot_url = base64.b64encode(image_png).decode('utf-8')

    # Render the template and pass the plot
    return render(request, 'predictor/stock_plot.html', {'plot_url': plot_url})
