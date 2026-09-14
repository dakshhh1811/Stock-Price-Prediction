import os
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def fetch_news_sentiment(company):
    api_key = os.environ.get('NEWS_API_KEY')
    if not api_key:
        return {
           "company": company,
            "sentiment_score": None,
            "summary": "News sentiment unavailable: NEWS_API_KEY is not set.",
            "top_news": []
        }
    url = f'https://newsapi.org/v2/everything?q={company}&language=en&sortBy=publishedAt&pageSize=5&apiKey={api_key}'

try:
       response = requests.get(url, timeout=10)
       response.raise_for_status()
       data = response.json()
except requests.RequestException:
        return {
            "company": company,
            "sentiment_score": None,
            "summary": f"Could not fetch news for {company} (request failed).",
            "top_news": []
        }
if data.get('status') != 'ok' or not data.get('articles'):
        return {
            "company": company,
            "sentiment_score": None,
            "summary": f"No recent news found for {company}.",
            "top_news": []
        }

    analyzer = SentimentIntensityAnalyzer()
    scores = []
    top_news = []

    for article in data['articles']:
        text = (article.get('title') or '') + ' ' + (article.get('description') or '')
        vs = analyzer.polarity_scores(text)
        scores.append(vs['compound'])
        top_news.append({
            "title": article.get('title'),
           "description": article.get('description'),
           "url": article.get('url')
        })

    avg_score = sum(scores) / len(scores)

    if avg_score >= 0.05:
        sentiment = "Positive"
    elif avg_score <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return {
        "company": company,
        "sentiment_score": round(avg_score, 3),
        "summary": f"Overall sentiment is {sentiment} based on recent news.",
        "top_news": top_news
    }

