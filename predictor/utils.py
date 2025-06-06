import requests
from textblob import TextBlob



import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def fetch_news_sentiment(company):
    api_key = '12624afdd2c04efd8a28b2313a9d67e1'
    url = f'https://newsapi.org/v2/everything?q={company}&language=en&sortBy=publishedAt&pageSize=5&apiKey={api_key}'

    response = requests.get(url)
    data = response.json()

    if data['status'] != 'ok' or not data['articles']:
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
        text = article['title'] + ' ' + (article['description'] or '')
        vs = analyzer.polarity_scores(text)
        scores.append(vs['compound'])
        top_news.append({
            "title": article['title'],
            "description": article['description'],
            "url": article['url']
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

