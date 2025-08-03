import torch
from services.news_api import FinancialModelingPrepNews, NewsAPIError


def main():
    print("Hello from anomalynews!")
    print("PyTorch version:", torch.__version__)
    print("MPS available (Apple Silicon):", torch.backends.mps.is_available())
    
    # Demo the news API wrapper
    try:
        with FinancialModelingPrepNews() as news_api:
            print(f"\n📰 Fetching latest general news...")
            articles = news_api.get_articles_for_anomaly_detection(limit=5, news_type="general")
            
            print(f"Found {len(articles)} articles for anomaly detection:")
            for i, article in enumerate(articles[:3], 1):
                print(f"\n{i}. {article['title']}")
                print(f"   Publisher: {article['publisher']}")
                print(f"   Published: {article['publishedDate']}")
                print(f"   Content preview: {article['snippet']}")
            
            
    except NewsAPIError as e:
        print(f"⚠️  News API Error: {e}")
        print("Make sure to set your FMP_API_KEY in the .env file")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
