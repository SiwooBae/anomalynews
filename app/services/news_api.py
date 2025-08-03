import os
from typing import List, Dict, Any, Optional
from datetime import date
import httpx
from dotenv import load_dotenv

load_dotenv()

class NewsAPIError(Exception):
    pass

class FinancialModelingPrepNews:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FMP_API_KEY")
        if not self.api_key:
            raise NewsAPIError("API key is required. Set FMP_API_KEY environment variable or pass api_key parameter.")
        
        self.base_url = "https://financialmodelingprep.com/stable/news"
        self.client = httpx.Client(timeout=30.0)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()
    
    def close(self):
        self.client.close()
    
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        params["apikey"] = self.api_key
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise NewsAPIError(f"HTTP error {e.response.status_code}: {e.response.text}")
        except httpx.RequestError as e:
            raise NewsAPIError(f"Request error: {str(e)}")
        except Exception as e:
            raise NewsAPIError(f"Unexpected error: {str(e)}")
    
    def get_general_news(
        self, 
        page: int = 0, 
        limit: int = 20,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get latest general news articles.
        
        Args:
            page: Page number (default: 0)
            limit: Number of articles per page (max: 250, default: 20)
            from_date: Start date in YYYY-MM-DD format (optional)
            to_date: End date in YYYY-MM-DD format (optional)
        """
        params = {"page": page, "limit": min(limit, 250)}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
            
        return self._make_request("general-latest", params)
    
    def get_press_releases(
        self,
        page: int = 0,
        limit: int = 20,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get latest company press releases.
        
        Args:
            page: Page number (default: 0)
            limit: Number of articles per page (max: 250, default: 20)
            from_date: Start date in YYYY-MM-DD format (optional)
            to_date: End date in YYYY-MM-DD format (optional)
        """
        params = {"page": page, "limit": min(limit, 250)}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
            
        return self._make_request("press-releases-latest", params)
    
    def get_stock_news(
        self,
        page: int = 0,
        limit: int = 20,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get latest stock market news.
        
        Args:
            page: Page number (default: 0)
            limit: Number of articles per page (max: 250, default: 20)
            from_date: Start date in YYYY-MM-DD format (optional)
            to_date: End date in YYYY-MM-DD format (optional)
        """
        params = {"page": page, "limit": min(limit, 250)}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
            
        return self._make_request("stock-latest", params)
    
    def search_press_releases(
        self,
        symbols: str,
        page: int = 0,
        limit: int = 20,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search press releases for specific company symbols.
        
        Args:
            symbols: Company symbol(s) (e.g., "AAPL" or "AAPL,MSFT")
            page: Page number (default: 0)
            limit: Number of articles per page (max: 250, default: 20)
            from_date: Start date in YYYY-MM-DD format (optional)
            to_date: End date in YYYY-MM-DD format (optional)
        """
        params = {"symbols": symbols, "page": page, "limit": min(limit, 250)}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
            
        return self._make_request("press-releases", params)
    
    def search_stock_news(
        self,
        symbols: str,
        page: int = 0,
        limit: int = 20,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search stock news for specific company symbols.
        
        Args:
            symbols: Company symbol(s) (e.g., "AAPL" or "AAPL,MSFT")
            page: Page number (default: 0)
            limit: Number of articles per page (max: 250, default: 20)
            from_date: Start date in YYYY-MM-DD format (optional)
            to_date: End date in YYYY-MM-DD format (optional)
        """
        params = {"symbols": symbols, "page": page, "limit": min(limit, 250)}
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
            
        return self._make_request("stock", params)
    
    def get_articles_for_anomaly_detection(
        self, 
        limit: int = 100,
        news_type: str = "general"
    ) -> List[Dict[str, str]]:
        """
        Get processed articles ready for anomaly detection.
        
        Args:
            limit: Number of articles to fetch
            news_type: Type of news ("general", "stock", "press")
        
        Returns:
            List of processed articles with title, content, url, publishedDate, site
        """
        if news_type == "general":
            articles = self.get_general_news(limit=limit)
        elif news_type == "stock":
            articles = self.get_stock_news(limit=limit)
        elif news_type == "press":
            articles = self.get_press_releases(limit=limit)
        else:
            raise NewsAPIError(f"Invalid news_type: {news_type}. Use 'general', 'stock', or 'press'")
        
        processed_articles = []
        for article in articles:
            if isinstance(article, dict) and article.get('title') and article.get('text'):
                processed_articles.append({
                    'title': article['title'],
                    'snippet': article['text'],
                    'url': article.get('url', ''),
                    'publishedDate': article.get('publishedDate', ''),
                    'site': article.get('site', ''),
                    'publisher': article.get('publisher', ''),
                    'symbol': article.get('symbol', '')
                })
        
        return processed_articles