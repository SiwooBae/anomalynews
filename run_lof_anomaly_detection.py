#!/usr/bin/env python3
"""
Local Outlier Factor (LoF) Anomaly Detection for News Articles

This script demonstrates how to use LoF to detect anomalous news articles.
LoF is an unsupervised anomaly detection algorithm that measures the local
density deviation of a given data point with respect to its neighbors.

Usage:
    python run_lof_anomaly_detection.py [--articles 100] [--contamination 0.1] [--neighbors 20]
"""

import argparse
import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Any

import torch
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.news_api import FinancialModelingPrepNews
from app.services.anomaly_detector import NewsAnomalyDetector

def setup_environment():
    """Set up the environment and API clients."""
    load_dotenv()
    torch.set_default_device("mps")
    
    # Check for required environment variables
    api_key = os.getenv("FMP_API_KEY")
    hf_token = os.getenv("HF_TOKEN")
    
    if not api_key:
        raise ValueError("FMP_API_KEY environment variable is required")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required")
    
    # Initialize clients
    client = InferenceClient(
        provider="auto",
        api_key=hf_token,
    )
    embedding_model = "Qwen/Qwen3-Embedding-8B"
    
    return client, embedding_model

def fetch_articles(limit: int = 100, news_type: str = "general") -> List[Dict[str, Any]]:
    """Fetch news articles for anomaly detection."""
    print(f"📰 Fetching {limit} {news_type} news articles...")
    
    with FinancialModelingPrepNews() as news_api:
        articles = news_api.get_articles_for_anomaly_detection(limit=limit, news_type=news_type)
    
    print(f"✅ Found {len(articles)} articles")
    return articles

def generate_embeddings(client, embedding_model: str, articles: List[Dict[str, Any]]) -> np.ndarray:
    """Generate embeddings for articles."""
    print("🔍 Generating embeddings...")
    
    # Prepare article text
    articles_text = []
    for article in articles:
        text = f"title: {article['title']}\ncontent: {article['snippet']}"
        articles_text.append(text)
    
    # Generate embeddings
    embeddings = client.feature_extraction(
        articles_text,
        model=embedding_model,
        normalize=True
    )
    
    embeddings = np.array(embeddings)
    print(f"✅ Generated embeddings with shape: {embeddings.shape}")
    
    return embeddings

def detect_anomalies(
    embeddings: np.ndarray,
    articles: List[Dict[str, Any]],
    contamination: float = 0.1,
    n_neighbors: int = 20
) -> tuple:
    """Detect anomalies using LoF."""
    print(f"🔍 Detecting anomalies with contamination={contamination}, neighbors={n_neighbors}...")
    
    # Initialize anomaly detector
    detector = NewsAnomalyDetector(
        contamination=contamination,
        n_neighbors=n_neighbors,
        novelty=False,  # Use outlier detection mode (default)
        random_state=42,
        metric="cosine"
    )
    
    # Detect anomalies
    anomaly_scores, anomaly_labels, anomalous_articles = detector.detect_anomalies(
        embeddings, articles
    )
    
    print(f"✅ Detected {len(anomalous_articles)} anomalous articles")
    
    return detector, anomaly_scores, anomaly_labels, anomalous_articles

def display_results(detector: NewsAnomalyDetector, articles: List[Dict[str, Any]]):
    """Display anomaly detection results."""
    summary = detector.get_anomaly_summary()
    
    print("\n" + "="*60)
    print("📊 ANOMALY DETECTION RESULTS")
    print("="*60)
    
    print(f"Total Articles: {summary['total_articles']}")
    print(f"Anomalous Articles: {summary['anomalous_articles']}")
    print(f"Normal Articles: {summary['normal_articles']}")
    print(f"Anomaly Rate: {summary['anomaly_rate']:.2%}")
    print(f"Mean Anomaly Score: {summary['mean_anomaly_score']:.4f}")
    print(f"Std Anomaly Score: {summary['std_anomaly_score']:.4f}")
    
    # Display top anomalies
    top_anomalies = detector.get_top_anomalies(top_k=10)
    
    print(f"\n🚨 TOP 10 MOST ANOMALOUS ARTICLES:")
    print("-" * 60)
    
    for i, article in enumerate(top_anomalies, 1):
        print(f"\n{i}. Anomaly Score: {article['anomaly_score']:.4f}")
        print(f"   Title: {article['title']}")
        print(f"   Content: {article['snippet'][:150]}...")
        print(f"   Source: {article.get('site', 'Unknown')}")
        print(f"   Date: {article.get('publishedDate', 'Unknown')}")

def save_results(
    detector: NewsAnomalyDetector,
    articles: List[Dict[str, Any]],
    anomaly_scores: np.ndarray,
    anomaly_labels: np.ndarray,
    output_dir: str = "results"
):
    """Save results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare results data
    summary = detector.get_anomaly_summary()
    top_anomalies = detector.get_top_anomalies(top_k=10)
    
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'summary': summary,
        'top_anomalies': top_anomalies,
        'parameters': {
            'contamination': detector.contamination,
            'n_neighbors': detector.n_neighbors,
            'algorithm': detector.algorithm,
            'metric': detector.metric
        }
    }
    
    # Save summary results
    with open(f'{output_dir}/lof_summary.json', 'w') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    # Save all articles with scores
    articles_with_scores = []
    for i, article in enumerate(articles):
        article_data = article.copy()
        article_data['anomaly_score'] = float(anomaly_scores[i])
        article_data['is_anomaly'] = bool(anomaly_labels[i] == -1)
        articles_with_scores.append(article_data)
    
    with open(f'{output_dir}/articles_with_scores.json', 'w') as f:
        json.dump(articles_with_scores, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to '{output_dir}/' directory")
    print(f"  - lof_summary.json: Summary and top anomalies")
    print(f"  - articles_with_scores.json: All articles with anomaly scores")

def main():
    """Main function to run LoF anomaly detection."""
    parser = argparse.ArgumentParser(description="LoF Anomaly Detection for News Articles")
    parser.add_argument("--articles", type=int, default=100, help="Number of articles to fetch")
    parser.add_argument("--contamination", type=float, default=0.1, help="Expected proportion of anomalies")
    parser.add_argument("--neighbors", type=int, default=20, help="Number of neighbors for LoF")
    parser.add_argument("--news-type", default="general", choices=["general", "stock", "press"], 
                       help="Type of news to fetch")
    parser.add_argument("--output-dir", default="results", help="Output directory for results")
    
    args = parser.parse_args()
    
    try:
        print("🚀 Starting LoF Anomaly Detection for News Articles")
        print("=" * 60)
        
        # Setup environment
        client, embedding_model = setup_environment()
        
        # Fetch articles
        articles = fetch_articles(args.articles, args.news_type)
        
        # Generate embeddings
        embeddings = generate_embeddings(client, embedding_model, articles)
        
        # Detect anomalies
        detector, anomaly_scores, anomaly_labels, anomalous_articles = detect_anomalies(
            embeddings, articles, args.contamination, args.neighbors
        )
        
        # Display results
        display_results(detector, articles)
        
        # Save results
        save_results(detector, articles, anomaly_scores, anomaly_labels, args.output_dir)
        
        print("\n🎉 LoF Anomaly Detection completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 