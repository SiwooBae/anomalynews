#!/usr/bin/env python3
"""
Local Outlier Factor (LoF) Anomaly Detection for News Articles
Uses the new BaseAnomalyDetector/LOFAnomalyDetector API via NewsAnomalyService.
"""

import argparse
import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Any, Tuple

import torch
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.news_api import FinancialModelingPrepNews
from app.services.anomaly_detector_service import NewsAnomalyService
from app.models.anomaly_detector_models import AnomalyResult

def setup_environment():
    load_dotenv()
    torch.set_default_device("mps")
    api_key = os.getenv("FMP_API_KEY")
    hf_token = os.getenv("HF_TOKEN")
    if not api_key:
        raise ValueError("FMP_API_KEY environment variable is required")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is required")
    client = InferenceClient(provider="auto", api_key=hf_token)
    embedding_model = "Qwen/Qwen3-Embedding-8B"
    return client, embedding_model

def fetch_articles(limit: int = 100, news_type: str = "general") -> List[Dict[str, Any]]:
    print(f"📰 Fetching {limit} {news_type} news articles...")
    with FinancialModelingPrepNews() as news_api:
        articles = news_api.get_articles_for_anomaly_detection(limit=limit, news_type=news_type)
    print(f"✅ Found {len(articles)} articles")
    return articles

def generate_embeddings(client, embedding_model: str, articles: List[Dict[str, Any]]) -> np.ndarray:
    print("🔍 Generating embeddings...")
    texts = [f"title: {a['title']}\ncontent: {a['snippet']}" for a in articles]
    embs = client.feature_extraction(texts, model=embedding_model, normalize=True)
    embs = np.array(embs)
    print(f"✅ Generated embeddings with shape: {embs.shape}")
    return embs

def detect_anomalies(
    embeddings: np.ndarray,
    articles: List[Dict[str, Any]],
    contamination: float = 0.1,
    n_neighbors: int = 20,
    metric: str = "cosine",
) -> Tuple[NewsAnomalyService, AnomalyResult]:
    print(f"🔍 Detecting anomalies (contamination={contamination}, n_neighbors={n_neighbors})...")
    svc = NewsAnomalyService(
        contamination=contamination,
        n_neighbors=n_neighbors,
        novelty=False,
        metric=metric,
        reducer_components=None
    )
    result: AnomalyResult = svc.detect(embeddings, articles)
    print(f"✅ Detected {len(result.indices_anomalous)} anomalous articles")
    return svc, result

def display_results(svc: NewsAnomalyService, result: AnomalyResult):
    summary = svc.get_anomaly_summary()
    print("\n" + "="*60)
    print("📊 ANOMALY DETECTION RESULTS")
    print("="*60)
    print(f"Total Articles:        {summary['total_items']}")
    print(f"Anomalous Articles:    {summary['anomalous_items']}")
    print(f"Normal Articles:       {summary['normal_items']}")
    print(f"Anomaly Rate:          {summary['anomaly_rate']:.2%}")
    print(f"Mean Anomaly Score:    {summary['mean_anomaly_score']:.4f}")
    print(f"Std Anomaly Score:     {summary['std_anomaly_score']:.4f}")

    top = svc.get_top_anomalies(top_k=10)
    print(f"\n🚨 TOP 10 MOST ANOMALOUS ARTICLES:")
    print("-" * 60)
    for i, art in enumerate(top, 1):
        print(f"\n{i}. Score: {art['anomaly_score']:.4f}")
        print(f"   Title:   {art['title']}")
        print(f"   Snippet: {art['snippet'][:150]}...")
        print(f"   Source:  {art.get('site', 'Unknown')}")
        print(f"   Date:    {art.get('publishedDate', 'Unknown')}")

def save_results(
    svc: NewsAnomalyService,
    result: AnomalyResult,
    articles: List[Dict[str, Any]],
    output_dir: str = "results"
):
    os.makedirs(output_dir, exist_ok=True)
    summary = svc.get_anomaly_summary()
    top = svc.get_top_anomalies(top_k=10)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "top_anomalies": top,
        "parameters": {
            "contamination": svc.detector.model.contamination,
            "n_neighbors": svc.detector.model.n_neighbors,
            "metric": svc.detector.model.metric,
            "novelty": svc.detector.novelty,
        }
    }
    with open(f"{output_dir}/lof_summary.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)

    all_with_scores = []
    for idx, art in enumerate(articles):
        a = art.copy()
        a["anomaly_score"] = float(result.scores[idx])
        a["is_anomaly"] = bool(result.labels[idx] == -1)
        all_with_scores.append(a)
    with open(f"{output_dir}/articles_with_scores.json", "w") as f:
        json.dump(all_with_scores, f, indent=2, default=str)

    print(f"\n✅ Results saved to '{output_dir}/'")

def main():
    p = argparse.ArgumentParser(description="LoF Anomaly Detection for News Articles")
    p.add_argument("--articles",    type=int,   default=100)
    p.add_argument("--contamination", type=float, default=0.1)
    p.add_argument("--neighbors",   type=int,   default=20)
    p.add_argument("--news-type",   choices=["general","stock","press"], default="general")
    p.add_argument("--metric",      choices=["cosine", "euclidean"], default="cosine")
    p.add_argument("--output-dir",  default="results")
    args = p.parse_args()

    try:
        print("🚀 Starting LoF Anomaly Detection")
        client, emb_model = setup_environment()
        arts = fetch_articles(args.articles, args.news_type)
        embs = generate_embeddings(client, emb_model, arts)
        svc, result = detect_anomalies(
            embs, arts,
            contamination=args.contamination,
            n_neighbors=args.neighbors,
            metric=args.metric,
        )
        display_results(svc, result)
        save_results(svc, result, arts, args.output_dir)
        print("\n🎉 Done!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
