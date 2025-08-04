# news_anomaly_service.py
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from app.core.anomaly_detectors.lof_anomaly_detector import LOFAnomalyDetector
from app.models.anomaly_detector_models import AnomalyResult

class NewsAnomalyService:
    """
    Convenience façade oriented around {embeddings, articles} pairs.
    """

    def __init__(
        self,
        *,
        novelty: bool = False,
        contamination: float = "auto",
        n_neighbors: int = 10,
        reducer_components: int = 256,   # example dimensionality reduction
        use_scaler: bool = False,
        metric: str = "cosine",
    ):
        scaler = StandardScaler() if use_scaler else None
        reducer = PCA(n_components=reducer_components) if reducer_components else None

        self.detector = LOFAnomalyDetector(
            contamination=contamination,
            n_neighbors=n_neighbors,
            novelty=novelty,
            use_l2_normalize=True,
            scaler=scaler,
            reducer=reducer,
            metric=metric
        )

    @staticmethod
    def _ensure_2d(X: np.ndarray) -> np.ndarray:
        return X.reshape(1, -1) if X.ndim == 1 else X

    def fit(self, embeddings: np.ndarray, articles: List[Dict[str, Any]]) -> "NewsAnomalyService":
        if len(embeddings) != len(articles):
            raise ValueError("Number of embeddings must match number of articles")
        embeddings = self._ensure_2d(embeddings)
        self.detector.fit(embeddings, metadata=articles)
        return self

    def detect(
        self, embeddings: np.ndarray, articles: List[Dict[str, Any]]
    ) -> AnomalyResult:
        if len(embeddings) != len(articles):
            raise ValueError("Number of embeddings must match number of articles")
        embeddings = self._ensure_2d(embeddings)
        res = self.detector.detect(embeddings, metadata=articles)
        return res

    # Pass-through utilities for convenience
    def get_anomaly_summary(self):
        return self.detector.get_anomaly_summary()

    def get_top_anomalies(self, top_k: int = 10):
        return self.detector.get_top_anomalies(top_k=top_k)

    def get_anomaly_threshold(self, percentile: float = 90.0):
        return self.detector.get_anomaly_threshold(percentile=percentile)

    def filter_by_threshold(self, threshold: float):
        return self.detector.filter_by_threshold(threshold=threshold)

    def analyze_anomaly_clusters(self, max_pairs: int = 10):
        return self.detector.analyze_anomaly_clusters(max_pairs=max_pairs)
