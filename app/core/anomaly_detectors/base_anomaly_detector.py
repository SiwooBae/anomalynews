# base_anomaly.py
from __future__ import annotations
import abc
from typing import Any, Dict, List, Optional, Tuple, Protocol
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.metrics.pairwise import cosine_similarity
from app.models.anomaly_detector_models import AnomalyResult

class ReducerLike(Protocol):
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "ReducerLike": ...
    def transform(self, X: np.ndarray) -> np.ndarray: ...

class BaseAnomalyDetector(abc.ABC):
    """
    Abstract base for anomaly detectors with a unified API and a pluggable feature pipeline.
    Score convention: higher score = more anomalous (subclasses must conform).
    """

    def __init__(
        self,
        *,
        novelty: bool = False,
        use_l2_normalize: bool = False,
        scaler: Optional[TransformerMixin] = None,
        reducer: Optional[ReducerLike] = None,
    ) -> None:
        self.novelty = novelty
        self.use_l2_normalize = use_l2_normalize
        self.scaler = scaler
        self.reducer = reducer

        # Fitted state / cache
        self._is_fitted: bool = False
        self._X_ref: Optional[np.ndarray] = None            # processed features used in last fit/detect
        self._metadata_ref: Optional[List[Dict[str, Any]]] = None
        self._last_scores: Optional[np.ndarray] = None
        self._last_labels: Optional[np.ndarray] = None

    # ---------- Public API ----------

    def fit(self, X: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> "BaseAnomalyDetector":
        """
        Fit the detector in novelty mode. In outlier mode, this can be a no-op for some detectors,
        but we still fit the pipeline so later detect() can transform consistently.
        """
        Xp = self._preprocess(X, fit_pipeline=True)
        self._fit_impl(Xp)
        self._is_fitted = True
        self._X_ref = Xp
        self._metadata_ref = metadata
        return self

    def detect(self, X: np.ndarray, metadata: Optional[List[Dict[str, Any]]] = None) -> AnomalyResult:
        """
        Detect anomalies on X (and annotate metadata if provided).
        - novelty=True: requires prior fit(); does NOT refit pipeline.
        - novelty=False: fits the pipeline on X (since we treat this batch as the population).
        """
        if self.novelty:
            if not self._is_fitted:
                raise ValueError("For novelty detection, call fit() first.")
            Xp = self._preprocess(X, fit_pipeline=False)
            scores, labels = self._predict_impl(Xp)
        else:
            # Outlier detection path: fit pipeline on X and allow fit_predict if subclass supports it
            Xp = self._preprocess(X, fit_pipeline=True)
            if self._supports_fit_predict():
                scores, labels = self._fit_predict_impl(Xp)
            else:
                self._fit_impl(Xp)
                scores, labels = self._predict_impl(Xp)

        if scores.shape[0] != X.shape[0] or labels.shape[0] != X.shape[0]:
            raise RuntimeError("Detector returned mismatched shapes for scores/labels.")

        idx_anom = np.where(labels == -1)[0]

        annotated = None
        if metadata is not None:
            annotated = []
            for i, m in enumerate(metadata):
                mm = dict(m)
                mm["anomaly_score"] = float(scores[i])
                mm["is_anomaly"] = bool(labels[i] == -1)
                annotated.append(mm)

        # cache last run
        self._X_ref = Xp
        self._metadata_ref = metadata
        self._last_scores = scores
        self._last_labels = labels

        return AnomalyResult(scores=scores, labels=labels, indices_anomalous=idx_anom, annotated_metadata=annotated)

    # ---------- Utilities (shared across detectors) ----------

    def get_anomaly_summary(self) -> Dict[str, Any]:
        if self._last_scores is None or self._last_labels is None:
            raise ValueError("Run detect() or fit()+detect() first.")
        total = self._last_scores.shape[0]
        anomalous = int((self._last_labels == -1).sum())
        normal = total - anomalous
        return {
            "total_items": total,
            "anomalous_items": anomalous,
            "normal_items": normal,
            "anomaly_rate": float(anomalous / total) if total else 0.0,
            "mean_anomaly_score": float(np.mean(self._last_scores)),
            "std_anomaly_score": float(np.std(self._last_scores)),
            "min_anomaly_score": float(np.min(self._last_scores)),
            "max_anomaly_score": float(np.max(self._last_scores)),
        }

    def get_top_anomalies(self, top_k: int = 10) -> List[Dict[str, Any]]:
        if self._last_scores is None or self._last_labels is None or self._metadata_ref is None:
            raise ValueError("Run detect() with metadata to use get_top_anomalies().")
        idx_anom = np.where(self._last_labels == -1)[0]
        if idx_anom.size == 0:
            return []
        # higher score = more anomalous → sort descending
        order = idx_anom[np.argsort(self._last_scores[idx_anom])[::-1]]
        out = []
        for r, i in enumerate(order[:top_k], 1):
            item = dict(self._metadata_ref[i])
            item["anomaly_score"] = float(self._last_scores[i])
            item["rank"] = r
            out.append(item)
        return out

    def get_anomaly_threshold(self, percentile: float = 90.0) -> float:
        if self._last_scores is None:
            raise ValueError("Run detect() first.")
        return float(np.percentile(self._last_scores, percentile))

    def filter_by_threshold(self, threshold: float) -> List[Dict[str, Any]]:
        if self._last_scores is None or self._metadata_ref is None:
            raise ValueError("Run detect() with metadata to use filter_by_threshold().")
        # higher score = more anomalous
        idx = np.where(self._last_scores >= threshold)[0]
        out: List[Dict[str, Any]] = []
        for i in idx:
            m = dict(self._metadata_ref[i])
            m["anomaly_score"] = float(self._last_scores[i])
            m["is_anomaly"] = bool(self._last_labels[i] == -1)
            out.append(m)
        return out

    def analyze_anomaly_clusters(self, max_pairs: int = 10) -> Dict[str, Any]:
        if self._X_ref is None or self._last_labels is None:
            raise ValueError("Run detect() first.")
        idx_anom = np.where(self._last_labels == -1)[0]
        if idx_anom.size < 2:
            return {"total_anomalies": int(idx_anom.size), "message": "Not enough anomalies for pairwise analysis."}
        X_anom = self._X_ref[idx_anom]
        S = cosine_similarity(X_anom)
        # Grab top similar pairs (i<j)
        n = S.shape[0]
        triples = []
        for i in range(n):
            for j in range(i + 1, n):
                triples.append({"i": int(idx_anom[i]), "j": int(idx_anom[j]), "similarity": float(S[i, j])})
        triples.sort(key=lambda t: t["similarity"], reverse=True)
        mean_sim = float(np.mean(S[np.triu_indices(n, k=1)]))
        return {"total_anomalies": int(idx_anom.size), "mean_similarity": mean_sim, "top_pairs": triples[:max_pairs]}

    # ---------- Hooks for subclasses ----------

    def _supports_fit_predict(self) -> bool:
        """Return True if the subclass can do fit+predict in a single call for outlier mode."""
        return False

    @abc.abstractmethod
    def _fit_impl(self, Xp: np.ndarray) -> None:
        """Fit the underlying detector on already preprocessed features."""
        raise NotImplementedError

    @abc.abstractmethod
    def _predict_impl(self, Xp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict on preprocessed X.
        Returns (scores, labels) with the convention: higher scores = more anomalous, labels in {-1, +1}.
        """
        raise NotImplementedError

    def _fit_predict_impl(self, Xp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Optional optimization for outlier mode to do fit+predict in one shot."""
        # Default just calls fit then predict.
        self._fit_impl(Xp)
        return self._predict_impl(Xp)

    # ---------- Feature pipeline ----------

    def _preprocess(self, X: np.ndarray, *, fit_pipeline: bool) -> np.ndarray:
        X = self._ensure_2d(X)
        if self.use_l2_normalize:
            X = self._l2_normalize(X)

        if self.scaler is not None:
            X = self._fit_or_transform(self.scaler, X, fit_pipeline)

        if self.reducer is not None:
            X = self._fit_or_transform(self.reducer, X, fit_pipeline)

        return X

    @staticmethod
    def _ensure_2d(X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            return X.reshape(1, -1)
        return X

    @staticmethod
    def _l2_normalize(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / (norms + eps)

    @staticmethod
    def _fit_or_transform(op: Any, X: np.ndarray, fit: bool) -> np.ndarray:
        if fit:
            op.fit(X)
        return op.transform(X)
