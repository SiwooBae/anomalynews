# lof_anomaly.py
from typing import Optional, Tuple
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from app.core.anomaly_detectors.base_anomaly_detector import BaseAnomalyDetector

class LOFAnomalyDetector(BaseAnomalyDetector):
    def __init__(
        self,
        *,
        contamination: float = "auto",
        n_neighbors: int = 20,
        algorithm: str = "auto",
        leaf_size: int = 30,
        metric: str = "minkowski",
        p: int = 2,
        metric_params: Optional[dict] = None,
        novelty: bool = False,
        n_jobs: Optional[int] = None,
        # pipeline knobs
        use_l2_normalize: bool = False,
        scaler=None,
        reducer=None,
    ) -> None:
        super().__init__(novelty=novelty, use_l2_normalize=use_l2_normalize, scaler=scaler, reducer=reducer)

        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            contamination=contamination,
            novelty=novelty,
            n_jobs=n_jobs,
        )

    def _supports_fit_predict(self) -> bool:
        # LOF supports fit_predict in outlier mode only
        return not self.novelty

    def _fit_impl(self, Xp: np.ndarray) -> None:
        # novelty=True requires separate fit; novelty=False *can* use fit_predict, but we also support explicit fit.
        self.model.fit(Xp)

    def _fit_predict_impl(self, Xp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Only valid when novelty=False
        labels = self.model.fit_predict(Xp)  # -1 anomalies, +1 normal
        # LOF negative_outlier_factor_ is more negative for more anomalous
        nof = self.model.negative_outlier_factor_
        scores = -nof  # higher score => more anomalous
        return scores, labels

    def _predict_impl(self, Xp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.novelty:
            labels = self.model.predict(Xp)
            df = self.model.decision_function(Xp)  # higher for inliers
            scores = -df  # higher => more anomalous
        else:
            # If someone calls predict path in outlier mode after _fit_impl
            labels = self.model.fit_predict(Xp)
            nof = self.model.negative_outlier_factor_
            scores = -nof
        return scores, labels
