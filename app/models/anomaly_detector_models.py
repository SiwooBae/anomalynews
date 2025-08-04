from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

@dataclass
class AnomalyResult:
    scores: np.ndarray               # higher = more anomalous (standardized)
    labels: np.ndarray               # -1 anomaly, +1 normal
    indices_anomalous: np.ndarray    # indices where labels == -1
    annotated_metadata: Optional[List[Dict[str, Any]]] = None