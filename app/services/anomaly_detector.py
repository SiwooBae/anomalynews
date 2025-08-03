import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class NewsAnomalyDetector:
    """
    Anomaly detection service for news articles using Local Outlier Factor (LoF).
    
    LoF is an unsupervised anomaly detection algorithm that measures the local density
    deviation of a given data point with respect to its neighbors. Points with lower
    local density are considered anomalies.
    
    This class supports both outlier detection (novelty=False) and novelty detection 
    (novelty=True) modes. The workflow differs between the two modes:
    
    - Outlier detection: Use fit_predict() directly on the data
    - Novelty detection: Use fit() on training data, then predict() on new data
    """
    
    def __init__(
        self,
        contamination: float = 'auto',
        n_neighbors: int = 20,
        algorithm: str = 'auto',
        leaf_size: int = 30,
        metric: str = 'minkowski',
        p: int = 2,
        metric_params: Optional[Dict] = None,
        novelty: bool = False,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None
    ):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination: The proportion of outliers in the dataset. Can be 'auto' or float in (0, 0.5]
            n_neighbors: Number of neighbors to use for LoF calculation
            algorithm: Algorithm used for nearest neighbors ('auto', 'ball_tree', 'kd_tree', 'brute')
            leaf_size: Leaf size passed to BallTree or KDTree
            metric: Metric to use for distance computation
            p: Power parameter for the Minkowski metric
            metric_params: Additional keyword arguments for the metric function
            novelty: If True, use novelty detection mode; if False, use outlier detection mode
            n_jobs: Number of parallel jobs to run for neighbors search
            random_state: Random state for reproducibility (not actually used by LOF but kept for API consistency)
        """
        # Validate contamination parameter
        if isinstance(contamination, str) and contamination != 'auto':
            raise ValueError("contamination must be 'auto' or a float in (0, 0.5]")
        if isinstance(contamination, float) and not (0 < contamination <= 0.5):
            raise ValueError("contamination must be a float in (0, 0.5]")
            
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.metric_params = metric_params
        self.novelty = novelty
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # Initialize the LoF model with correct parameters
        self.lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            contamination=contamination,
            novelty=novelty,
            n_jobs=n_jobs
        )
        
        # Store training data and results
        self.embeddings = None
        self.articles = None
        self.anomaly_scores = None
        self.anomaly_labels = None
        self.is_fitted = False
    
    def prepare_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Prepare embeddings for anomaly detection.
        
        Args:
            embeddings: Raw embeddings from the model
            
        Returns:
            Processed embeddings ready for LoF
        """
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        
        # Normalize embeddings if they're not already normalized
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / (norms + 1e-8)
        
        return normalized_embeddings
    
    def fit(self, embeddings: np.ndarray, articles: List[Dict[str, Any]]) -> 'NewsAnomalyDetector':
        """
        Fit the LoF model to the embeddings.
        
        This method is only used in novelty detection mode (novelty=True).
        For outlier detection (novelty=False), use detect_anomalies() directly.
        
        Args:
            embeddings: Article embeddings (n_samples, n_features)
            articles: List of article dictionaries
            
        Returns:
            Self for method chaining
        """
        if not self.novelty:
            raise ValueError("fit() should only be used in novelty detection mode (novelty=True). "
                           "For outlier detection, use detect_anomalies() directly.")
        
        if len(embeddings) != len(articles):
            raise ValueError("Number of embeddings must match number of articles")
        
        # Prepare embeddings
        processed_embeddings = self.prepare_embeddings(embeddings)
        
        # Fit LoF model (only for novelty detection)
        self.lof.fit(processed_embeddings)
        
        # Store data
        self.embeddings = processed_embeddings
        self.articles = articles
        self.is_fitted = True
        
        logger.info(f"Fitted LoF model with {len(articles)} articles for novelty detection")
        return self
    
    def detect_anomalies(
        self, 
        embeddings: np.ndarray, 
        articles: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Detect anomalies in news articles using LoF.
        
        This method works differently depending on the novelty setting:
        - novelty=False (outlier detection): Performs fit_predict() on the given data
        - novelty=True (novelty detection): Uses predict() on new data (requires fit() first)
        
        Args:
            embeddings: Article embeddings (n_samples, n_features)
            articles: List of article dictionaries
            
        Returns:
            Tuple of (anomaly_scores, anomaly_labels, anomalous_articles)
        """
        if len(embeddings) != len(articles):
            raise ValueError("Number of embeddings must match number of articles")
        
        # Prepare embeddings
        processed_embeddings = self.prepare_embeddings(embeddings)
        
        if self.novelty:
            # Novelty detection mode - requires fit() to be called first
            if not self.is_fitted:
                raise ValueError("For novelty detection, you must call fit() first")
            
            # Use predict() for new data
            anomaly_labels = self.lof.predict(processed_embeddings)
            # Get decision scores for novelty detection
            anomaly_scores = self.lof.decision_function(processed_embeddings)
            
        else:
            # Outlier detection mode - use fit_predict()
            anomaly_labels = self.lof.fit_predict(processed_embeddings)
            # Get negative outlier factor (only available after fit_predict in outlier detection mode)
            anomaly_scores = self.lof.negative_outlier_factor_
            
            # Store data for outlier detection
            self.embeddings = processed_embeddings
            self.articles = articles
            self.is_fitted = True
        
        # Store results
        self.anomaly_scores = anomaly_scores
        self.anomaly_labels = anomaly_labels
        
        # Get anomalous articles (label = -1 for anomalies)
        anomalous_indices = np.where(anomaly_labels == -1)[0]
        anomalous_articles = [articles[i] for i in anomalous_indices]
        
        # Add anomaly scores to articles
        for i, article in enumerate(articles):
            article['anomaly_score'] = float(anomaly_scores[i])
            article['is_anomaly'] = bool(anomaly_labels[i] == -1)
        
        logger.info(f"Detected {len(anomalous_articles)} anomalous articles out of {len(articles)} total")
        
        return anomaly_scores, anomaly_labels, anomalous_articles
    
    def predict_new_data(
        self, 
        embeddings: np.ndarray, 
        articles: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Predict anomalies for new data using a fitted novelty detection model.
        
        This is a convenience method that's essentially the same as detect_anomalies()
        when novelty=True, but makes the intent clearer.
        
        Args:
            embeddings: Article embeddings (n_samples, n_features)
            articles: List of article dictionaries
            
        Returns:
            Tuple of (anomaly_scores, anomaly_labels, anomalous_articles)
        """
        if not self.novelty:
            raise ValueError("predict_new_data() is only available for novelty detection (novelty=True)")
        
        return self.detect_anomalies(embeddings, articles)
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the anomaly detection results.
        
        Returns:
            Dictionary with anomaly detection summary
        """
        if not self.is_fitted or self.anomaly_scores is None:
            raise ValueError("Model must be fitted and anomalies detected first")
        
        total_articles = len(self.articles)
        anomalous_count = np.sum(self.anomaly_labels == -1)
        normal_count = total_articles - anomalous_count
        
        return {
            'total_articles': total_articles,
            'anomalous_articles': int(anomalous_count),
            'normal_articles': int(normal_count),
            'anomaly_rate': float(anomalous_count / total_articles),
            'mean_anomaly_score': float(np.mean(self.anomaly_scores)),
            'std_anomaly_score': float(np.std(self.anomaly_scores)),
            'min_anomaly_score': float(np.min(self.anomaly_scores)),
            'max_anomaly_score': float(np.max(self.anomaly_scores))
        }
    
    def get_top_anomalies(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Get the top-k most anomalous articles.
        
        Args:
            top_k: Number of top anomalies to return
            
        Returns:
            List of top anomalous articles with their scores
        """
        if not self.is_fitted or self.anomaly_scores is None:
            raise ValueError("Model must be fitted and anomalies detected first")
        
        # Get indices of anomalous articles
        anomalous_indices = np.where(self.anomaly_labels == -1)[0]
        
        if len(anomalous_indices) == 0:
            return []
        
        # Sort by anomaly score (lower scores = more anomalous)
        sorted_indices = anomalous_indices[np.argsort(self.anomaly_scores[anomalous_indices])]
        
        # Get top-k anomalies
        top_indices = sorted_indices[:top_k]
        
        top_anomalies = []
        for idx in top_indices:
            article = self.articles[idx].copy()
            article['anomaly_score'] = float(self.anomaly_scores[idx])
            article['rank'] = len(top_anomalies) + 1
            top_anomalies.append(article)
        
        return top_anomalies
    
    def analyze_anomaly_clusters(self, n_clusters: int = 5) -> Dict[str, Any]:
        """
        Analyze clusters of anomalous articles to understand patterns.
        
        Args:
            n_clusters: Number of clusters to analyze
            
        Returns:
            Dictionary with cluster analysis results
        """
        if not self.is_fitted or self.anomaly_scores is None:
            raise ValueError("Model must be fitted and anomalies detected first")
        
        # Get anomalous articles
        anomalous_indices = np.where(self.anomaly_labels == -1)[0]
        
        if len(anomalous_indices) < n_clusters:
            return {"error": f"Not enough anomalous articles ({len(anomalous_indices)}) for {n_clusters} clusters"}
        
        # Get embeddings of anomalous articles
        anomalous_embeddings = self.embeddings[anomalous_indices]
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(anomalous_embeddings)
        
        # Find most similar pairs
        n_anomalies = len(anomalous_indices)
        similarities = []
        
        for i in range(n_anomalies):
            for j in range(i + 1, n_anomalies):
                similarities.append({
                    'article1_idx': anomalous_indices[i],
                    'article2_idx': anomalous_indices[j],
                    'similarity': float(similarity_matrix[i, j])
                })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            'total_anomalies': len(anomalous_indices),
            'similarity_analysis': similarities[:10],  # Top 10 most similar pairs
            'mean_similarity': float(np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]))
        }
    
    def get_anomaly_threshold(self, percentile: float = 90) -> float:
        """
        Get anomaly score threshold at a given percentile.
        
        Args:
            percentile: Percentile for threshold (0-100)
            
        Returns:
            Anomaly score threshold
        """
        if not self.is_fitted or self.anomaly_scores is None:
            raise ValueError("Model must be fitted and anomalies detected first")
        
        return float(np.percentile(self.anomaly_scores, percentile))
    
    def filter_by_threshold(self, threshold: float) -> List[Dict[str, Any]]:
        """
        Filter articles by anomaly score threshold.
        
        Args:
            threshold: Anomaly score threshold (articles with scores <= threshold are considered anomalous)
            
        Returns:
            List of articles that meet the threshold criteria
        """
        if not self.is_fitted or self.anomaly_scores is None:
            raise ValueError("Model must be fitted and anomalies detected first")
        
        filtered_indices = np.where(self.anomaly_scores <= threshold)[0]
        
        filtered_articles = []
        for idx in filtered_indices:
            article = self.articles[idx].copy()
            article['anomaly_score'] = float(self.anomaly_scores[idx])
            filtered_articles.append(article)
        
        return filtered_articles 