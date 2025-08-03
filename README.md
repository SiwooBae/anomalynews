## Anomalynews: news article anomaly detection

### How it works (currently):
- get news article title and snippet from API
- use Qwen-3 Embedding (even 8B is locally runnable if brave enough)
- PCA for dimensionality reduction
- Use sklearn local outlier factor and find top 5% anomalous news articles
- Show them

