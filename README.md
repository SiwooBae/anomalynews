# AnomalyNews: News Article Anomaly Detection

This project uses machine learning to identify anomalous news articles. It fetches the latest news, generates text embeddings, and then uses the Local Outlier Factor (LoF) algorithm to find articles that are semantically different from the majority.

## How It Works

1.  **Fetch News**: Retrieves the latest news articles from the [FinancialModelingPrep API](https://site.financialmodelingprep.com/developer/docs).
2.  **Generate Embeddings**: Uses a powerful sentence transformer model (`Qwen/Qwen3-Embedding-8B`) from the Hugging Face Hub to create numerical representations (embeddings) of the news articles.
3.  **Detect Anomalies**: Applies the Local Outlier Factor (LoF) algorithm from `scikit-learn` to the embeddings. LoF is an unsupervised learning algorithm that identifies outliers by measuring the local density deviation of a given data point with respect to its neighbors.
4.  **Analyze Results**: The script outputs the top anomalous articles and saves a detailed report in the `results/` directory.

## Setup and Installation

### Prerequisites

*   Python 3.12 or higher
*   An API key from [FinancialModelingPrep](https://site.financialmodelingprep.com/developer/docs)
*   A Hugging Face Hub token

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/anomalynews.git
    cd anomalynews
    ```

2.  **Create a virtual environment:**
    It's recommended to use a virtual environment.
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    This project uses `uv` or `pip` for dependency management. The dependencies are listed in `pyproject.toml`.
    ```bash
    # With pip
    pip install .

    # Or with uv
    uv pip install -e .
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root of the project and add your API keys:
    ```
    FMP_API_KEY="YOUR_FINANCIALMODELINGPREP_API_KEY"
    HF_TOKEN="YOUR_HUGGING_FACE_HUB_TOKEN"
    ```

## How to Run

The main script for anomaly detection is `run_lof_anomaly_detection.py`. You can run it from the command line with several options:

```bash
python run_lof_anomaly_detection.py --articles 200 --contamination 0.05 --neighbors 25
```

### Command-line Arguments

*   `--articles`: The number of articles to fetch (default: 100).
*   `--contamination`: The expected proportion of anomalies in the dataset (default: 0.1). This is a key parameter for the LoF algorithm.
*   `--neighbors`: The number of neighbors to use for the LoF algorithm (default: 20).
*   `--news-type`: The type of news to fetch. Choices are `general`, `stock`, `press` (default: `general`).
*   `--output-dir`: The directory to save the results (default: `results`).

### Example

To run the detection on 500 general news articles with an expected anomaly rate of 5%:

```bash
python run_lof_anomaly_detection.py --articles 500 --contamination 0.05 --news-type general
```

## Output

The script will print the top 10 most anomalous articles to the console. It will also save two files in the `results/` directory:

*   `lof_summary.json`: A summary of the run, including parameters, anomaly statistics, and the top 10 anomalous articles.
*   `articles_with_scores.json`: A JSON file containing all the fetched articles along with their calculated anomaly scores.