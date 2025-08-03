# AnomalyNews Project Overview

This document provides a comprehensive overview of the AnomalyNews project, designed to help any LLM understand the codebase, its purpose, and how to contribute effectively.

## 1. Project Purpose

AnomalyNews is a Python-based application that detects anomalies in news articles. It fetches news from an external API, generates text embeddings for the articles, and then uses the Local Outlier Factor (LoF) algorithm to identify articles that are unusual or different from the rest.

The core workflow is as follows:

1.  **Fetch News:** Retrieve news articles (title and snippet) from the FinancialModelingPrep API.
2.  **Generate Embeddings:** Use a pre-trained language model (e.g., Qwen-3 Embedding) to convert the news articles into numerical representations (embeddings).
3.  **Detect Anomalies:** Apply the LoF algorithm from scikit-learn to the embeddings to calculate an anomaly score for each article.
4.  **Present Results:** Identify and display the most anomalous articles based on their scores.

## 2. Project Structure

The project is organized into the following key directories and files:

```
/Users/siwoobae/Projects/anomalynews/
├───app/
│   ├───main.py
│   ├───services/
│   │   ├───anomaly_detector.py
│   │   └───news_api.py
│   └───utils/
│       └───embedding_helpers.py
├───run_lof_anomaly_detection.py
├───anomalynews_experiment.ipynb
├───pyproject.toml
├───README.md
└───results/
```

### Key Files and Directories:

*   **`run_lof_anomaly_detection.py`**: The main script for running the anomaly detection process from the command line. It handles fetching articles, generating embeddings, running the LoF algorithm, and saving the results.
*   **`app/`**: The core application package.
    *   **`main.py`**: A simple entry point for the application, mainly for demonstration and testing purposes.
    *   **`services/news_api.py`**: Contains the `FinancialModelingPrepNews` class, which is a wrapper for the FinancialModelingPrep API. It handles fetching news articles and requires an `FMP_API_KEY` environment variable.
    *   **`services/anomaly_detector.py`**: Implements the `NewsAnomalyDetector` class, which encapsulates the LoF algorithm. It provides methods for fitting the model, detecting anomalies, and summarizing the results.
    *   **`utils/embedding_helpers.py`**: Contains helper functions for formatting text before generating embeddings, specifically for the Qwen-3 model.
*   **`anomalynews_experiment.ipynb`**: A Jupyter notebook for experimenting with the anomaly detection process.
*   **`pyproject.toml`**: The project's configuration file, which lists all the dependencies.
*   **`results/`**: The directory where the output of the anomaly detection script is saved, including a summary of the results and a list of all articles with their anomaly scores.

## 3. Dependencies

The project's dependencies are listed in the `pyproject.toml` file. Key libraries include:

*   **`torch`**: For deep learning and tensor operations.
*   **`transformers`** and **`sentence-transformers`**: For using pre-trained language models from the Hugging Face Hub.
*   **`scikit-learn`**: For the LoF algorithm and other machine learning utilities.
*   **`httpx`**: For making HTTP requests to the news API.
*   **`python-dotenv`**: For managing environment variables.
*   **`numpy`** and **`pandas`**: For numerical operations and data manipulation.

## 4. How to Run

The main entry point for the anomaly detection process is the `run_lof_anomaly_detection.py` script. It can be run from the command line with the following arguments:

```bash
python run_lof_anomaly_detection.py [--articles 100] [--contamination 0.1] [--neighbors 20]
```

*   `--articles`: The number of articles to fetch.
*   `--contamination`: The expected proportion of anomalies in the dataset.
*   `--neighbors`: The number of neighbors to use for the LoF algorithm.

Before running the script, you need to set the following environment variables in a `.env` file:

*   `FMP_API_KEY`: Your API key for the FinancialModelingPrep API.
*   `HF_TOKEN`: Your Hugging Face Hub token for accessing the embedding model.

## 5. Key Concepts and Implementation Details

### 5.1. News Fetching

The `FinancialModelingPrepNews` class in `app/services/news_api.py` is responsible for fetching news articles. It uses the `httpx` library to make requests to the API and handles different types of news (general, stock, and press releases).

### 5.2. Embedding Generation

The `run_lof_anomaly_detection.py` script uses the `InferenceClient` from the `huggingface_hub` library to generate embeddings for the news articles. The default model is `Qwen/Qwen3-Embedding-8B`. The `embedding_helpers.py` file contains a helper function to format the text before it is passed to the model.

### 5.3. Anomaly Detection

The `NewsAnomalyDetector` class in `app/services/anomaly_detector.py` implements the LoF algorithm. It takes the embeddings as input and calculates an anomaly score for each article. The class is well-documented and explains the difference between outlier detection and novelty detection.

### 5.4. Results

The script saves the results in the `results/` directory in two JSON files:

*   `lof_summary.json`: Contains a summary of the anomaly detection results, including the number of anomalies, the mean anomaly score, and the top 10 most anomalous articles.
*   `articles_with_scores.json`: Contains a list of all the articles with their corresponding anomaly scores and a flag indicating whether they are an anomaly or not.