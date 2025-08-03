# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Anomalynews is a news article anomaly detection system that identifies unusual or outlier news articles from news APIs. The system uses machine learning techniques to detect anomalous content patterns.

### Architecture

The application follows this workflow:
1. Fetches news article titles and snippets from news APIs
2. Generates embeddings using Qwen-3 Embedding model (8B variant can run locally)
3. Applies PCA for dimensionality reduction
4. Uses sklearn's Local Outlier Factor to identify the top 5% most anomalous articles
5. Displays the anomalous articles to users

Currently, the main application logic is in `main.py` (basic stub implementation).

## Development Commands

This project uses `uv` as the package manager and build tool.

### Setup and Installation
```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # or `uv shell`
```

### Running the Application
```bash
# Run the main application
uv run python main.py

# Or with activated environment
python main.py
```

### Package Management
```bash
# Add a new dependency
uv add <package-name>

# Add development dependency
uv add --dev <package-name>

# Update dependencies
uv sync --upgrade
```

## Project Structure

- `main.py` - Main application entry point (currently minimal implementation)
- `pyproject.toml` - Project configuration and dependencies
- `uv.lock` - Dependency lock file
- `README.md` - Project description and workflow explanation

## Key Dependencies

The project currently has no declared dependencies in `pyproject.toml`, but based on the README, the planned tech stack includes:
- Qwen-3 Embedding model for text embeddings
- scikit-learn for Local Outlier Factor and PCA
- News API integration (specific API not yet determined)