# 🧠 Iris Prediction — End-to-End ML Project

A compact yet complete **machine learning project** built around the classic **Iris dataset**.  
The goal is to demonstrate an **end-to-end ML workflow** — from data ingestion and preprocessing to model training, evaluation, and inference — using clean, production-style Python code and a modular structure.

---

## 🎯 Project Intent

This project aims to:

- Develop a reproducible ML pipeline using scikit-learn.
- Structure the codebase following best practices for maintainability.
- Generate and store trained models and evaluation metrics.
- Provide an easy-to-use CLI for model training and inference.
- Enable seamless future extensions into experiment tracking, model serving, or CI/CD.

---

## 🏗️ Project Structure

```bash
iris-predict/
  ├── src/iris_predict/
  │   ├── __init__.py
  │   ├── data.py          # Load and return the Iris dataset
  │   ├── features.py      # Preprocessing pipelines (scaling, imputation)
  │   ├── model.py         # Model builders (logreg, rf, svm)
  │   ├── train.py         # Training, evaluation, artifact saving
  │   ├── infer.py         # Prediction interface for new samples
  │   └── utils.py         # Config management
  ├── configs/
  │   └── default.yaml     # Global configuration
  ├── notebooks/
  │   ├── 00_eda.ipynb     # Exploratory data analysis
  │   └── 01_baseline.ipynb
  ├── tests/
  │   └── test_model.py    # Unit tests for training/inference
  ├── artifacts/           # Saved models and metrics (gitignored)
  ├── pyproject.toml       # Project metadata, dependencies, CLI entrypoint
  └── README.md
