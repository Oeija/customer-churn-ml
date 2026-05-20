# Customer Churn Prediction — MLOps Pipeline

End-to-end machine learning pipeline for predicting telecom customer churn. Built with **scikit-learn**, **XGBoost**, **Great Expectations**, **MLflow**, and **FastAPI**.

---

## Overview

This project predicts which customers are likely to churn (leave) a telecom service. It includes:

- **Data validation** with Great Expectations
- **Feature engineering** (tenure groups, service counts, derived metrics)
- **Model training** with configurable classifiers (Random Forest, LightGBM, XGBoost)
- **Hyperparameter tuning** with Optuna
- **Explainability** with SHAP summary plots
- **Experiment tracking** with MLflow
- **Real-time serving** via FastAPI
- **CI/CD** with GitHub Actions
- **Containerisation** with Docker

---

## Architecture

```
Raw CSV
    │
    ▼
┌─────────────────┐
│  Data Ingestion │  ← snake_case columns, coerce types, drop ID
└────────┬────────┘
         │
         ▼
┌────────────────────┐
│ Feature Engineering │  ← tenure_group, num_services, avg_monthly_charge, flags
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ Great Expectations │  ← 43 expectations: columns, categories, not-null
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│ Train / Test Split │  ← stratified, 80/20
└────────┬───────────┘
         │
         ▼
┌──────────────────────────────┐
│ sklearn ColumnTransformer    │  ← SimpleImputer + StandardScaler
│   + OneHotEncoder            │     + binary passthrough
└────────┬─────────────────────┘
         │
         ▼
┌─────────────────┐
│  Model Training │  ← XGBoost with scale_pos_weight
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  Evaluate + SHAP  │  ← classification report, threshold sweep, SHAP plot
└────────┬────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐  ┌─────────────┐
│ MLflow │  │ Local       │
│  Log   │  │ Artifacts   │
└────────┘  └─────────────┘
    │
    ▼
┌──────────────┐
│ FastAPI Serve │  ← POST /predict
└──────────────┘
```

---

## Project Structure

```
customer-churn-ml/
├── config/
│   └── config.yaml                    # Central configuration
├── data/
│   ├── raw/
│   │   └── telco-customer-churn-raw.csv
│   └── processed/                     # Feature-engineered train/test splits (ignored by git)
│       ├── train_features.parquet
│       ├── test_features.parquet
│       ├── train_labels.parquet
│       └── test_labels.parquet
├── great_expectations/
│   └── expectations/
│       └── telco_expectation_suite.json
├── mlflow.db                          # MLflow tracking database (SQLite)
├── notebooks/
│   ├── eda.ipynb                      # Exploratory data analysis
│   └── modeling.ipynb                 # Notebook prototype (refactored)
├── outputs/                           # Generated artifacts (predictions, SHAP plots, evaluation reports)
├── scripts/
│   ├── train.py                       # End-to-end training pipeline
│   ├── evaluate.py                    # Evaluate saved model
│   ├── predict.py                     # Batch prediction CLI
│   └── save_ge_suite.py               # Save GE suite to JSON
├── src/customer_churn_ml/
│   ├── app/
│   │   ├── main.py                    # FastAPI application
│   │   └── schemas.py                 # Pydantic request/response models
│   ├── data/
│   │   ├── ingestion.py               # CSV loading + cleaning
│   │   ├── preprocess.py              # sklearn ColumnTransformer pipeline
│   │   └── validation.py              # Great Expectations wrapper
│   ├── features/
│   │   └── build_features.py          # Derived feature creation
│   ├── models/
│   │   ├── train.py                   # Configurable model trainer
│   │   ├── evaluate.py                # Metrics + threshold sweep
│   │   ├── tune.py                    # Optuna hyperparameter optimisation
│   │   └── predict.py                 # Inference utilities
│   └── utils/
│       ├── config.py                  # YAML loader
│       ├── logger.py                  # Structured logging
│       └── metrics.py                 # Evaluation helpers
├── artifacts/                         # Saved model artifacts (ignored by git)
├── tests/
│   ├── test_ingestion.py
│   ├── test_preprocess.py
│   └── test_api.py
├── .github/workflows/
│   ├── ci.yml                         # pytest + ruff on push/PR
│   └── train.yml                      # Automated training on config changes
├── Dockerfile                         # Multi-stage container
├── pyproject.toml                     # Dependencies + pytest config
└── README.md                          # This file
```

---

## Quick Start

### 1. Install dependencies

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync --all-extras

# Activate virtual environment
source .venv/bin/activate
```

### 2. Train a model

```bash
# Baseline training (uses config hyperparameters)
python scripts/train.py

# With Optuna hyperparameter tuning
python scripts/train.py --tune
```

This will:
1. Load and validate the raw CSV
2. Engineer features
3. Save feature-engineered train/test splits to `data/processed/`
4. Build and fit a sklearn `ColumnTransformer`
5. Train an XGBoost classifier
6. Evaluate on the test set
7. Generate a SHAP summary plot (logged to MLflow artifacts)
8. Log params, metrics, models, and SHAP plot to MLflow
9. Save `preprocessor.joblib`, `model.ubj`, `config.yaml`, and `feature_names.json` to `artifacts/`

### 3. Evaluate the saved model

```bash
# Default: evaluate on the exact processed test split from training
python scripts/evaluate.py

# Or force a fresh random split from raw data (generalization test)
python scripts/evaluate.py --fresh-split
```

Results are printed to console and saved to `outputs/evaluation_YYYYMMDD_HHMMSS.txt`.

### 4. Batch predict on new data

```bash
# Default: saves to outputs/predictions_YYYYMMDD_HHMMSS.csv
python -m scripts.predict --input data/raw/telco-customer-churn-raw.csv

# Or specify a custom output path
python -m scripts.predict \
    --input data/raw/telco-customer-churn-raw.csv \
    --output outputs/my_predictions.csv
```

### 5. Serve via FastAPI

```bash
# Start the server
uvicorn src.customer_churn_ml.app.main:app --reload --port 8000

# Health check
curl http://localhost:8000/health

# Predict on a single customer
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '[{
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "tenure": 12,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "DSL",
    "online_security": "No",
    "online_backup": "Yes",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 29.85,
    "total_charges": 29.85
  }]'
```

### 6. View MLflow experiments

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://localhost:5000 to browse runs, metrics, and model artifacts.

---

## Configuration

All pipeline behaviour is controlled via `config/config.yaml`:

| Section | What it controls |
|---------|-----------------|
| `paths` | Data locations, artifacts directory, MLflow URI |
| `features` | Column groups (numeric, binary, categorical), binary mapping, expected categorical values |
| `feature_engineering` | Tenure bins, service columns, boolean flag toggles |
| `preprocessing` | Imputer strategy, scaler toggle, OneHotEncoder settings |
| `models` | Active model(s), per-model hyperparameters, evaluation threshold(s) |
| `mlflow` | Experiment name, artifact paths |
| `split` | Test size, random state, stratification |

To switch models, edit `models.active`:

```yaml
models:
  active: ["xgboost"]   # options: random_forest, lightgbm, xgboost
```

---

## Testing

```bash
# Run the full test suite
pytest tests/ -v

# Run a specific test file
pytest tests/test_preprocess.py -v
```

Tests cover:
- **Ingestion**: column standardisation, type coercion, blank handling
- **Preprocessing**: binary encoding, ColumnTransformer structure, no NaN output, feature name extraction
- **API**: health endpoint, empty list validation, graceful degradation without artifacts

---

## Docker

Build and run the containerised API:

```bash
# Build
docker build -t churn-api .

# Run (mount model artifacts)
docker run -p 8000:8000 \
  -v $(pwd)/artifacts:/app/artifacts \
  churn-api
```

---

## CI/CD

Two GitHub Actions workflows are configured:

| Workflow | Trigger | What it does |
|----------|---------|-------------|
| `ci.yml` | push / PR to `main` | Runs `pytest` and `ruff` linting |
| `train.yml` | push to `main` (config/src changes) | Runs full training pipeline, uploads model artifacts |

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **sklearn `ColumnTransformer`** | Keeps preprocessing reproducible and serialisable; no data leakage risk |
| **Feature engineering before validation** | Derived columns are quality-checked by Great Expectations just like raw columns |
| **`SimpleImputer(median)`** | Retains all 7,043 rows instead of dropping 11 with blank `total_charges` |
| **`StandardScaler` for tree models** | Future-proof: works if we ever switch to Logistic Regression, SVM, or neural nets |
| **`OneHotEncoder(drop='first')`** | Avoids collinearity in linear models |
| **MLflow + local artifacts** | MLflow for experiment tracking; local files for FastAPI cold-start without a running MLflow server |
| **SHAP on every run** | Explainability is not optional — stakeholders need to know *why* a customer is flagged |
| **Great Expectations V3 (minimal)** | Data quality without the heavy setup of a full GE project |

---

## Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) - 7,043 customers, 21 features. Target: `Churn` (Yes/No).

---

## Author

Vincent Oei (oei.vincent20@gmail.com)
