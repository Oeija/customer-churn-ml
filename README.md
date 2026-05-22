# Customer Churn ML

End-to-end machine learning pipeline for predicting telecom customer churn.

**Docs:** [`http://3.93.153.38:8000/docs`](http://3.93.153.38:8000/docs)

---

## Project Overview

This project predicts which customers are likely to churn (leave) a telecom service. It includes:

- **Data validation** with Great Expectations
- **Feature engineering** (tenure groups, service counts, derived metrics)
- **Model training** with configurable classifiers (Random Forest, LightGBM, XGBoost)
- **Hyperparameter tuning** with Optuna
- **Explainability** with SHAP (per-user feature importance + business recommendations)
- **Experiment tracking** with MLflow
- **Real-time serving** via FastAPI with Pydantic request/response validation
- **CI/CD** with GitHub Actions (test, train, deploy)
- **Containerisation** with Docker + Docker Compose
- **Production deployment** on AWS EC2

---

## Tech Stack

| Layer | Libraries & Tools |
|-------|-------------------|
| **Core ML / Data** | pandas, NumPy, scikit-learn |
| **Model Libraries** | XGBoost, LightGBM, Random Forest |
| **Hyperparameter Tuning** | Optuna |
| **API / Serving** | FastAPI, Pydantic, Uvicorn |
| **Experiment Tracking** | MLflow |
| **Data Quality** | Great Expectations |
| **Explainability** | SHAP, numba |
| **Serialization** | Joblib |
| **Configuration** | PyYAML |
| **Visualization** | Matplotlib |
| **Storage** | Parquet (processed splits) |
| **DevOps** | Docker, Docker Compose, GitHub Actions |
| **Dev Tools** | uv (dependency management), pytest, ruff, httpx |
| **Cloud** | AWS EC2 |

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
┌─────────────────────────┐
│    Data Validation      │  ← Great Expectations: 43 expectations
│  (Great Expectations)   │     columns, categories, not-null
└────────┬────────────────┘
         │
         ▼
┌────────────────────┐
│ Train / Test Split │  ← stratified, 80/20
└────────┬───────────┘
         │
         ▼
┌──────────────────────────────────┐
│     Data Transformation          │  ← ColumnTransformer
│  (Imputer + Scaler + OneHot)   │     binary passthrough
└────────┬───────────────────────┘
         │
         ▼
┌─────────────────┐
│  Model Training │  ← XGBoost with scale_pos_weight
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  Evaluate + SHAP    │  ← classification report, threshold sweep, SHAP plot
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
┌──────────────────────┐
│ FastAPI Serve        │  ← POST /predict
│ ├─ /health           │  ← POST /predict?explain=true
│ ├─ /predict          │  ← POST /explain
│ └─ /explain          │
└──────────────────────┘
    │
    ▼
┌──────────────────────────────┐
│ SHAP Explainability          │  ← Top churn-driving features per user
│ + Business Recommendations   │  ← Actionable retention advice
└──────────────────────────────┘
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
│   └── processed/                     # Feature-engineered train/test splits as Parquet (ignored by git)
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
│   ├── modeling.ipynb                 # Notebook prototype (refactored)
│   └── shap_explainability.ipynb      # SHAP explainability analysis
├── outputs/                           # Generated artifacts (predictions, SHAP plots, evaluation reports)
├── scripts/
│   ├── train.py                       # End-to-end training pipeline
│   ├── evaluate.py                    # Evaluate saved model
│   ├── predict.py                     # Batch prediction CLI
│   └── save_ge_suite.py               # Save GE suite to JSON
├── src/customer_churn_ml/
│   ├── app/
│   │   ├── main.py                    # FastAPI application
│   │   ├── schemas.py                 # Pydantic request/response models (validated API contracts)
│   │   ├── explain.py                 # SHAP per-user explainability
│   │   └── recommendations.py         # Business recommendation engine
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
├── artifacts/                         # Saved model artifacts — Joblib preprocessor + UBJ model (ignored by git)
├── tests/
│   ├── test_ingestion.py
│   ├── test_preprocess.py
│   ├── test_api.py
│   └── test_recommendations.py        # Recommendation engine tests
├── .github/workflows/
│   ├── ci.yml                         # pytest + ruff on push/PR
│   ├── train.yml                      # Automated training on config changes
│   └── deploy.yml                     # Docker Hub → EC2 deployment
├── Dockerfile                         # Multi-stage container
├── docker-compose.yml                 # EC2 deployment orchestration
├── pyproject.toml                     # Dependencies + tool configs
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

### 5. Serve via FastAPI (local)

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

# Predict with explainability (returns SHAP-based recommendations for churners)
curl -X POST "http://localhost:8000/predict?explain=true" \
  -H "Content-Type: application/json" \
  -d '[{
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "tenure": 1,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 85.0,
    "total_charges": 85.0
  }]'

# Standalone explainability endpoint (returns top features + recommendations)
curl -X POST http://localhost:8000/explain \
  -H "Content-Type: application/json" \
  -d '[{...same customer JSON...}]'
```

### 6. View MLflow experiments

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open http://localhost:5000 to browse runs, metrics, and model artifacts.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info + available endpoints |
| `/health` | GET | Health check + model/explainability status |
| `/predict` | POST | Batch churn prediction |
| `/predict?explain=true` | POST | Batch prediction + SHAP recommendations for churners |
| `/explain` | POST | Standalone SHAP breakdown + recommendations for any customer |
| `/docs` | GET | Interactive Swagger UI (auto-generated) |

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
- **API**: health endpoint, empty list validation, graceful degradation without artifacts, predictions with/without explainability
- **Recommendations**: display name mapping, recommendation rules per feature, fallback behaviour

---

## Docker

### Local development

```bash
# Build
docker build -t churney-api .

# Run (mount model artifacts)
docker run -p 8000:8000 \
  -v $(pwd)/artifacts:/app/artifacts \
  churney-api
```

### Production (Docker Compose on EC2)

```bash
# Uses docker-compose.yml
docker compose up -d
```

---

## CI/CD

Three GitHub Actions workflows are configured:

| Workflow | Trigger | What it does |
|----------|---------|-------------|
| `ci.yml` | push / PR to `main` | Runs `pytest` and `ruff` linting + formatting checks |
| `train.yml` | push to `main` (config/src changes) | Runs full training pipeline, uploads model artifacts |
| `deploy.yml` | manual (`workflow_dispatch`) | Builds Docker image → Docker Hub → deploys to AWS EC2 |

---

## Production Deployment (AWS EC2)

The project includes a complete CI/CD pipeline for deploying to AWS EC2.

### Architecture

```
GitHub Repo
    │
    ▼ (workflow_dispatch — manual trigger)
┌─────────────────────────────┐
│  GitHub Actions: deploy.yml │
│  1. Train model with --tune   │
│  2. Build Docker image        │
│  3. Push to Docker Hub        │
│  4. SCP artifacts/ → EC2      │
│  5. SSH → pull & run          │
└─────────────────────────────┘
    │
    ▼
Docker Hub
    │
    ▼
AWS EC2 (t3.small)
┌─────────────────────────────┐
│  Docker daemon              │
│  ├── Image: churney-api     │
│  └── Volume: ~/artifacts/   │
│      → /app/artifacts        │
└─────────────────────────────┘
    │
    ▼
Port 8000 (public IP)
```

### Prerequisites

1. **Docker Hub** account with an access token
2. **AWS EC2** instance with:
   - Ubuntu OS
   - Docker + Docker Compose installed
   - Security group allowing port 8000
   - SSH key pair
3. **GitHub Secrets** configured:
   - `DOCKERHUB_USERNAME`
   - `DOCKERHUB_TOKEN`
   - `EC2_HOST` (public IP)
   - `EC2_USER` (`ubuntu`)
   - `EC2_SSH_KEY` (full `.pem` contents)

### Deploy

1. Go to **Actions → Deploy to EC2 → Run workflow**
2. Enter a version tag (e.g., `1.0.0`, `1.0.1`)
3. Click **Run workflow**
4. Wait ~3–4 minutes (training + build + deploy)
5. Verify: `curl http://<EC2_PUBLIC_IP>:8000/health`

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
| **Per-user SHAP + recommendations** | Moves beyond summary plots to actionable, customer-specific retention advice |
| **Great Expectations V3 (minimal)** | Data quality without the heavy setup of a full GE project |
| **Docker + Docker Compose** | Reproducible local and production environments |
| **Pydantic schemas** | Type-safe API contracts with automatic validation and Swagger/OpenAPI documentation generation |
| **GitHub Actions + EC2** | Simple, cost-effective production deployment without managed container orchestration |

---

## Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) - 7,043 customers, 21 features. Target: `Churn` (Yes/No).

---

## Author

Vincent Oei (oei.vincent20@gmail.com)
