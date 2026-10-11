# Project Structure

## Folder layout

```
attrition-predictor/
├── data/
│   ├── raw/            # Original, immutable source data (CSV from IBM HR Analytics or similar)
│   └── processed/      # Cleaned, encoded, scaled data ready for modelling
├── notebooks/          # Exploratory data analysis and experiment notebooks
├── src/                # All reusable Python source modules
│   ├── data.py         # Load raw data, validate schema, basic sanity checks
│   ├── preprocess.py   # Feature engineering, encoding, scaling, train/test split
│   ├── train.py        # Model training, hyperparameter tuning, SMOTE/imbalance handling
│   ├── evaluate.py     # Metrics (AUC-ROC, F1, confusion matrix), threshold tuning
│   └── explain.py      # SHAP explainer setup, summary/waterfall/force plot generators
├── models/             # Serialised model artefacts (.joblib) and preprocessing pipelines
├── app/
│   └── app.py          # Streamlit what-if UI (loads model + pipeline from models/)
├── reports/
│   └── figures/        # Saved plots (PNG/SVG) for the project report
├── tests/              # pytest unit and integration tests mirroring src/ layout
├── config.py           # Central config: all paths, hyperparameter defaults, random_state
├── requirements.txt    # Pinned dependencies
└── README.md           # Setup, usage, and project summary
```

## Key conventions
- `data/raw/` is read-only — never write to it programmatically.
- Every `src/` module is importable standalone; side-effects only happen inside `if __name__ == "__main__"` guards.
- `models/` stores one artefact per experiment run, named with a timestamp or version tag.
- Notebooks are for exploration only; production logic lives in `src/`.
