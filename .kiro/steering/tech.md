# Technology Stack

## Language & runtime
- **Python 3.11** — minimum version; use f-strings, `match` statements, and `tomllib` where helpful.

## Core libraries (pinned in requirements.txt)
| Purpose | Library |
|---|---|
| Data wrangling | `pandas`, `numpy` |
| ML pipeline | `scikit-learn`, `imbalanced-learn` |
| Boosting models | `xgboost`, `lightgbm` |
| Explainability | `shap` |
| Visualisation | `matplotlib`, `seaborn` |
| Model persistence | `joblib` |
| App / UI | `streamlit` |
| Notebooks | `jupyter` |
| Testing | `pytest` |

## Environment management
- Always use a **virtual environment** (`.venv/` at the project root, created with `python -m venv .venv`).
- `requirements.txt` must have **pinned versions** (e.g. `pandas==2.2.2`), not open ranges.
- Never install packages globally for this project.
- Activate before running anything: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Unix).

## Dependency hygiene
- After adding a package run `pip freeze > requirements.txt` to keep the file current.
- Do not add packages that duplicate existing ones (e.g. don't add `catboost` if `xgboost`/`lightgbm` already cover the requirement).
