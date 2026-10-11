# Coding Standards

## Style
- Follow **PEP 8** strictly: 4-space indentation, max line length 88 characters (Black-compatible), two blank lines between top-level definitions.
- Use **type hints** on every function signature — parameters and return type.
- Write a **docstring on every function and class** using Google style:
  ```python
  def train_model(X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
      """Train an XGBoost classifier on the provided features and labels.

      Args:
          X: Feature matrix after preprocessing.
          y: Binary target vector (1 = attrition, 0 = retained).

      Returns:
          Fitted XGBClassifier instance.
      """
  ```

## Paths & configuration
- **No hard-coded paths anywhere in src/, app/, or tests/.**
- All file paths, directory names, and tunable constants live in `config.py` at the project root.
- Import config in modules: `from config import DATA_RAW_DIR, MODELS_DIR, RANDOM_STATE`.

## Reproducibility
- Pass `random_state=42` (sourced from `config.RANDOM_STATE`) to every stochastic operation: train/test split, model constructors, SMOTE, cross-validation.

## Function design
- Keep functions **small and single-purpose** — if a function does more than one logical thing, split it.
- Prefer pure functions (same input → same output, no hidden state) wherever possible.
- Add inline comments that explain **why**, not what:
  ```python
  # Use SMOTE on training data only to prevent data leakage into the test set
  X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
  ```

## Error handling
- Validate inputs at module boundaries (e.g., check expected columns exist before preprocessing).
- Raise descriptive `ValueError` or `FileNotFoundError` with context, never bare `except` clauses.

## Testing
- Every function in `src/` should have a corresponding test in `tests/`.
- Use `pytest` fixtures for shared data setup.
- Tests must not write to `data/raw/` or `models/` — use `tmp_path` fixtures for file I/O.

## Notebooks
- Notebooks in `notebooks/` are for exploration only; do not import from notebooks in `src/`.
- Clear all outputs before committing a notebook to version control.
