"""Streamlit what-if app for the Employee Attrition Predictor.

Tabs
----
Tab 1 — What-If Analysis
    Sidebar inputs (sliders + dropdowns, pre-filled with dataset medians/modes).
    Main area:
      • Risk gauge  (colour-coded arc: green / amber / red)
      • Predicted attrition probability
      • SHAP bar chart of the top-10 factors for this employee
      • Baseline vs What-If comparison table

Tab 2 — Model Overview
    • Global SHAP feature importance (bar chart)
    • CV model comparison table (from reports/model_comparison.csv)
    • Key test-set metrics (from reports/metrics.json)

Run with:
    streamlit run app/app.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import joblib
import shap
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup — ensure project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    BEST_MODEL_FILE,
    EXPLAINER_FILE,
    FIGURES_DIR,
    METRICS_FILE,
    MODEL_COMPARISON_FILE,
    PIPELINE_FILE,
    RAW_DATA_FILE,
)
from src.preprocess import encode_binary_columns
from src.explain import compute_shap_values

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached resource loaders — loaded once per session
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model …")
def load_artefacts() -> tuple:
    """Load the preprocessing pipeline, best model, and SHAP explainer.

    On Streamlit Community Cloud the shap_explainer.joblib may not be present
    (it is gitignored due to size). In that case the explainer is built
    on-the-fly from the best model and a small background sample.

    Returns:
        Tuple of (pipeline, model, explainer).

    Raises:
        FileNotFoundError: If pipeline or model artefacts are missing.
    """
    for path in (PIPELINE_FILE, BEST_MODEL_FILE):
        if not path.exists():
            raise FileNotFoundError(
                f"Required artefact not found: {path}\n"
                "Run `python src/train.py` first."
            )
    pipeline  = joblib.load(PIPELINE_FILE)
    model     = joblib.load(BEST_MODEL_FILE)

    # Try loading the pre-built explainer; build on-the-fly if absent.
    if EXPLAINER_FILE.exists():
        explainer = joblib.load(EXPLAINER_FILE)
    else:
        # Build a lightweight explainer from the model on cloud deployment.
        from src.explain import build_explainer, _unwrap_classifier
        import pandas as pd
        from config import PROCESSED_TRAIN_FILE, TARGET_COLUMN
        clf = _unwrap_classifier(model)
        if PROCESSED_TRAIN_FILE.exists():
            train_df = pd.read_csv(PROCESSED_TRAIN_FILE)
            feat_cols = [c for c in train_df.columns if c != TARGET_COLUMN]
            X_bg = train_df[feat_cols].values[:100]
        else:
            # Last-resort tiny background from the test set
            from config import PROCESSED_TEST_FILE
            test_df = pd.read_csv(PROCESSED_TEST_FILE)
            feat_cols = [c for c in test_df.columns if c != TARGET_COLUMN]
            X_bg = test_df[feat_cols].values[:50]
        explainer = build_explainer(model, X_bg)

    return pipeline, model, explainer


@st.cache_data(show_spinner="Loading reference data …")
def load_reference_data() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Load raw dataset, saved metrics, and model comparison table.

    Returns:
        Tuple of (raw_df, metrics_dict, comparison_df).
    """
    raw = pd.read_csv(RAW_DATA_FILE)
    metrics: dict = {}
    if METRICS_FILE.exists():
        with open(METRICS_FILE, encoding="utf-8") as fh:
            metrics = json.load(fh)
    comparison = pd.DataFrame()
    if MODEL_COMPARISON_FILE.exists():
        comparison = pd.read_csv(MODEL_COMPARISON_FILE, index_col=0)
    return raw, metrics, comparison


@st.cache_data(show_spinner="Computing global SHAP values …")
def load_global_shap() -> tuple[np.ndarray, list[str]]:
    """Load processed test set and return pre-computed SHAP values.

    Returns:
        Tuple of (shap_values, feature_names).
    """
    from config import PROCESSED_TEST_FILE, TARGET_COLUMN
    test_df = pd.read_csv(PROCESSED_TEST_FILE)
    feature_names = [c for c in test_df.columns if c != TARGET_COLUMN]
    X_test = test_df[feature_names].values
    _, _, explainer = load_artefacts()
    sv = compute_shap_values(explainer, X_test)
    return sv, feature_names


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def build_input_row(inputs: dict) -> pd.DataFrame:
    """Assemble sidebar widget values into a single-row raw DataFrame.

    Args:
        inputs: Dict mapping raw feature name → widget value.

    Returns:
        Single-row DataFrame with all 30 input features.
    """
    return pd.DataFrame([inputs])


def predict(pipeline, model, raw_row: pd.DataFrame) -> tuple[float, np.ndarray, list[str]]:
    """Run end-to-end inference on a single raw employee row.

    Args:
        pipeline: Fitted sklearn preprocessing Pipeline.
        model: Fitted classifier (ImbPipeline or bare estimator).
        raw_row: Single-row DataFrame with raw (un-encoded) feature values.

    Returns:
        Tuple of (probability, shap_values_1d, feature_names).
    """
    encoded   = encode_binary_columns(raw_row.copy())
    X_proc    = pipeline.transform(encoded)
    prob      = float(model.predict_proba(X_proc)[0][1])
    feature_names = pipeline.get_feature_names_out().tolist()

    _, _, explainer = load_artefacts()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = compute_shap_values(explainer, X_proc)
    shap_1d = sv[0]   # shape (n_features,)
    return prob, shap_1d, feature_names


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _gauge_figure(prob: float) -> plt.Figure:
    """Draw a semi-circular risk gauge coloured green / amber / red.

    Args:
        prob: Attrition probability in [0, 1].

    Returns:
        Matplotlib Figure with the gauge.
    """
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={"aspect": "equal"})
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.1, 1.2)
    ax.axis("off")

    # Background arc segments: green (0–33%), amber (33–66%), red (66–100%).
    def arc_patch(theta1: float, theta2: float, color: str, alpha: float = 0.25):
        t = np.linspace(np.radians(theta1), np.radians(theta2), 120)
        xs = np.concatenate([[0], np.cos(t) * 1.0, [0]])
        ys = np.concatenate([[0], np.sin(t) * 1.0, [0]])
        ax.fill(xs, ys, color=color, alpha=alpha, zorder=1)

    arc_patch(0,   60,  "#4CAF50")   # green
    arc_patch(60,  120, "#FF9800")   # amber
    arc_patch(120, 180, "#F44336")   # red

    # Needle: maps prob → angle (0=right=0°, 1=left=180°).
    angle = np.radians(180 * (1 - prob))
    ax.annotate(
        "", xy=(np.cos(angle) * 0.85, np.sin(angle) * 0.85),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="#212121", lw=2.5),
        zorder=5,
    )
    ax.scatter([0], [0], s=60, color="#212121", zorder=6)

    # Risk label.
    if prob < 0.33:
        risk_label, risk_color = "LOW RISK",    "#2E7D32"
    elif prob < 0.66:
        risk_label, risk_color = "MEDIUM RISK", "#E65100"
    else:
        risk_label, risk_color = "HIGH RISK",   "#B71C1C"

    ax.text(0, -0.08, f"{prob:.0%}", ha="center", va="top",
            fontsize=18, fontweight="bold", color=risk_color)
    ax.text(0, -0.3, risk_label, ha="center", va="top",
            fontsize=10, fontweight="bold", color=risk_color)
    plt.tight_layout(pad=0)
    return fig


def _shap_bar_figure(
    shap_1d: np.ndarray,
    feature_names: list[str],
    top_n: int = 10,
) -> plt.Figure:
    """Draw a horizontal SHAP bar chart for a single employee.

    Red bars increase attrition risk; blue bars decrease it.

    Args:
        shap_1d: 1-D SHAP array (n_features,) for this employee.
        feature_names: Feature names aligned with shap_1d.
        top_n: Number of top factors to display.

    Returns:
        Matplotlib Figure.
    """
    # Pick top-N by absolute value, sorted for display (smallest at top).
    abs_vals = np.abs(shap_1d)
    top_idx  = np.argsort(abs_vals)[::-1][:top_n]
    top_idx  = top_idx[::-1]   # reverse so largest is at top of chart

    names  = [_clean_feat_name(feature_names[i]) for i in top_idx]
    values = [shap_1d[i] for i in top_idx]
    colors = ["#EF5350" if v > 0 else "#42A5F5" for v in values]

    fig, ax = plt.subplots(figsize=(7, max(3, top_n * 0.38)))
    bars = ax.barh(names, values, color=colors, edgecolor="white", height=0.6)
    ax.axvline(0, color="#424242", linewidth=0.8)
    ax.bar_label(bars, fmt=lambda v: f"{v:+.3f}", padding=3, fontsize=8)
    ax.set_xlabel("SHAP value  (impact on predicted log-odds)", fontsize=9)
    ax.set_title("Top factors for this employee", fontsize=11, fontweight="bold")
    ax.set_xlim(
        min(values) * 1.3 if min(values) < 0 else -0.1,
        max(values) * 1.3 if max(values) > 0 else 0.1,
    )
    # Legend patches.
    ax.legend(
        handles=[
            mpatches.Patch(color="#EF5350", label="↑ increases risk"),
            mpatches.Patch(color="#42A5F5", label="↓ decreases risk"),
        ],
        loc="lower right", fontsize=8,
    )
    plt.tight_layout()
    return fig


def _global_bar_figure(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
) -> plt.Figure:
    """Draw a global mean |SHAP| bar chart across the test set.

    Args:
        shap_values: 2-D SHAP array (n_samples, n_features).
        feature_names: Feature names.
        top_n: Number of features to show.

    Returns:
        Matplotlib Figure.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    pairs = sorted(zip(feature_names, mean_abs), key=lambda x: x[1])
    pairs = pairs[-top_n:]   # keep top N
    names, vals = zip(*pairs)
    names = [_clean_feat_name(n) for n in names]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.38)))
    ax.barh(names, vals, color="#1565C0", edgecolor="white", height=0.6)
    ax.set_xlabel("Mean |SHAP value|  (average impact on model output)", fontsize=9)
    ax.set_title("Global Feature Importance — Mean |SHAP|",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(vals) * 1.15)
    plt.tight_layout()
    return fig


def _clean_feat_name(name: str) -> str:
    """Strip ColumnTransformer prefix (``num__``, ``ord__``, ``nom__``).

    Args:
        name: Prefixed feature name from the ColumnTransformer.

    Returns:
        Human-readable feature name.
    """
    for prefix in ("num__", "ord__", "nom__"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------

SATISFACTION_LABELS = {1: "1 – Low", 2: "2 – Medium", 3: "3 – High", 4: "4 – Very High"}
EDUCATION_LABELS    = {1: "1 – Below College", 2: "2 – College", 3: "3 – Bachelor",
                       4: "4 – Master", 5: "5 – Doctor"}
TRAVEL_OPTS   = ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
DEPT_OPTS     = ["Human Resources", "Research & Development", "Sales"]
EDU_FIELD_OPTS= ["Human Resources", "Life Sciences", "Marketing",
                 "Medical", "Other", "Technical Degree"]
JOB_ROLE_OPTS = ["Healthcare Representative", "Human Resources",
                 "Laboratory Technician", "Manager",
                 "Manufacturing Director", "Research Director",
                 "Research Scientist", "Sales Executive", "Sales Representative"]
MARITAL_OPTS  = ["Divorced", "Married", "Single"]
GENDER_OPTS   = ["Female", "Male"]


def render_sidebar() -> dict:
    """Render all sidebar widgets and return the collected inputs.

    Returns:
        Dict mapping raw feature name → selected value.
    """
    st.sidebar.header("👤 Employee Profile")
    st.sidebar.caption("Adjust values to model different scenarios.")

    with st.sidebar.expander("🔑 Key Risk Factors", expanded=True):
        age             = st.slider("Age",                     18,  60,  36)
        monthly_income  = st.slider("Monthly Income ($)",    1009, 19999, 4919, step=100)
        overtime        = st.selectbox("OverTime",            ["No", "Yes"])
        job_satisfaction= st.select_slider(
            "Job Satisfaction", options=[1, 2, 3, 4],
            value=4, format_func=lambda x: SATISFACTION_LABELS[x])
        wlb             = st.select_slider(
            "Work-Life Balance", options=[1, 2, 3, 4],
            value=3, format_func=lambda x: SATISFACTION_LABELS[x])
        years_company   = st.slider("Years at Company",        0,   40,   5)
        years_promo     = st.slider("Years Since Promotion",   0,   15,   1)
        distance        = st.slider("Distance From Home (km)", 1,   29,   7)

    with st.sidebar.expander("💼 Job Details", expanded=False):
        job_role        = st.selectbox("Job Role",         JOB_ROLE_OPTS,
                                       index=JOB_ROLE_OPTS.index("Research Scientist"))
        department      = st.selectbox("Department",       DEPT_OPTS,
                                       index=DEPT_OPTS.index("Research & Development"))
        job_level       = st.select_slider("Job Level",   options=[1,2,3,4,5], value=1)
        job_involvement = st.select_slider(
            "Job Involvement", options=[1,2,3,4],
            value=3, format_func=lambda x: SATISFACTION_LABELS[x])
        env_sat         = st.select_slider(
            "Environment Satisfaction", options=[1,2,3,4],
            value=3, format_func=lambda x: SATISFACTION_LABELS[x])
        rel_sat         = st.select_slider(
            "Relationship Satisfaction", options=[1,2,3,4],
            value=3, format_func=lambda x: SATISFACTION_LABELS[x])
        perf_rating     = st.selectbox("Performance Rating",
                                       [3, 4],
                                       format_func=lambda x: {3:"3 – Excellent",4:"4 – Outstanding"}[x])

    with st.sidebar.expander("📊 Background & Compensation", expanded=False):
        total_yrs       = st.slider("Total Working Years",     0,   40,  10)
        years_role      = st.slider("Years in Current Role",   0,   18,   3)
        years_mgr       = st.slider("Years With Manager",      0,   17,   3)
        num_companies   = st.slider("Num Companies Worked",    0,    9,   2)
        training        = st.slider("Training Times Last Year",0,    6,   3)
        stock           = st.select_slider("Stock Option Level",options=[0,1,2,3],value=0)
        percent_hike    = st.slider("Percent Salary Hike",    11,   25,  14)
        daily_rate      = st.slider("Daily Rate",            102, 1499, 802)
        hourly_rate     = st.slider("Hourly Rate",            30,  100,  66)
        monthly_rate    = st.slider("Monthly Rate",         2094,26999,14235,step=100)
        num_comp_wkd    = num_companies  # already captured above

    with st.sidebar.expander("👤 Personal", expanded=False):
        gender          = st.selectbox("Gender",          GENDER_OPTS)
        marital_status  = st.selectbox("Marital Status",  MARITAL_OPTS,
                                       index=MARITAL_OPTS.index("Married"))
        education       = st.select_slider(
            "Education Level", options=[1,2,3,4,5],
            value=3, format_func=lambda x: EDUCATION_LABELS[x])
        edu_field       = st.selectbox("Education Field", EDU_FIELD_OPTS,
                                       index=EDU_FIELD_OPTS.index("Life Sciences"))
        business_travel = st.selectbox("Business Travel", TRAVEL_OPTS,
                                       index=TRAVEL_OPTS.index("Travel_Rarely"))

    return {
        "Age":                     age,
        "DailyRate":               daily_rate,
        "DistanceFromHome":        distance,
        "HourlyRate":              hourly_rate,
        "MonthlyIncome":           monthly_income,
        "MonthlyRate":             monthly_rate,
        "NumCompaniesWorked":      num_companies,
        "PercentSalaryHike":       percent_hike,
        "TotalWorkingYears":       total_yrs,
        "TrainingTimesLastYear":   training,
        "YearsAtCompany":          years_company,
        "YearsInCurrentRole":      years_role,
        "YearsSinceLastPromotion": years_promo,
        "YearsWithCurrManager":    years_mgr,
        "BusinessTravel":          business_travel,
        "Education":               education,
        "EnvironmentSatisfaction": env_sat,
        "JobInvolvement":          job_involvement,
        "JobLevel":                job_level,
        "JobSatisfaction":         job_satisfaction,
        "PerformanceRating":       perf_rating,
        "RelationshipSatisfaction":rel_sat,
        "StockOptionLevel":        stock,
        "WorkLifeBalance":         wlb,
        "Department":              department,
        "EducationField":          edu_field,
        "Gender":                  gender,
        "JobRole":                 job_role,
        "MaritalStatus":           marital_status,
        "OverTime":                overtime,
    }


# ---------------------------------------------------------------------------
# Tab 1 — What-If Analysis
# ---------------------------------------------------------------------------

def render_tab_whatif(
    pipeline,
    model,
    inputs: dict,
) -> None:
    """Render the What-If Analysis tab.

    Args:
        pipeline: Fitted preprocessing Pipeline.
        model: Fitted classifier.
        inputs: Raw feature values from the sidebar.
    """
    # ---- Run inference ----
    raw_row = build_input_row(inputs)
    prob, shap_1d, feature_names = predict(pipeline, model, raw_row)

    # ---- Header row: gauge + probability metric ----
    col_gauge, col_prob, col_space = st.columns([2, 2, 3])

    with col_gauge:
        st.subheader("Risk Gauge")
        fig_gauge = _gauge_figure(prob)
        st.pyplot(fig_gauge, use_container_width=False)
        plt.close(fig_gauge)

    with col_prob:
        st.subheader("Prediction")
        st.metric("Attrition Probability", f"{prob:.1%}")

        if prob < 0.33:
            st.success("🟢 Low Risk — employee likely to stay.")
        elif prob < 0.66:
            st.warning("🟡 Medium Risk — worth monitoring.")
        else:
            st.error("🔴 High Risk — consider retention action.")

        # Threshold context
        st.caption(
            "Model threshold (Youden's J) chosen to minimise cost "
            "assuming a missed leaver costs 5× a false alarm."
        )

    st.divider()

    # ---- SHAP bar chart ----
    col_shap, col_baseline = st.columns([3, 2])

    with col_shap:
        st.subheader("Top Factors for This Employee")
        st.caption(
            "Red bars push attrition risk **up**; blue bars push it **down**. "
            "Lengths reflect SHAP value magnitude."
        )
        fig_bar = _shap_bar_figure(shap_1d, feature_names, top_n=10)
        st.pyplot(fig_bar, use_container_width=True)
        plt.close(fig_bar)

    with col_baseline:
        st.subheader("Baseline vs What-If")
        st.caption("Dataset median / mode employee (baseline) vs your profile.")

        # Baseline: dataset-average employee
        baseline_inputs = {
            "Age": 36, "DailyRate": 802, "DistanceFromHome": 7,
            "HourlyRate": 66, "MonthlyIncome": 4919, "MonthlyRate": 14235,
            "NumCompaniesWorked": 2, "PercentSalaryHike": 14,
            "TotalWorkingYears": 10, "TrainingTimesLastYear": 3,
            "YearsAtCompany": 5, "YearsInCurrentRole": 3,
            "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 3,
            "BusinessTravel": "Travel_Rarely", "Education": 3,
            "EnvironmentSatisfaction": 3, "JobInvolvement": 3,
            "JobLevel": 1, "JobSatisfaction": 4, "PerformanceRating": 3,
            "RelationshipSatisfaction": 3, "StockOptionLevel": 0,
            "WorkLifeBalance": 3, "Department": "Research & Development",
            "EducationField": "Life Sciences", "Gender": "Male",
            "JobRole": "Research Scientist", "MaritalStatus": "Married",
            "OverTime": "No",
        }
        baseline_row   = build_input_row(baseline_inputs)
        baseline_prob, _, _ = predict(pipeline, model, baseline_row)

        delta = prob - baseline_prob
        delta_str = f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}"

        comparison_df = pd.DataFrame({
            "Profile":     ["Baseline (median)", "Your What-If"],
            "Probability": [f"{baseline_prob:.1%}", f"{prob:.1%}"],
            "Risk Δ":      ["—", delta_str],
        }).set_index("Profile")
        st.dataframe(comparison_df, use_container_width=True)

        st.markdown("**Key differences from baseline:**")
        changed = {
            k: (baseline_inputs[k], v)
            for k, v in inputs.items()
            if v != baseline_inputs[k]
        }
        if changed:
            rows = []
            for feat, (base_val, new_val) in list(changed.items())[:8]:
                rows.append({"Feature": feat,
                             "Baseline": str(base_val),
                             "Your Value": str(new_val)})
            st.dataframe(pd.DataFrame(rows).set_index("Feature"),
                         use_container_width=True)
        else:
            st.info("No changes from the baseline profile.")

    st.divider()

    # ---- Raw SHAP values expander ----
    with st.expander("📋 Full SHAP values for this employee", expanded=False):
        shap_df = pd.DataFrame({
            "Feature":        [_clean_feat_name(n) for n in feature_names],
            "SHAP Value":     shap_1d.round(4),
            "Direction":      ["↑ risk" if v > 0 else "↓ risk" for v in shap_1d],
        }).sort_values("SHAP Value", key=abs, ascending=False)
        st.dataframe(shap_df, use_container_width=True, height=300)

    with st.expander("📋 Raw input values", expanded=False):
        st.dataframe(pd.DataFrame([inputs]).T.rename(columns={0: "Value"}),
                     use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2 — Model Overview
# ---------------------------------------------------------------------------

def render_tab_overview(metrics: dict, comparison: pd.DataFrame) -> None:
    """Render the Model Overview tab.

    Args:
        metrics: Dict loaded from reports/metrics.json.
        comparison: DataFrame loaded from reports/model_comparison.csv.
    """
    st.subheader("Global Feature Importance — Mean |SHAP| (Test Set)")
    st.caption(
        "Average absolute SHAP value across all 294 test-set employees. "
        "Longer bars = more influential feature overall."
    )
    try:
        sv, feat_names = load_global_shap()
        fig_global = _global_bar_figure(sv, feat_names, top_n=20)
        st.pyplot(fig_global, use_container_width=True)
        plt.close(fig_global)
    except Exception as exc:
        st.warning(f"Could not load global SHAP values: {exc}")

    # Also try to show pre-saved beeswarm if it exists
    beeswarm_path = FIGURES_DIR / "shap_summary.png"
    if beeswarm_path.exists():
        with st.expander("🔍 SHAP Beeswarm (full detail)", expanded=False):
            st.image(str(beeswarm_path), use_container_width=True)

    st.divider()

    col_metrics, col_comparison = st.columns([1, 2])

    with col_metrics:
        st.subheader("Test-Set Metrics")
        if metrics:
            threshold = metrics.get("threshold_cost", 0.5)
            st.caption(
                f"Business cost threshold = **{threshold:.3f}**  "
                f"(FN:FP ratio = {metrics.get('fn_fp_ratio', 5.0):.0f}:1)"
            )
            metric_rows = [
                ("ROC-AUC",  metrics.get("roc_auc",   "—")),
                ("PR-AUC",   metrics.get("pr_auc",    "—")),
                ("F1 Score", metrics.get("f1",        "—")),
                ("Precision",metrics.get("precision", "—")),
                ("Recall",   metrics.get("recall",    "—")),
            ]
            for label, val in metric_rows:
                if isinstance(val, float):
                    st.metric(label, f"{val:.4f}")
                else:
                    st.metric(label, val)

            # Default 0.5 threshold comparison
            if "@default_0.5" in metrics:
                with st.expander("Compare at default threshold (0.5)"):
                    d = metrics["@default_0.5"]
                    st.json(d)
        else:
            st.info("No metrics.json found. Run `python src/evaluate.py` first.")

    with col_comparison:
        st.subheader("5-Fold CV Model Comparison")
        if not comparison.empty:
            st.caption("Trained on SMOTE-resampled training set with imbalance-aware pipelines.")
            # Show just the mean columns, rename for readability.
            display_cols = {
                "ROC_AUC_mean": "ROC-AUC",
                "PR_AUC_mean":  "PR-AUC",
                "F1_mean":      "F1",
                "Precision_mean": "Precision",
                "Recall_mean":  "Recall",
            }
            disp_df = (comparison[list(display_cols.keys())]
                       .rename(columns=display_cols)
                       .sort_values("ROC-AUC", ascending=False))

            # Highlight the best row.
            best_model = disp_df["ROC-AUC"].idxmax()

            def highlight_best(row):
                return ["background-color: #E8F5E9; font-weight: bold"
                        if row.name == best_model
                        else "" for _ in row]

            st.dataframe(
                disp_df.style
                       .apply(highlight_best, axis=1)
                       .format("{:.4f}"),
                use_container_width=True,
            )
            st.caption(f"🏆 Best model by ROC-AUC: **{best_model}**")
        else:
            st.info("No model_comparison.csv found. Run `python src/train.py` first.")

    st.divider()
    st.subheader("Pipeline Architecture")
    st.markdown("""
| Step | Detail |
|---|---|
| Preprocessing | `StandardScaler` (numerical) · `OrdinalEncoder` (ordinal) · `OneHotEncoder` (nominal) |
| Imbalance handling | `class_weight="balanced"` (LR / RF) · SMOTE inside `ImbPipeline` (XGBoost / LightGBM) |
| CV evaluation | Stratified 5-fold, scored on F1, Precision, Recall, ROC-AUC, PR-AUC |
| Tuning | `RandomizedSearchCV` on top-2 models (n_iter=20, scoring=roc_auc) |
| Threshold | Minimises expected cost: 5 × FN + 1 × FP (Youden's J as fallback) |
| Explainability | `shap.TreeExplainer` (XGBoost) / `shap.LinearExplainer` (LR) |
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for the Streamlit app."""
    st.title("📊 Employee Attrition Predictor")
    st.caption(
        "Adjust the employee profile in the sidebar to see live attrition risk, "
        "SHAP explanations, and a baseline comparison.  "
        "Use the second tab for global model insights."
    )

    # Load artefacts — show a clear error if missing.
    try:
        pipeline, model, _ = load_artefacts()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    _, metrics, comparison = load_reference_data()

    # Sidebar — collect inputs.
    inputs = render_sidebar()

    # Tabs.
    tab1, tab2 = st.tabs(["🔍 What-If Analysis", "📈 Model Overview"])

    with tab1:
        render_tab_whatif(pipeline, model, inputs)

    with tab2:
        render_tab_overview(metrics, comparison)


if __name__ == "__main__":
    main()
