import os
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import requests
import json
from dotenv import load_dotenv

from backend.infer import ModelBundle, predict_for_new_patient
from backend.recommend import rank_antibiotics

MODELS_DIR = "models"
DATASET = "microbiology_combined_clean.csv"

st.set_page_config(page_title="Antibiotic Resistance Prediction", layout="wide")


def nice_label(name: str) -> str:
    """Make column names user-friendly for UI labels."""
    s = str(name).replace("_", " ").replace(".", " ").strip()
    # Compact known verbose tokens
    s = s.replace("vs", "vs")
    # Title case for readability
    return s.title()

@st.cache_data(show_spinner=False)
def load_dataset():
    """Load the training dataset for defaults and dropdown options."""
    try:
        return pd.read_csv(DATASET)
    except Exception:
        return pd.read_csv(os.path.join(os.getcwd(), DATASET))

def cat_options(df: pd.DataFrame, col: str, limit: int = 50):
    vals = df[col].dropna().astype(str)
    return list(vals.value_counts().index[:limit])

def default_for_col(df: pd.DataFrame, col: str, is_cat: bool):
    if col not in df.columns:
        return None
    series = df[col].dropna()
    if is_cat:
        return str(series.mode().iloc[0]) if not series.empty else ""
    try:
        return float(series.median()) if not series.empty else 0.0
    except Exception:
        return None

@st.cache_resource
def load_bundle():
    bundle = ModelBundle(MODELS_DIR)
    bundle.load()
    return bundle

bundle = load_bundle()
df = load_dataset()
feature_cols = bundle.feature_columns()
organism_col = bundle.organism_col()
antibiotic_col = bundle.antibiotic_col()
organisms = bundle.metadata.get("organisms", [])
antibiotics = bundle.metadata.get("antibiotics", [])
cat_cols = set(bundle.metadata.get("categorical_feature_cols", []))
order_time_default = str(default_for_col(df, "order_time_jittered_utc", True) or "2024-07-15 09:00:00+00:00")

# Load environment variables from .env if present
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o")
OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "")
OPENROUTER_X_TITLE = os.getenv("OPENROUTER_X_TITLE", "")

st.title("Antibiotic Resistance Prediction System")
st.caption("Powered by CatBoost backend + Streamlit UI")
tabs = st.tabs(["Predict", "EDA"])

with st.sidebar:
    st.header("Prediction Setup")
    easy_mode = st.toggle("Easy Mode", value=True, help="Simplified inputs with sensible defaults")
    # Organism selection
    organism_val = None
    if organism_col:
        organism_choices = organisms or cat_options(df, organism_col)
        organism_val = st.selectbox("Organism", organism_choices)
    # Candidate antibiotic selection
    antibiotic_val = None
    if antibiotic_col:
        antibiotic_choices = antibiotics or cat_options(df, antibiotic_col)
        antibiotic_val = st.selectbox("Antibiotic", antibiotic_choices)

    st.divider()
    st.subheader("Patient Features")

    input_values: Dict = {}
    if easy_mode:
        # Curated, simple inputs with defaults and ranges
        # Hidden IDs and non-clinical fields will use medians/modes
        simple_numeric = {
            "time_to_culturetime": (0, int(df["time_to_culturetime"].quantile(0.95)), int(default_for_col(df, "time_to_culturetime", False) or 0)),
            "medication_time_to_culturetime": (0, int(df["medication_time_to_culturetime"].quantile(0.95)), int(default_for_col(df, "medication_time_to_culturetime", False) or 0)),
            "prior_infecting_organism_days_to_culutre": (0, int(df["prior_infecting_organism_days_to_culutre"].quantile(0.95)), int(default_for_col(df, "prior_infecting_organism_days_to_culutre", False) or 0)),
        }

        input_values["time_to_culturetime"] = st.slider(nice_label("time_to_culturetime"), min_value=simple_numeric["time_to_culturetime"][0], max_value=simple_numeric["time_to_culturetime"][1], value=simple_numeric["time_to_culturetime"][2])
        input_values["medication_time_to_culturetime"] = st.slider(nice_label("medication_time_to_culturetime"), min_value=simple_numeric["medication_time_to_culturetime"][0], max_value=simple_numeric["medication_time_to_culturetime"][1], value=simple_numeric["medication_time_to_culturetime"][2])
        input_values["prior_infecting_organism_days_to_culutre"] = st.slider(nice_label("prior_infecting_organism_days_to_culutre"), min_value=simple_numeric["prior_infecting_organism_days_to_culutre"][0], max_value=simple_numeric["prior_infecting_organism_days_to_culutre"][1], value=simple_numeric["prior_infecting_organism_days_to_culutre"][2])

        input_values["ordering_mode"] = st.selectbox(nice_label("ordering_mode"), cat_options(df, "ordering_mode"))
        input_values["culture_description"] = st.selectbox(nice_label("culture_description"), cat_options(df, "culture_description"))
        input_values["age"] = st.selectbox(nice_label("age"), cat_options(df, "age"))
        input_values["gender"] = st.radio(nice_label("gender"), options=[0, 1], index=1)
        input_values["prior_organism"] = st.selectbox(nice_label("prior_organism"), cat_options(df, "prior_organism"))
        input_values["medication_category"] = st.selectbox(nice_label("medication_category"), cat_options(df, "medication_category"))
        input_values["medication_name"] = st.selectbox(nice_label("medication_name"), cat_options(df, "medication_name"))
        input_values["antibiotic_class"] = st.selectbox(nice_label("antibiotic_class"), cat_options(df, "antibiotic_class"))
        # Auto-fill order_time_jittered_utc to avoid manual entry
        input_values["order_time_jittered_utc"] = order_time_default

        # Set hidden/utility defaults
        input_values["was_positive"] = 1
        if "pat_enc_csn_id_coded" in df.columns:
            input_values["pat_enc_csn_id_coded"] = int(default_for_col(df, "pat_enc_csn_id_coded", False) or 0)
        if "order_proc_id_coded" in df.columns:
            input_values["order_proc_id_coded"] = int(default_for_col(df, "order_proc_id_coded", False) or 0)
    else:
        # Advanced mode: all feature columns with better widgets and defaults
        for col in feature_cols:
            if organism_col and col == organism_col:
                input_values[col] = organism_val
                continue
            if antibiotic_col and col == antibiotic_col:
                input_values[col] = antibiotic_val
                continue
            # Skip manual entry for order_time_jittered_utc and auto-fill
            if col == "order_time_jittered_utc":
                input_values[col] = order_time_default
                continue
            if col in cat_cols:
                opts = cat_options(df, col)
                default = default_for_col(df, col, True)
                if len(opts) > 0:
                    try:
                        default_index = opts.index(default) if default in opts else 0
                    except Exception:
                        default_index = 0
                    input_values[col] = st.selectbox(nice_label(col), opts, index=default_index)
                else:
                    input_values[col] = st.text_input(nice_label(col), value=str(default or ""))
            else:
                default_num = default_for_col(df, col, False)
                input_values[col] = st.number_input(nice_label(col), value=float(default_num or 0.0), step=1.0, format="%f")

    run_btn = st.button("Run Prediction")

if run_btn:
    # Build row and run inference
    row = dict(input_values)
    if organism_col and organism_val is not None:
        row[organism_col] = organism_val
    if antibiotic_col and antibiotic_val is not None:
        row[antibiotic_col] = antibiotic_val

    # Ensure categorical features are strings
    for c in cat_cols:
        if c in row and row[c] is not None:
            row[c] = str(row[c])

    try:
        out = predict_for_new_patient(bundle, row)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # Results display
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Resistance Probability",
            value=f"{out['resistance_probability']:.3f}",
            help="P(resistant | patient, organism, antibiotic)"
        )
    with col2:
        ttr = out.get("predicted_time_to_resistance_days")
        if ttr is None or (isinstance(ttr, float) and np.isnan(ttr)):
            st.metric(label="Predicted Time to Resistance (days)", value="N/A", help="Regressor unavailable or time censored")
        else:
            st.metric(label="Predicted Time to Resistance (days)", value=f"{ttr:.1f}")

    st.divider()
    st.subheader("Top Recommended Alternative Antibiotics")
    try:
        alt_df = rank_antibiotics(bundle, row, organism_val, top_k=10)
        # Exclude the selected antibiotic from alternatives
        if antibiotic_col and antibiotic_val is not None:
            alt_df = alt_df[alt_df["antibiotic"] != antibiotic_val]
        st.dataframe(alt_df, use_container_width=True)
    except Exception as e:
        st.error(f"Ranking failed: {e}")

    # Persist last prediction for AI summary outside of this block
    st.session_state["last_pred"] = {
        "prob": out["resistance_probability"],
        "ttr": ttr,
        "organism": organism_val or "",
        "antibiotic": antibiotic_val or "",
    }
    st.session_state["last_alts"] = alt_df.copy() if isinstance(alt_df, pd.DataFrame) else None

st.info("Tip: Use the sidebar to adjust features and rerun.")

# -----------------------
# AI Summary (always visible)
# -----------------------
st.divider()
st.subheader("AI Summary")
st.caption("Summarize predictions via OpenRouter. Store your key in .env (OPENROUTER_API_KEY) and run a prediction first.")
if OPENROUTER_API_KEY:
    st.success("OpenRouter key detected from .env")
else:
    st.warning("No OpenRouter key found (.env OPENROUTER_API_KEY). Fallback local summary will be used.")

def _summarize(prob: float, ttr_val: float, organism: str, antibiotic: str, alternatives_df: pd.DataFrame) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Missing OpenRouter API key (.env OPENROUTER_API_KEY)")
    threshold = bundle.metadata.get("best_threshold", 0.5)
    top_alts = []
    try:
        top_alts = alternatives_df.head(3).to_dict(orient="records")
    except Exception:
        top_alts = []
    user_prompt = (
        "Summarize the following antibiotic resistance prediction for a clinician. "
        "Use clear, non-technical language and short bullets. Include: risk interpretation, time-to-resistance context, "
        "and top alternative antibiotics with brief rationale. Avoid any patient identifiers.\n\n"
        f"Organism: {organism}\n"
        f"Antibiotic: {antibiotic}\n"
        f"Resistance Probability: {prob:.3f}\n"
        f"Best Threshold (F1-optimized): {threshold}\n"
        f"Predicted Time to Resistance (days): {('N/A' if (ttr_val is None or (isinstance(ttr_val, float) and np.isnan(ttr_val))) else round(ttr_val, 1))}\n"
        f"Top Alternatives: {top_alts}\n"
    )
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful clinical decision support assistant that writes concise, safe summaries."},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.3,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    if OPENROUTER_HTTP_REFERER:
        headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
    if OPENROUTER_X_TITLE:
        headers["X-Title"] = OPENROUTER_X_TITLE

    resp = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        data=json.dumps(payload),
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "")

def _local_summary(prob: float, ttr_val: float, organism: str, antibiotic: str, alternatives_df: pd.DataFrame) -> str:
    """Fallback deterministic summary without external API."""
    risk_level = "low" if prob < 0.2 else ("moderate" if prob < 0.5 else "high")
    ttr_text = "N/A" if (ttr_val is None or (isinstance(ttr_val, float) and np.isnan(ttr_val))) else f"~{ttr_val:.0f} days"
    lines = [
        f"- Organism: {organism}",
        f"- Antibiotic: {antibiotic}",
        f"- Resistance risk: {prob:.3f} ({risk_level})",
        f"- Predicted time to resistance: {ttr_text}",
    ]
    try:
        top = alternatives_df.head(3).to_dict(orient="records")
        if top:
            lines.append("- Top alternatives:")
            for r in top:
                lines.append(f"  • {r['antibiotic']}: prob={r['resistance_probability']:.3f}, time={r['predicted_time_to_resistance_days']:.1f}d")
    except Exception:
        pass
    lines.append("- Note: This is a heuristic summary generated locally without an AI model.")
    return "\n".join(lines)

def _list_models() -> pd.DataFrame:
    """Fetch available models from OpenRouter for validation."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Missing OpenRouter API key (.env OPENROUTER_API_KEY)")
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    }
    if OPENROUTER_HTTP_REFERER:
        headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
    if OPENROUTER_X_TITLE:
        headers["X-Title"] = OPENROUTER_X_TITLE
    r = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    items = data.get("data", [])
    rows = [{"id": m.get("id"), "name": m.get("name"), "pricing": m.get("pricing", {}).get("prompt", ""), "free": (":free" in (m.get("id") or ""))} for m in items]
    return pd.DataFrame(rows)

if st.button("Generate AI Summary"):
    last = st.session_state.get("last_pred")
    alts = st.session_state.get("last_alts")
    if not last or alts is None:
        st.warning("Run a prediction first, then try again.")
    else:
        try:
            summary_text = _summarize(last["prob"], last["ttr"], last["organism"], last["antibiotic"], alts)
            st.write(summary_text)
        except requests.exceptions.HTTPError as e:
            code = getattr(e.response, "status_code", None)
            if code == 402:
                st.warning("OpenRouter returned 402 Payment Required. Ensure your key has billing enabled or switch to a free model via .env OPENROUTER_MODEL (e.g., meta-llama/llama-3.1-8b-instruct:free). Showing local fallback summary below.")
                st.write(_local_summary(last["prob"], last["ttr"], last["organism"], last["antibiotic"], alts))
            else:
                detail = getattr(e.response, "text", "")
                st.error(f"AI summary failed: HTTP {code}. {detail[:300]} Showing local fallback summary.")
                st.write(_local_summary(last["prob"], last["ttr"], last["organism"], last["antibiotic"], alts))
        except Exception as e:
            st.error(f"AI summary failed: {e}. Showing local fallback summary.")
            st.write(_local_summary(last["prob"], last["ttr"], last["organism"], last["antibiotic"], alts))

# Connectivity & model check
if st.button("Check OpenRouter Setup"):
    try:
        models_df = _list_models()
        st.write({"configured_model": OPENROUTER_MODEL})
        st.write("Available models (top 20):")
        st.dataframe(models_df.head(20), use_container_width=True)
        if not models_df.empty and OPENROUTER_MODEL not in set(models_df["id"].astype(str)):
            st.warning("Configured model not found in your accessible list. Set OPENROUTER_MODEL in .env to a valid ID, e.g., meta-llama/llama-3.1-8b-instruct:free or openai/gpt-4o-mini.")
        else:
            st.success("Configured model appears available.")
    except requests.exceptions.HTTPError as e:
        st.error(f"Model listing failed: HTTP {getattr(e.response, 'status_code', None)} - {getattr(e.response, 'text', '')[:300]}")
    except Exception as e:
        st.error(f"Model listing failed: {e}")

# -----------------------
# EDA Tab (visualizations)
# -----------------------
with tabs[1]:
    st.header("Exploratory Data Analysis")
    st.caption("Quick glance at dataset distributions and model metrics")

    # Overview
    st.subheader("Dataset Overview")
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    st.write({"rows": int(df.shape[0]), "columns": int(df.shape[1]), "memory_mb": round(mem_mb, 2)})

    # Target distribution (if present)
    if "susceptibility" in df.columns:
        st.subheader("Susceptibility Distribution")
        st.bar_chart(df["susceptibility"].value_counts())

    # Numeric distributions
    st.subheader("Numeric Distributions")
    num_cols = [
        c for c in [
            "time_to_culturetime",
            "resistant_time_to_culturetime",
            "medication_time_to_culturetime",
            "prior_infecting_organism_days_to_culutre",
        ]
        if c in df.columns
    ]
    cols = st.columns(len(num_cols) or 1)
    for i, c in enumerate(num_cols):
        with cols[i % len(cols)]:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.hist(df[c].dropna(), bins=30, color="#4e79a7")
            ax.set_title(nice_label(c))
            ax.set_xlabel(c)
            ax.set_ylabel("Count")
            st.pyplot(fig, clear_figure=True)

    # Top categories
    st.subheader("Top Categories")
    cat_cols_show = [
        col for col in [
            "medication_category",
            "medication_name",
            "antibiotic_class",
            "ordering_mode",
            "culture_description",
            "age",
            "prior_organism",
        ]
        if col in df.columns
    ]
    for col in cat_cols_show:
        vc = df[col].astype(str).value_counts().head(15).rename("Count").to_frame()
        st.write(f"Top values: {nice_label(col)}")
        st.dataframe(vc, use_container_width=True)

    # Correlation heatmap (numeric)
    st.subheader("Correlation (Numeric Features)")
    numeric_df = df[num_cols].dropna()
    if not numeric_df.empty:
        corr = numeric_df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index)
        fig.colorbar(im, ax=ax)
        st.pyplot(fig, clear_figure=True)

    # Model metrics (if available)
    st.subheader("Model Metrics")
    try:
        metrics_path = os.path.join(MODELS_DIR, "metrics.json")
        metrics = pd.read_json(metrics_path, typ="series").to_dict()
        show_keys = [
            "roc_auc",
            "f1",
            "precision",
            "recall",
            "classifier_best_threshold",
            "rmse",
        ]
        disp = {k: round(metrics[k], 4) if isinstance(metrics.get(k), (float, int)) else metrics.get(k) for k in show_keys if k in metrics}
        st.write(disp)
    except Exception:
        st.info("Metrics file not found. Train the models to generate metrics.")
