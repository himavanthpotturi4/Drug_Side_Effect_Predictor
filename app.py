import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "src"))

import random
import streamlit as st

from drug_predictor.architecture.architecture_inference import ArchitecturePredictor
from drug_predictor.architecture.module_8_user_interface import APP_TITLE


def simplify_use_label(label: str) -> str:
    t = (label or "").strip().lower()
    replacements = [
        ("analgesic", "Pain relief"),
        ("antipyretic", "Fever reduction"),
        ("anti-parkinson", "Parkinson's symptom control"),
        ("antithrombotic", "Blood clot prevention"),
        ("antiinflammatory", "Inflammation reduction"),
        ("anti-inflammatory", "Inflammation reduction"),
        ("antibacterial", "Bacterial infection treatment"),
        ("antiviral", "Viral infection treatment"),
        ("antifungal", "Fungal infection treatment"),
        ("antihypertensive", "Blood pressure control"),
        ("antidiabetic", "Blood sugar control"),
        ("gastro", "Stomach/acid disorder treatment"),
    ]
    for key, plain in replacements:
        if key in t:
            return plain

    # Make raw condition labels readable for non-technical users.
    # Example: "Aortic Aneurysm, Abdominal" -> "Treatment of abdominal aortic aneurysm"
    txt = (label or "").strip()
    if "," in txt:
        parts = [p.strip() for p in txt.split(",") if p.strip()]
        if len(parts) == 2:
            txt = f"{parts[1]} {parts[0]}"

    txt = txt.replace("_", " ").strip().lower()
    txt = " ".join(txt.split())

    friendly = {
        "asthma": "Asthma management",
        "muscular dystrophy duchenne": "Duchenne muscular dystrophy management",
    }
    if txt in friendly:
        return friendly[txt]

    return f"Treatment of {txt}"


@st.cache_resource(show_spinner="Loading prediction model...")
def get_predictor() -> ArchitecturePredictor:
    return ArchitecturePredictor()

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.8rem; max-width: 1100px;}
    .hero {
        background: linear-gradient(120deg, #eaf4ff 0%, #f7fbff 55%, #eefaf4 100%);
        border: 1px solid #d7e6f7;
        border-radius: 16px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }
    .hero h1 {margin: 0; color: #0c2d48; font-size: 2.2rem;}
    .hero p {margin: 6px 0 0 0; color: #335a75;}
    .result-card {
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 14px;
        background: #ffffff;
        color: #0f172a;
    }
    .result-card b {color: #0b3a6d;}
    .use-chip {
        display: inline-block;
        background: #0b2239;
        color: #dceeff;
        border: 1px solid #1e4367;
        padding: 8px 12px;
        border-radius: 999px;
        margin: 4px 6px 4px 0;
        font-size: 0.95rem;
    }
    .side-chip {
        display: inline-block;
        background: #ffffff;
        color: #0b3a6d;
        border: 1px solid #d7e6f7;
        padding: 8px 12px;
        border-radius: 999px;
        margin: 4px 6px 4px 0;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero">
      <h1>Drug Side-Effect Prediction System</h1>
      <p>Enter a drug and view likely side effects plus one primary biological target type.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
try:
    predictor = get_predictor()
except Exception as exc:  # noqa: BLE001
    st.error(
        "Architecture model not ready. Run:\n"
        "`python scripts/run_architecture_pipeline.py --max-drugs 300 --epochs 50`"
    )
    st.stop()

if "drug_input" not in st.session_state:
    st.session_state["drug_input"] = ""
if "pending_random_drug" not in st.session_state:
    st.session_state["pending_random_drug"] = None
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

if st.button("Random Drug"):
    sample_names = predictor.drug_index["drug_name"].dropna().tolist()
    if sample_names:
        st.session_state["pending_random_drug"] = random.choice(sample_names)
        st.rerun()

if st.session_state["pending_random_drug"] is not None:
    st.session_state["drug_input"] = st.session_state["pending_random_drug"]
    st.session_state["pending_random_drug"] = None

with st.form("prediction_form", clear_on_submit=False):
    col1, col2 = st.columns([2, 1])
    with col1:
        drug_name = st.text_input(
            "Enter drug name",
            key="drug_input",
            placeholder="e.g., aspirin",
        )
    with col2:
        top_k = st.slider("Top side effects", min_value=5, max_value=25, value=12, step=1)
    predict_clicked = st.form_submit_button("Predict", type="primary")

if predict_clicked:
    if not drug_name.strip():
        st.warning("Enter a valid drug name.")
    else:
        result = predictor.predict(drug_name, top_k=top_k)
        st.session_state["last_result"] = result

result = st.session_state.get("last_result")
if result:
    if not result["found"]:
        st.error(result["message"])
    else:
        st.success(f"Matched as: {result['resolved_drug']}")

        st.subheader("Predicted Side Effects")
        side_effect_names = []
        for item in result.get("predicted_side_effects", []):
            label = str(item.get("side_effect", "")).replace("_", " ").title().strip()
            if label:
                side_effect_names.append(label)
        if side_effect_names:
            side_chips = "".join([f'<span class="side-chip">{name}</span>' for name in side_effect_names])
            st.markdown(side_chips, unsafe_allow_html=True)
        else:
            st.info("No side-effect prediction available.")

        action = result["predicted_action"]
        target_label = action["class"].replace("_", " ").title()
        st.subheader("Primary Target Type")
        st.markdown(
            f"""
            <div class="result-card">
              <b>{target_label}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Common Uses")
        treated_for = result.get("treated_for", [])
        if treated_for:
            pretty_uses = []
            seen = set()
            for u in treated_for[:3]:
                label = simplify_use_label(u)
                key = label.strip().lower()
                if key in seen:
                    continue
                seen.add(key)
                pretty_uses.append(label)
            chips = "".join([f'<span class="use-chip">{u}</span>' for u in pretty_uses])
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.info("No treatment indication data found for this drug in the current dataset.")
