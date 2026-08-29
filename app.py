"""
app.py
======
Streamlit front-end for the automated ED triage model.

This is the file to point Streamlit Cloud at (Main file path = app.py).
It ONLY needs two files sitting next to it in the repo:
    - triage_model.joblib
    - triage_common.py

It does NOT need the training dataset (KTAS_data_cleaned.xlsx) and does NOT
retrain anything — it just loads the already-trained model once and serves
predictions from a form.
"""

import joblib
import pandas as pd
import streamlit as st
from triage_common import select_text  # noqa: F401  (required to unpickle the model)

MODEL_PATH = "triage_model.joblib"

URGENCY_LABELS = {
    1: "Resuscitation (immediate)",
    2: "Emergent",
    3: "Urgent",
    4: "Less urgent",
    5: "Non-urgent",
}

URGENCY_COLORS = {
    1: "#8B0000",
    2: "#D9534F",
    3: "#F0AD4E",
    4: "#5BC0DE",
    5: "#5CB85C",
}

SEX_OPTIONS = {1: "Male", 2: "Female"}
ARRIVAL_OPTIONS = {
    1: "Walking in",
    2: "119 Ambulance (public EMS)",
    3: "Private vehicle",
    4: "Private ambulance",
    5: "Wheelchair",
    6: "Carried in",
    7: "Other",
}
INJURY_OPTIONS = {1: "No injury", 2: "Injury"}
MENTAL_OPTIONS = {
    1: "Alert",
    2: "Verbal response",
    3: "Pain response",
    4: "Unresponsive",
}
PAIN_OPTIONS = {0: "No pain reported", 1: "Pain reported"}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def main():
    st.set_page_config(page_title="ED Triage Assistant", page_icon="🏥", layout="centered")
    st.title("🏥 Automated ED Triage Assistant")
    st.caption(
        "Predicts a KTAS triage level (1 = most urgent, 5 = least urgent) from "
        "vitals, basic patient info, and the presenting complaint."
    )

    st.warning(
        "⚠️ Decision-support prototype only — trained on ~1,270 patients. "
        "Not a certified medical device. A qualified clinician must always "
        "make the final triage call.",
        icon="⚠️",
    )

    try:
        bundle = load_model()
    except FileNotFoundError:
        st.error(
            "Could not find `triage_model.joblib` next to this app. "
            "Make sure it's committed to the repo alongside app.py."
        )
        st.stop()

    pipeline = bundle["pipeline"]

    st.subheader("Patient information")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=45)
        sex = st.selectbox("Sex", options=list(SEX_OPTIONS.keys()), format_func=lambda x: SEX_OPTIONS[x])
        arrival_mode = st.selectbox(
            "Arrival mode", options=list(ARRIVAL_OPTIONS.keys()), format_func=lambda x: ARRIVAL_OPTIONS[x]
        )
        injury = st.selectbox("Injury", options=list(INJURY_OPTIONS.keys()), format_func=lambda x: INJURY_OPTIONS[x])
        mental = st.selectbox("Mental status", options=list(MENTAL_OPTIONS.keys()), format_func=lambda x: MENTAL_OPTIONS[x])
        pts_per_hour = st.number_input("Patients arriving per hour (site load)", min_value=0, max_value=50, value=6)

    with col2:
        pain = st.selectbox("Pain reported?", options=list(PAIN_OPTIONS.keys()), format_func=lambda x: PAIN_OPTIONS[x])
        nrs_pain = st.slider("Pain score (NRS, 0-10)", min_value=0, max_value=10, value=0, disabled=(pain == 0))
        sbp = st.number_input("Systolic BP (mmHg)", min_value=0, max_value=300, value=120)
        dbp = st.number_input("Diastolic BP (mmHg)", min_value=0, max_value=200, value=80)
        hr = st.number_input("Heart rate (bpm)", min_value=0, max_value=250, value=80)
        rr = st.number_input("Respiratory rate (/min)", min_value=0, max_value=80, value=18)
        bt = st.number_input("Body temperature (°C)", min_value=25.0, max_value=45.0, value=36.5, step=0.1)
        saturation = st.number_input("O2 Saturation (%)", min_value=0, max_value=100, value=98)

    chief_complain = st.text_area(
        "Chief complaint (free text)",
        placeholder="e.g. sudden chest pain radiating to left arm, shortness of breath",
    )

    group = st.selectbox("Site / Group code", options=[1, 2], index=0)

    if st.button("Predict triage level", type="primary", use_container_width=True):
        patient = {
            "Group": group,
            "Sex": sex,
            "Age": age,
            "Patients number per hour": pts_per_hour,
            "Arrival mode": arrival_mode,
            "Injury": injury,
            "Mental": mental,
            "Pain": pain,
            "NRS_pain": nrs_pain if pain == 1 else None,
            "SBP": sbp,
            "DBP": dbp,
            "HR": hr,
            "RR": rr,
            "BT": bt,
            "Saturation": saturation,
            "Chief_complain": chief_complain,
        }

        cols = bundle["numeric_features"] + bundle["categorical_features"] + [bundle["text_feature"]]
        row = {c: patient.get(c, None) for c in cols}
        df = pd.DataFrame([row])
        df[bundle["text_feature"]] = df[bundle["text_feature"]].fillna("").astype(str)
        for c in bundle["categorical_features"]:
            df[c] = df[c].astype("Int64").astype(str)

        pred = int(pipeline.predict(df)[0])
        proba = pipeline.predict_proba(df)[0]
        classes = pipeline.classes_
        proba_dict = {int(cls): float(p) for cls, p in zip(classes, proba)}

        st.divider()
        st.subheader("Result")

        color = URGENCY_COLORS.get(pred, "#333")
        st.markdown(
            f"<div style='padding:1rem;border-radius:8px;background:{color}22;"
            f"border-left:6px solid {color};'>"
            f"<span style='font-size:1.4rem;font-weight:700;color:{color};'>"
            f"KTAS {pred} — {URGENCY_LABELS[pred]}</span></div>",
            unsafe_allow_html=True,
        )

        st.write("**Class probabilities:**")
        proba_df = pd.DataFrame(
            {
                "KTAS level": [f"{k} ({URGENCY_LABELS[k]})" for k in sorted(proba_dict)],
                "Probability": [proba_dict[k] for k in sorted(proba_dict)],
            }
        ).set_index("KTAS level")
        st.bar_chart(proba_df)

        top_prob = max(proba_dict.values())
        if top_prob < 0.5:
            st.info(
                f"Model confidence is low ({top_prob:.0%} for the top class) — "
                "this case should be reviewed by a human triage nurse without delay."
            )

    with st.expander("Model details"):
        st.write(f"Model type: **{bundle['model_name']}**")
        st.write("Held-out test-set metrics:")
        st.json(bundle["test_metrics"])
        st.caption(
            "under_triage_rate = fraction of test cases where the model predicted "
            "a LESS urgent level than the true expert-assigned level — the "
            "clinically dangerous direction of error."
        )


if __name__ == "__main__":
    main()
