"""Streamlit application for Mental Health Treatment Prediction."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import streamlit as st
import pandas as pd
import numpy as np
from src.predict import MentalHealthPredictor
from src.utils import load_model, get_model_path

st.set_page_config(
    page_title="Mental Health Treatment Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .risk-high {
        background-color: #ffe0e0;
        border-left: 5px solid #ff4444;
        padding: 1rem;
        border-radius: 8px;
    }
    .risk-low {
        background-color: #e0ffe0;
        border-left: 5px solid #44ff44;
        padding: 1rem;
        border-radius: 8px;
    }
    .stMetric > div {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Load the trained model predictor."""
    try:
        return MentalHealthPredictor()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def main():
    """Main Streamlit application."""
    st.markdown('<p class="main-header">🧠 Mental Health Treatment Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict whether a person is likely to seek mental health treatment</p>', unsafe_allow_html=True)

    predictor = load_predictor()
    if predictor is None:
        st.stop()

    st.sidebar.header("ℹ️ About This App")
    st.sidebar.info(
        "This app predicts whether a person in the tech industry "
        "is likely to seek mental health treatment based on various "
        "workplace and personal factors."
    )
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Model Info")
    st.sidebar.write("Algorithm: Best performing model")
    st.sidebar.write("Metric: F1 Score")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("👤 Personal Information")

        age = st.number_input("Age", min_value=18, max_value=75, value=30, step=1)

        gender = st.selectbox("Gender", ["Male", "Female", "Other"])

        self_employed = st.selectbox("Self Employed", ["Yes", "No"])

        family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])

    with col2:
        st.subheader("🏢 Work Environment")

        work_interfere = st.selectbox(
            "Work Interference",
            ["Never", "Rarely", "Sometimes", "Often"],
        )

        no_employees = st.selectbox(
            "Company Size",
            ["1-5", "6-25", "26-100", "101-500", "501-1000", "More than 1000"],
        )

        remote_work = st.selectbox("Remote Work", ["Yes", "No"])

        tech_company = st.selectbox("Tech Company", ["Yes", "No"])

    col3, col4 = st.columns([1, 1])

    with col3:
        st.subheader("💼 Benefits & Support")

        benefits = st.selectbox(
            "Mental Health Benefits",
            ["Yes", "No", "Don't Know"],
        )

        care_options = st.selectbox(
            "Care Options Available",
            ["Yes", "No", "Not Sure"],
        )

        wellness_program = st.selectbox(
            "Wellness Program",
            ["Yes", "No", "Don't Know"],
        )

        seek_help = st.selectbox(
            "Employer Encourages Seeking Help",
            ["Yes", "No", "Don't Know"],
        )

    with col4:
        st.subheader("🔒 Privacy & Culture")

        anonymity = st.selectbox(
            "Anonymity Protected",
            ["Yes", "No", "Don't Know"],
        )

        leave = st.selectbox(
            "Ease of Medical Leave",
            [
                "Very Easy",
                "Somewhat Easy",
                "Don't Know",
                "Somewhat Difficult",
                "Very Difficult",
            ],
        )

        coworkers = st.selectbox(
            "Discuss with Coworkers",
            ["Yes", "No", "Some of them"],
        )

        supervisor = st.selectbox(
            "Discuss with Supervisor",
            ["Yes", "No", "Some of them"],
        )

    col5, col6 = st.columns([1, 1])

    with col5:
        mental_health_consequence = st.selectbox(
            "Mental Health Consequence",
            ["Yes", "No", "Maybe"],
        )

        phys_health_consequence = st.selectbox(
            "Physical Health Consequence",
            ["Yes", "No", "Maybe"],
        )

        mental_health_interview = st.selectbox(
            "Discuss Mental Health in Interview",
            ["Yes", "No", "Maybe"],
        )

    with col6:
        phys_health_interview = st.selectbox(
            "Discuss Physical Health in Interview",
            ["Yes", "No", "Maybe"],
        )

        mental_vs_physical = st.selectbox(
            "Mental vs Physical Health Importance",
            ["Yes", "No", "Don't Know"],
        )

        obs_consequence = st.selectbox(
            "Observed Consequences of Mental Health Issues",
            ["Yes", "No"],
        )

    st.markdown("---")

    if st.button("🔮 Predict Treatment Seeking", type="primary", use_container_width=True):
        input_data = _prepare_input(
            age, gender, self_employed, family_history, work_interfere,
            no_employees, remote_work, tech_company, benefits, care_options,
            wellness_program, seek_help, anonymity, leave, coworkers,
            supervisor, mental_health_consequence, phys_health_consequence,
            mental_health_interview, phys_health_interview,
            mental_vs_physical, obs_consequence,
        )

        result = predictor.predict(input_data)

        st.markdown("---")
        st.subheader("📋 Prediction Results")

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.metric("Prediction", result["risk_level"])

        with col_b:
            st.metric("Treatment Probability", f"{result['probability_treatment']:.1%}")

        with col_c:
            st.metric("No Treatment Probability", f"{result['probability_no_treatment']:.1%}")

        st.markdown("---")

        if result["prediction"] == 1:
            st.markdown(
                f'<div class="risk-high">'
                f"<h3>⚠️ High Risk</h3>"
                f"<p>{result['message']}</p>"
                f"<p>The model predicts this person is <strong>likely</strong> "
                f"to seek mental health treatment.</p>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="risk-low">'
                f"<h3>✅ Low Risk</h3>"
                f"<p>{result['message']}</p>"
                f"<p>The model predicts this person is <strong>unlikely</strong> "
                f"to seek mental health treatment.</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.subheader("📈 Probability Distribution")
        prob_df = pd.DataFrame({
            "Category": ["No Treatment", "Treatment"],
            "Probability": [
                result["probability_no_treatment"],
                result["probability_treatment"],
            ],
        })
        st.bar_chart(prob_df.set_index("Category"))


def _prepare_input(
    age, gender, self_employed, family_history, work_interfere,
    no_employees, remote_work, tech_company, benefits, care_options,
    wellness_program, seek_help, anonymity, leave, coworkers,
    supervisor, mental_health_consequence, phys_health_consequence,
    mental_health_interview, phys_health_interview,
    mental_vs_physical, obs_consequence,
):
    """Prepare input data for prediction."""
    mappings = {
        "gender": {"Male": 1, "Female": 0, "Other": 2},
        "yes_no": {"Yes": 1, "No": 0},
        "self_employed": {"Yes": 1, "No": 0},
        "work_interfere": {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3},
        "company_size": {
            "1-5": 0, "6-25": 1, "26-100": 2,
            "101-500": 3, "501-1000": 4, "More than 1000": 5,
        },
        "yes_no_dk": {"Yes": 2, "No": 0, "Don't Know": 1, "Not Sure": 1},
        "leave": {
            "Very Easy": 4, "Somewhat Easy": 3, "Don't Know": 2,
            "Somewhat Difficult": 1, "Very Difficult": 0,
        },
        "coworkers": {"Yes": 2, "No": 0, "Some of them": 1},
        "supervisor": {"Yes": 2, "No": 0, "Some of them": 1},
        "consequence": {"Yes": 2, "No": 0, "Maybe": 1},
        "mental_vs_physical": {"Yes": 1, "No": 0, "Don't Know": 1},
    }

    input_dict = {
        "Age": age,
        "Gender": mappings["gender"].get(gender, 1),
        "self_employed": mappings["self_employed"].get(self_employed, 0),
        "family_history": mappings["yes_no"].get(family_history, 0),
        "work_interfere": mappings["work_interfere"].get(work_interfere, 2),
        "no_employees": mappings["company_size"].get(no_employees, 2),
        "remote_work": mappings["yes_no"].get(remote_work, 0),
        "tech_company": mappings["yes_no"].get(tech_company, 1),
        "benefits": mappings["yes_no_dk"].get(benefits, 1),
        "care_options": mappings["yes_no_dk"].get(care_options, 1),
        "wellness_program": mappings["yes_no_dk"].get(wellness_program, 1),
        "seek_help": mappings["yes_no_dk"].get(seek_help, 1),
        "anonymity": mappings["yes_no_dk"].get(anonymity, 1),
        "leave": mappings["leave"].get(leave, 2),
        "mental_health_consequence": mappings["consequence"].get(
            mental_health_consequence, 1
        ),
        "phys_health_consequence": mappings["consequence"].get(
            phys_health_consequence, 1
        ),
        "coworkers": mappings["coworkers"].get(coworkers, 1),
        "supervisor": mappings["supervisor"].get(supervisor, 1),
        "mental_health_interview": mappings["consequence"].get(
            mental_health_interview, 1
        ),
        "phys_health_interview": mappings["consequence"].get(
            phys_health_interview, 1
        ),
        "mental_vs_physical": mappings["mental_vs_physical"].get(
            mental_vs_physical, 1
        ),
        "obs_consequence": mappings["yes_no"].get(obs_consequence, 0),
    }

    return pd.DataFrame([input_dict])


if __name__ == "__main__":
    main()
