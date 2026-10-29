
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Clinical Ensemble Model",
    page_icon="🩺",
    layout="wide",
)

st.title("Clinical Ensemble Model Dashboard")
st.warning(
    "Educational project only. These experimental predictions "
    "must not be used for medical diagnosis or treatment."
)

MODEL_PATH = Path("models/xgboost_pipeline.joblib")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error(
    "XGBoost model file not found. "
    "Check that models/xgboost_pipeline.joblib exists."
    )
    st.stop()

model = load_model()

page = st.sidebar.radio(
    "Navigate",
    [
        "Overview",
        "Feature Importance",
        "SHAP Explainability",
        "Experimental Prediction",
    ],
)

if page == "Overview":
    st.header("Project Overview")
    st.write(
        "Comparing Logistic Regression, Random Forest, XGBoost, "
        "and Soft Voting for an experimental binary classification task."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Selected model", "XGBoost")
    col2.metric("Test ROC-AUC", "0.8066")
    col3.metric("Test F1-score", "0.7281")

    results_path = Path("cross_validation_results.csv")
    if results_path.exists():
        st.subheader("Model comparison")
        results = pd.read_csv(results_path)
        st.dataframe(results, use_container_width=True)

elif page == "Feature Importance":
    st.header("Feature Importance")

    csv_path = Path("reports/feature_importance.csv")
    if csv_path.exists():
        importance = pd.read_csv(csv_path)
        top = importance.head(15).sort_values("importance")
        st.bar_chart(top.set_index("feature")["importance"])
        st.dataframe(importance.head(15), use_container_width=True)
    else:
        st.info("Run src/feature_importance.py first.")
elif page == "SHAP Explainability":
    st.header("SHAP Explainability")

    st.write(
        "SHAP estimates how individual features influence the "
        "saved XGBoost model's predictions. These explanations "
        "describe model behavior, not disease causation."
    )

    shap_plot = Path("reports/shap_summary.png")
    shap_csv = Path("reports/shap_global_importance.csv")

    if shap_plot.exists():
        st.subheader("SHAP Summary Plot")
        st.image(str(shap_plot), use_container_width=True)
    else:
        st.info("Run src/shap_analysis.py to generate the SHAP plot.")

    if shap_csv.exists():
        st.subheader("Global SHAP Importance")
        shap_results = pd.read_csv(shap_csv)
        st.dataframe(shap_results.head(15), use_container_width=True)
    else:
        st.info("The SHAP importance table has not been generated.")
elif page == "Experimental Prediction":
    st.header("Experimental Prediction")
    st.caption(
        "Feature names are anonymized. Verify their meanings and "
        "valid ranges before interpreting model outputs."
    )

    with st.form("prediction_form"):
        st.subheader("Numeric features")
        f0 = st.number_input("f_0", value=50.0)
        f1 = st.number_input("f_1", value=160.0)
        f2 = st.number_input("f_2", value=70.0)
        f3 = st.number_input("f_3", value=120.0)
        f4 = st.number_input("f_4", value=80.0)

        st.subheader("Categorical features")
        f5 = st.selectbox("f_5", ["1", "2"])
        f6 = st.selectbox("f_6", ["1", "2", "3"])
        f7 = st.selectbox("f_7", ["1", "2", "3"])
        f8 = st.selectbox("f_8", ["0", "1"])
        f9 = st.selectbox("f_9", ["0", "1"])
        f10 = st.selectbox("f_10", ["0", "1"])

        submitted = st.form_submit_button("Run prediction")

    if submitted:
        inputs = pd.DataFrame([{
            "f_0": f0, "f_1": f1, "f_2": f2,
            "f_3": f3, "f_4": f4, "f_5": f5,
            "f_6": f6, "f_7": f7, "f_8": f8,
            "f_9": f9, "f_10": f10,
        }])

        prediction = model.predict(inputs)[0]
        probability = model.predict_proba(inputs)[0, 1]

        st.metric("Predicted dataset class", str(prediction))
        st.metric("Model score for class 1", f"{probability:.1%}")
        st.caption(
            "This is an experimental dataset-class prediction, "
            "not a confirmed disease diagnosis."
        )
