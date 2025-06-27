# air_quality_streamlit_app.py
"""
Streamlit app to predict Indian Air Quality Category 
(Good, Satisfactory, Moderate, Poor, Very Poor, Severe) from pollutant concentrations.

Run locally with:
    streamlit run air_quality_streamlit_app.py

Dependencies: streamlit scikit-learn pandas numpy
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------------
CSV_PATH = Path("city_day.csv")
FEATURES = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "NH3"]
TARGET = "AQI_Bucket"
CATEGORY_ORDER = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

# -----------------------------------------------------------------------------
# 2. Model loading / training (cached)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_and_train():
    if not CSV_PATH.exists():
        st.error(f"Dataset '{CSV_PATH.name}' not found. Place it in the same folder and restart.")
        st.stop()

    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=[TARGET])
    X = df[FEATURES]
    label_to_int = {lab: idx for idx, lab in enumerate(CATEGORY_ORDER)}
    int_to_label = {idx: lab for lab, idx in label_to_int.items()}
    y = df[TARGET].map(label_to_int)

    num_prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer([("num", num_prep, FEATURES)])

    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    pipe = Pipeline([("prep", preprocessor), ("rf", rf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipe.fit(X_train, y_train)
    val_acc = pipe.score(X_test, y_test)

    return pipe, int_to_label, val_acc

# -----------------------------------------------------------------------------
# 3. Streamlit UI
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Air Quality Category Predictor",
    page_icon="🌫️",
    layout="centered",
)

st.title("🌫️ Air Quality Category Predictor")
st.write(
    "Predict whether the air quality is **Good**, **Satisfactory**, **Moderate**, "
    "**Poor**, **Very Poor**, or **Severe** based on pollutant concentrations."
)

with st.expander("ℹ️ What do the AQI categories mean?"):
    st.markdown(
        """
        | Category      | AQI Range | Description |
        |---------------|-----------|-------------|
        | **Good**          | 0–50      | Minimal impact on health |
        | **Satisfactory**  | 51–100    | Minor discomfort to sensitive people |
        | **Moderate**      | 101–200   | Breathing discomfort for people with lung/heart issues |
        | **Poor**          | 201–300   | Breathing discomfort on prolonged exposure |
        | **Very Poor**     | 301–400   | Respiratory illness on prolonged exposure |
        | **Severe**        | 401–500   | Affects healthy people and seriously impacts those with existing diseases |
        """
    )

with st.spinner("Training / loading model…"):
    model, idx_to_label, val_acc = load_and_train()


st.divider()

# -----------------------------------------------------------------------------
# 4. Input form (manual only)
# -----------------------------------------------------------------------------
st.subheader("🧮 Enter pollutant measurements")
col1, col2, col3 = st.columns(3)
with col1:
    pm25 = st.number_input("PM2.5 (µg/m³) 🟢", 0.0, 1000.0, step=1.0, format="%.2f", help="🟢 Fine particles that penetrate lungs. Safe ≤ 60 µg/m³")
    no2 = st.number_input("NO2 (µg/m³) 🟠", 0.0, 1000.0, step=1.0, format="%.2f", help="🟠 Nitrogen Dioxide – harms lungs. Safe ≤ 80 µg/m³")
    co = st.number_input("CO (µg/m³) 🟡", 0.0, 1000.0, step=1.0, format="%.2f", help="🟡 Carbon Monoxide – reduces oxygen. Safe ≤ 1000 µg/m³")
with col2:
    pm10 = st.number_input("PM10 (µg/m³) 🟢", 0.0, 1000.0, step=1.0, format="%.2f", help="🟢 Dust particles that irritate throat. Safe ≤ 100 µg/m³")
    so2 = st.number_input("SO2 (µg/m³) 🔴", 0.0, 1000.0, step=1.0, format="%.2f", help="🔴 Sulfur Dioxide – affects lungs & eyes. Safe ≤ 80 µg/m³")
    o3 = st.number_input("O3 (µg/m³) 🔵", 0.0, 1000.0, step=1.0, format="%.2f", help="🔵 Ozone – causes chest pain & coughing. Safe ≤ 100 µg/m³")
with col3:
    nh3 = st.number_input("NH3 (µg/m³) 🟣", 0.0, 1000.0, step=1.0, format="%.2f", help="🟣 Ammonia – irritates eyes & throat. Safe ≤ 400 µg/m³")

feature_vector = pd.DataFrame([[pm25, pm10, no2, so2, co, o3, nh3]], columns=FEATURES)


# -----------------------------------------------------------------------------
# 5. Prediction
# -----------------------------------------------------------------------------
if st.button("🔮 Predict Air Quality", type="primary"):
    if feature_vector.isna().any().any():
        st.warning("Please fill in **all** pollutant values before predicting.")
    else:
        pred_int = int(model.predict(feature_vector)[0])
        pred_label = idx_to_label[pred_int]

        st.success(f"### Predicted Air Quality: {pred_label}")

        probas = model.predict_proba(feature_vector)[0]
        proba_df = pd.DataFrame(
            {
                "Category": [idx_to_label[i] for i in range(len(probas))],
                "Probability": probas,
            }
        ).sort_values("Probability", ascending=False)

        st.write("#### Category probabilities")
        st.dataframe(proba_df, use_container_width=True)

        emoji = {
            "Good": "😊",
            "Satisfactory": "🙂",
            "Moderate": "😐",
            "Poor": "😷",
            "Very Poor": "😫",
            "Severe": "🚨",
        }
        st.markdown(f"### {emoji.get(pred_label, '🧐')}  Air quality is **{pred_label}**")
        st.divider()
with st.expander("📘 What do these pollutants mean?"):
    st.markdown(
        """
        | Pollutant | Full Name             | Health Impact |
        |-----------|------------------------|----------------|
        | **PM2.5** | Fine particulate matter | Penetrates deep into lungs, affects breathing |
        | **PM10**  | Dust particles          | Irritates nose and throat |
        | **NO₂**   | Nitrogen dioxide        | Causes lung irritation and worsens asthma |
        | **SO₂**   | Sulfur dioxide          | Affects respiratory system and eyes |
        | **CO**    | Carbon monoxide         | Reduces oxygen in blood, harmful to heart |
        | **O₃**    | Ozone                   | Causes chest pain, coughing, and shortness of breath |
        | **NH₃**   | Ammonia                 | Irritates eyes, nose, throat |
        """
    )

    st.caption("This prediction is a guide. Always rely on official air quality reports for health decisions.")
