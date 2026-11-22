import streamlit as st
import pandas as pd
import joblib
import os

# ==============================
# 1. PENGATURAN WEBSITE
# ==============================

st.set_page_config(
    page_title="PREDICARE",
    page_icon="🩺",
    layout="wide"
)

# ==============================
# 2. LOAD MODEL AI
# ==============================

MODEL_PATH = "models/predicare_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Model belum ditemukan. Jalankan train_model.py terlebih dahulu.")
    st.stop()

saved_data = joblib.load(MODEL_PATH)
model = saved_data["model"]

# ==============================
# 3. TAMPILAN WEBSITE
# ==============================

st.title("🩺 PREDICARE")
st.caption("Predict Earlier, Care Better.")

st.markdown("""
### Diabetes 30-Day Readmission Prediction
Sistem prototipe untuk memperkirakan risiko pasien diabetes
mengalami rawat inap kembali dalam waktu 30 hari.
""")

st.divider()

# ==============================
# 4. INPUT DATA PASIEN
# ==============================

st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.selectbox(
        "Age Group",
        ["[0-10)", "[10-20)", "[20-30)", "[30-40)",
         "[40-50)", "[50-60)", "[60-70)", "[70-80)",
         "[80-90)", "[90-100)"]
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    inpatient = st.selectbox(
        "Previous Inpatient Visits",
        ["0", "1", "2", "3", "4", "5+"]
    )

    emergency = st.selectbox(
        "Emergency Visits",
        ["0", "1", "2", "3", "4+"]
    )

    time_in_hospital = st.slider(
        "Length of Stay (days)",
        min_value=1,
        max_value=14,
        value=5
    )

    diagnoses = st.slider(
        "Number of Diagnoses",
        min_value=1,
        max_value=9,
        value=3
    )

with col2:
    a1c = st.selectbox(
        "HbA1c Result",
        ["None", "Norm", ">7", ">8"]
    )

    glucose = st.selectbox(
        "Max Glucose Serum",
        ["None", "Norm", ">200", ">300"]
    )

    medications = st.slider(
        "Number of Medications",
        min_value=0,
        max_value=25,
        value=5
    )

    insulin = st.selectbox(
        "Insulin",
        ["No", "Steady", "Up", "Down"]
    )

    medication_change = st.selectbox(
        "Medication Changed",
        ["No", "Ch"]
    )

st.divider()

# ==============================
# 5. PREDIKSI
# ==============================

if st.button("Predict Readmission Risk", use_container_width=True):

    patient_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "inpatient_group": inpatient,
        "emergency_group": emergency,
        "time_in_hospital": time_in_hospital,
        "number_diagnoses": diagnoses,
        "A1Cresult": a1c,
        "max_glu_serum": glucose,
        "num_medications": medications,
        "insulin": insulin,
        "change": medication_change
    }])

    probability = model.predict_proba(patient_data)[0][1]

    percentage = probability * 100

    # Kategori risiko prototype
    if probability < 0.30:
        risk = "LOW"
        color = "success"
        recommendation = "Pemantauan rutin sesuai kondisi pasien."
    elif probability < 0.60:
        risk = "MODERATE"
        color = "warning"
        recommendation = "Pertimbangkan pemantauan lebih lanjut."
    else:
        risk = "HIGH"
        color = "error"
        recommendation = "Prioritaskan evaluasi dan tindak lanjut pasien."

    st.subheader("Prediction Result")

    st.metric(
        label="Readmission Probability",
        value=f"{percentage:.2f}%"
    )

    st.progress(float(probability))

    if risk == "LOW":
        st.success(f"Risk Category: {risk}")
    elif risk == "MODERATE":
        st.warning(f"Risk Category: {risk}")
    else:
        st.error(f"Risk Category: {risk}")

    st.write(recommendation)

    st.caption(
        "Risk thresholds are illustrative prototype values, "
        "not clinically validated cutoffs."
    )

# ==============================
# 6. DISCLAIMER
# ==============================

st.divider()

st.info("""
**Disclaimer:** PREDICARE merupakan prototipe akademik.
Hasil prediksi bukan diagnosis medis, tidak menggantikan
penilaian tenaga kesehatan, dan belum tervalidasi untuk
penggunaan klinis.
""")