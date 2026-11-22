# PREDICARE - MODEL TRAINING

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# 1. BACA DATASET
print("Membaca dataset diabetes...")

df = pd.read_csv("data/diabetic_data.csv")

# 2. TENTUKAN TARGET PREDIKSI
# 1 = readmission dalam 30 hari
# 0 = tidak readmission dalam 30 hari

df = df[df["readmitted"].isin(["<30", ">30", "NO"])].copy()
df["target"] = (df["readmitted"] == "<30").astype(int)

# 3. KELOMPOKKAN JUMLAH KUNJUNGAN
def group_inpatient(x):
    if pd.isna(x):
        return np.nan
    if x >= 5:
        return "5+"
    return str(int(x))

def group_emergency(x):
    if pd.isna(x):
        return np.nan
    if x >= 4:
        return "4+"
    return str(int(x))

df["inpatient_group"] = df["number_inpatient"].apply(group_inpatient)
df["emergency_group"] = df["number_emergency"].apply(group_emergency)

# 4. PILIH FITUR YANG DIPAKAI MODEL
features = [
    "age",
    "gender",
    "inpatient_group",
    "emergency_group",
    "time_in_hospital",
    "number_diagnoses",
    "A1Cresult",
    "max_glu_serum",
    "num_medications",
    "insulin",
    "change"
]

numeric_features = [
    "time_in_hospital",
    "number_diagnoses",
    "num_medications"
]

categorical_features = [
    "age",
    "gender",
    "inpatient_group",
    "emergency_group",
    "A1Cresult",
    "max_glu_serum",
    "insulin",
    "change"
]

X = df[features].copy()
y = df["target"]
groups = df["patient_nbr"]

# Bersihkan nilai '?' agar dianggap data kosong
X = X.replace("?", np.nan)

# 5. SIAPKAN PREPROCESSING
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 6. BUAT MODEL MACHINE LEARNING
model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# 7. BAGI DATA TRAINING DAN TESTING
# Pasien yang sama tidak boleh masuk ke kedua kelompok

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_index, test_index = next(splitter.split(X, y, groups=groups))

X_train = X.iloc[train_index]
X_test = X.iloc[test_index]

y_train = y.iloc[train_index]
y_test = y.iloc[test_index]

print("Jumlah data training:", len(X_train))
print("Jumlah data testing:", len(X_test))

# 8. LATIH MODEL
print("\nSedang melatih model PREDICARE...")

model.fit(X_train, y_train)

# 9. EVALUASI MODEL
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\n===== HASIL EVALUASI PREDICARE =====")
print("Accuracy :", round(accuracy_score(y_test, predictions), 4))
print("Precision:", round(precision_score(y_test, predictions, zero_division=0), 4))
print("Recall   :", round(recall_score(y_test, predictions, zero_division=0), 4))
print("F1 Score:", round(f1_score(y_test, predictions, zero_division=0), 4))
print("ROC-AUC :", round(roc_auc_score(y_test, probabilities), 4))

# 10. SIMPAN MODEL
os.makedirs("models", exist_ok=True)

joblib.dump({
    "model": model,
    "features": features,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features
}, "models/predicare_model.pkl")

print("\nModel berhasil disimpan!")
print("Lokasi: models/predicare_model.pkl")
print("Training PREDICARE selesai!")