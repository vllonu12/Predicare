
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Membaca dataset asli
print("Membaca dataset...")
df = pd.read_csv("data/diabetic_data.csv")

print("Data awal:", df.shape)

# Mempertahankan data dengan kategori readmission yang valid
df = df[df["readmitted"].isin(["<30", ">30", "NO"])].copy()

# Mempertahankan hanya kolom yang dibutuhkan model
selected_columns = [
    "patient_nbr",
    "readmitted",
    "age",
    "gender",
    "number_inpatient",
    "number_emergency",
    "time_in_hospital",
    "number_diagnoses",
    "A1Cresult",
    "max_glu_serum",
    "num_medications",
    "insulin",
    "change"
]

df = df[selected_columns]

# Mengambil 20.000 baris secara stratified
df_small, _ = train_test_split(
    df,
    train_size=100,
    stratify=df["readmitted"],
    random_state=42
)

# Mengacak urutan data
df_small = df_small.sample(frac=1, random_state=42)

# Membuat folder data jika belum ada
os.makedirs("data", exist_ok=True)

# Menyimpan dataset hasil pemangkasan
df_small.to_csv(
    "data/diabetic_data_small.csv",
    index=False
)

print("\nPemangkasan selesai!")
print("Data awal:", len(pd.read_csv("data/diabetic_data.csv")))
print("Data baru:", len(df_small))
print("Jumlah kolom:", len(df_small.columns))
print("Distribusi target:")
print(df_small["readmitted"].value_counts())
print("\nFile tersimpan: data/diabetic_data_small.csv")
