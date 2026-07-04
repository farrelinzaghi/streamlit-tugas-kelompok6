import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ==========================
# Konfigurasi Halaman
# ==========================
st.set_page_config(
    page_title="Analisis Persebaran Sekolah Indonesia",
    page_icon="🏫",
    layout="wide",
)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)

st.title("🏫 Analisis Hubungan Penduduk Usia Sekolah dengan Persebaran Sekolah di Indonesia")
st.markdown("---")

# ====================================================
# TAHAP 1
# ====================================================
st.header("1. Problem Definition")

st.markdown("""
### Rumusan Masalah
1. Bagaimana karakteristik persebaran penduduk usia sekolah, luas wilayah, dan jumlah sekolah di Indonesia?
2. Apakah terdapat hubungan linear antara jumlah penduduk usia sekolah dan luas wilayah terhadap jumlah sekolah?
3. Faktor mana yang memberikan kontribusi paling besar terhadap jumlah sekolah?

### Hipotesis
- **H0:** Tidak terdapat pengaruh signifikan.
- **H1:** Terdapat pengaruh positif dan signifikan.
""")

# ====================================================
# TAHAP 2
# ====================================================
st.header("2. Data Collection")

np.random.seed(42)

n_prov = 38

total_pop = np.random.normal(3000000, 1000000, n_prov)
school_age = total_pop * np.random.uniform(0.18, 0.22, n_prov)
area = np.random.normal(50000, 15000, n_prov)

schools = (
    school_age * 0.003
    + area * 0.01
    + np.random.normal(0, 100, n_prov)
)

df = pd.DataFrame({
    "Provinsi": [f"Provinsi {i+1}" for i in range(n_prov)],
    "Penduduk Usia Sekolah": school_age.astype(int),
    "Total Penduduk": total_pop.astype(int),
    "Luas Wilayah": area.astype(int),
    "Jumlah Sekolah": schools.astype(int),
})

st.subheader("Dataset")
st.dataframe(df, use_container_width=True)

# ====================================================
# TAHAP 3
# ====================================================
st.header("3. Data Cleaning")

st.write("### Missing Value")
st.dataframe(df.isnull().sum().to_frame("Jumlah"))

st.write("### Data Duplikat")
st.write(df.duplicated().sum())

kolom_numerik = [
    "Penduduk Usia Sekolah",
    "Total Penduduk",
    "Luas Wilayah",
    "Jumlah Sekolah",
]

for col in kolom_numerik:
    df = df[df[col] >= 0]

st.success(f"Jumlah data setelah cleaning : {len(df)}")

# ====================================================
# TAHAP 4
# ====================================================
st.header("4. Exploratory Data Analysis")

st.subheader("Statistik Deskriptif")

deskriptif = (
    df[kolom_numerik]
    .describe()
    .T[["mean", "std", "min", "50%", "max"]]
)

deskriptif.columns = [
    "Mean",
    "Std",
    "Minimum",
    "Median",
    "Maximum",
]

st.dataframe(deskriptif)

st.subheader("Heatmap Korelasi")

corr = df[kolom_numerik].corr()

fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(
    corr,
    annot=True,
    cmap="Blues",
    fmt=".2f",
    ax=ax,
)

st.pyplot(fig)

st.info("""
Penduduk usia sekolah memiliki korelasi positif yang kuat dengan jumlah sekolah.
""")

# ====================================================
# TAHAP 5
# ====================================================
st.header("5. Statistical Modeling")

X_check = df[
    [
        "Penduduk Usia Sekolah",
        "Total Penduduk",
        "Luas Wilayah",
    ]
]

X_check = sm.add_constant(X_check)

vif = pd.DataFrame()

vif["Variabel"] = X_check.columns

vif["VIF"] = [
    variance_inflation_factor(X_check.values, i)
    for i in range(X_check.shape[1])
]

st.subheader("Uji VIF")

st.dataframe(vif)

X = df[
    [
        "Penduduk Usia Sekolah",
        "Luas Wilayah",
    ]
]

X = sm.add_constant(X)

y = df["Jumlah Sekolah"]

model = sm.OLS(y, X).fit()

st.subheader("Ringkasan Regresi")

st.text(model.summary())

df["Prediksi"] = model.predict(X)
df["Residual"] = y - df["Prediksi"]

fig2, ax2 = plt.subplots(figsize=(9,5))

sns.scatterplot(
    x=df["Prediksi"],
    y=df["Residual"],
    color="red",
    ax=ax2,
)

ax2.axhline(0, linestyle="--", color="black")

ax2.set_title("Residual vs Fitted")

st.pyplot(fig2)

# ====================================================
# TAHAP 6
# ====================================================
st.header("6. Interpretasi")

st.success("""
### Kesimpulan

- Penduduk usia sekolah berpengaruh signifikan terhadap jumlah sekolah.
- Luas wilayah juga memberikan pengaruh terhadap kebutuhan pembangunan sekolah.
- Model regresi layak digunakan sebagai dasar analisis.

### Rekomendasi

- Pemerintah sebaiknya memprioritaskan pembangunan sekolah berdasarkan jumlah penduduk usia sekolah.
- Luas wilayah dapat dijadikan faktor tambahan dalam distribusi anggaran pendidikan.
""")
