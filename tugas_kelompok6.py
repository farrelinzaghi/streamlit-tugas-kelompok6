"""
==============================================================================
APLIKASI ANALISIS DATA PENDIDIKAN PER PROVINSI
==============================================================================
Aplikasi Streamlit untuk analisis hubungan antara jumlah penduduk usia
sekolah dan jumlah sekolah per provinsi, meliputi:
- Statistik deskriptif
- Visualisasi (boxplot, scatter plot, heatmap korelasi)
- Uji korelasi Pearson
- Regresi Linear Berganda

Cara menjalankan:
    streamlit run app.py
==============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import statsmodels.api as sm

# ==============================================================================
# KONFIGURASI HALAMAN
# ==============================================================================

st.set_page_config(
    page_title="Analisis Data Pendidikan per Provinsi",
    page_icon="🏫",
    layout="wide"
)

st.title("🏫 Analisis Hubungan Penduduk Usia Sekolah dan Jumlah Sekolah")
st.caption("Dashboard interaktif — Statistik Deskriptif, Korelasi Pearson, dan Regresi Linear Berganda")

with st.container(border=True):
    st.markdown("**👥 Kelompok**")
    st.markdown(
        """
        - Fariz Firmansyah – 0102525005
        - Sayid Muhammad Al Husain – 0102525018
        - Muhammad Farrel Inzaghi Santoso – 0102525023
        - Azzindan Zulvan – 0102525701
        """
    )

# ==============================================================================
# FUNGSI BANTUAN
# ==============================================================================

@st.cache_data
def buat_data_dummy():
    """Data dummy sebagai contoh jika pengguna belum mengunggah file."""
    data = {
        "province_name": ["Province A", "Province A", "Province B", "Province B", "Province C", "Province C"],
        "school_name": ["School 1", "School 2", "School 3", "School 4", "School 5", "School 6"],
        "total_population": [100000, 100000, 200000, 200000, 150000, 150000],
        "total_education_age_population": [20000, 20000, 40000, 40000, 30000, 30000],
    }
    return pd.DataFrame(data)


@st.cache_data
def bersihkan_data(df):
    df = df.drop_duplicates()
    df = df.dropna()
    return df


@st.cache_data
def buat_data_provinsi(df):
    provinsi = (
        df.groupby("province_name", as_index=False)
        .agg(
            jumlah_sekolah=("school_name", "count"),
            total_penduduk=("total_population", "max"),
            penduduk_usia_sekolah=("total_education_age_population", "max"),
        )
    )
    return provinsi


# ==============================================================================
# SIDEBAR — SUMBER DATA
# ==============================================================================

st.sidebar.header("⚙️ Sumber Data")

file_upload = st.sidebar.file_uploader(
    "Unggah file CSV (kolom wajib: province_name, school_name, total_population, total_education_age_population)",
    type=["csv"]
)

if file_upload is not None:
    df = pd.read_csv(file_upload)
    st.sidebar.success(f"Berhasil memuat: {file_upload.name}")
else:
    df = buat_data_dummy()
    st.sidebar.info("Belum ada file diunggah. Menggunakan data dummy sebagai contoh.")

kolom_wajib = {
    "province_name", "school_name", "total_population", "total_education_age_population"
}
if not kolom_wajib.issubset(set(df.columns)):
    st.error(
        "File CSV tidak memiliki kolom yang dibutuhkan: "
        f"{', '.join(sorted(kolom_wajib))}"
    )
    st.stop()

# ==============================================================================
# 1. DATASET
# ==============================================================================

st.header("1️⃣ Dataset")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Pratinjau Data")
    st.dataframe(df.head(10), use_container_width=True)

with col2:
    st.subheader("Informasi Dataset")
    st.write(f"**Jumlah baris:** {df.shape[0]}")
    st.write(f"**Jumlah kolom:** {df.shape[1]}")
    st.write("**Tipe data:**")
    st.dataframe(df.dtypes.astype(str).rename("tipe"), use_container_width=True)

# ==============================================================================
# 2. DATA CLEANING
# ==============================================================================

st.header("2️⃣ Pembersihan Data (Data Cleaning)")

df_bersih = bersihkan_data(df)

c1, c2, c3 = st.columns(3)
c1.metric("Data Awal", len(df))
c2.metric("Data Setelah Cleaning", len(df_bersih))
c3.metric("Data Terhapus (duplikat/kosong)", len(df) - len(df_bersih))

# ==============================================================================
# 3. DATA PER PROVINSI
# ==============================================================================

st.header("3️⃣ Data Agregat per Provinsi")

provinsi = buat_data_provinsi(df_bersih)
st.dataframe(provinsi, use_container_width=True)

if len(provinsi) < 3:
    st.warning(
        "⚠️ Jumlah provinsi kurang dari 3. Hasil uji statistik (korelasi & regresi) "
        "mungkin tidak bermakna secara statistik, namun perhitungan tetap ditampilkan."
    )

# ==============================================================================
# 4. STATISTIK DESKRIPTIF
# ==============================================================================

st.header("4️⃣ Statistik Deskriptif")
st.dataframe(provinsi.describe(), use_container_width=True)

# ==============================================================================
# 5. BOXPLOT
# ==============================================================================

st.header("5️⃣ Boxplot")

fig1, ax1 = plt.subplots(1, 2, figsize=(10, 5))
sns.boxplot(y=provinsi["jumlah_sekolah"], ax=ax1[0])
ax1[0].set_title("Jumlah Sekolah")
sns.boxplot(y=provinsi["penduduk_usia_sekolah"], ax=ax1[1])
ax1[1].set_title("Penduduk Usia Sekolah")
plt.tight_layout()
st.pyplot(fig1)

# ==============================================================================
# 6. SCATTER PLOT
# ==============================================================================

st.header("6️⃣ Scatter Plot")

fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=provinsi, x="penduduk_usia_sekolah", y="jumlah_sekolah", s=80, ax=ax2)
ax2.set_title("Penduduk Usia Sekolah vs Jumlah Sekolah")
ax2.grid(True)
st.pyplot(fig2)

# ==============================================================================
# 7. KORELASI PEARSON
# ==============================================================================

st.header("7️⃣ Korelasi Pearson")

kolom_korelasi = ["jumlah_sekolah", "penduduk_usia_sekolah", "total_penduduk"]
corr = provinsi[kolom_korelasi].corr(method="pearson")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Matriks Korelasi")
    st.dataframe(corr.style.format("{:.3f}"), use_container_width=True)

with col_b:
    st.subheader("Heatmap")
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="RdYlBu", fmt=".3f", ax=ax3)
    ax3.set_title("Heatmap Korelasi Pearson")
    st.pyplot(fig3)

# ==============================================================================
# 8. UJI PEARSON
# ==============================================================================

st.header("8️⃣ Uji Signifikansi Pearson")

r1, p1 = pearsonr(
    provinsi["penduduk_usia_sekolah"].astype(float),
    provinsi["jumlah_sekolah"].astype(float),
)
r2, p2 = pearsonr(
    provinsi["total_penduduk"].astype(float),
    provinsi["jumlah_sekolah"].astype(float),
)

col_x, col_y = st.columns(2)

with col_x:
    st.markdown("**Penduduk Usia Sekolah vs Jumlah Sekolah**")
    st.write(f"r = {r1:.4f}")
    st.write(f"p = {p1:.5f}")
    if p1 < 0.05:
        st.success("Hubungan signifikan")
    else:
        st.warning("Hubungan tidak signifikan")

with col_y:
    st.markdown("**Total Penduduk vs Jumlah Sekolah**")
    st.write(f"r = {r2:.4f}")
    st.write(f"p = {p2:.5f}")
    if p2 < 0.05:
        st.success("Hubungan signifikan")
    else:
        st.warning("Hubungan tidak signifikan")

# ==============================================================================
# 9. REGRESI LINEAR BERGANDA
# ==============================================================================

st.header("9️⃣ Regresi Linear Berganda")

X = provinsi[["penduduk_usia_sekolah", "total_penduduk"]].copy()
Y = provinsi["jumlah_sekolah"].copy()

X = X.apply(pd.to_numeric)
Y = pd.to_numeric(Y)
X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()

with st.expander("📄 Lihat Ringkasan Model Regresi (model.summary())", expanded=True):
    st.text(model.summary())

# ==============================================================================
# 10. PREDIKSI
# ==============================================================================

st.header("🔟 Prediksi vs Aktual")

provinsi["prediksi"] = model.predict(X)
provinsi["residual"] = Y - provinsi["prediksi"]

st.dataframe(
    provinsi[["province_name", "jumlah_sekolah", "prediksi", "residual"]],
    use_container_width=True
)

col_p, col_r = st.columns(2)

with col_p:
    st.subheader("Aktual vs Prediksi")
    fig4, ax4 = plt.subplots(figsize=(6, 5))
    ax4.scatter(Y, provinsi["prediksi"], s=70)
    ax4.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color="red")
    ax4.set_xlabel("Data Aktual")
    ax4.set_ylabel("Prediksi")
    ax4.set_title("Aktual vs Prediksi")
    ax4.grid(True)
    st.pyplot(fig4)

with col_r:
    st.subheader("Distribusi Residual")
    fig5, ax5 = plt.subplots(figsize=(6, 5))
    sns.histplot(provinsi["residual"], bins=8, kde=True, ax=ax5)
    ax5.set_title("Distribusi Residual")
    st.pyplot(fig5)

# ==============================================================================
# 11. KESIMPULAN
# ==============================================================================

st.header("✅ Kesimpulan")

k1, k2, k3, k4 = st.columns(4)
k1.metric("R²", round(model.rsquared, 3))
k2.metric("Adjusted R²", round(model.rsquared_adj, 3))
k3.metric("F Statistic", round(model.fvalue, 3))
k4.metric("Prob (F)", f"{model.f_pvalue:.5f}")

if model.f_pvalue < 0.05:
    st.success("Model regresi **signifikan** secara statistik (p < 0.05).")
else:
    st.warning("Model regresi **tidak signifikan** secara statistik (p ≥ 0.05).")

st.caption("Dibuat dengan Streamlit • Pandas • Statsmodels • Seaborn")
