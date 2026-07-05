"""
==============================================================================
STREAMLIT APP — TUGAS KELOMPOK 6
Analisis Pengaruh Penduduk Usia Sekolah & Luas Wilayah terhadap Jumlah Sekolah
==============================================================================
Mata Kuliah   : Analisis Data Statistik
Dosen         : Tri Aji Nugroho
Kelompok      : Kelompok 6

Cara menjalankan:
    streamlit run app_kelompok6.py
==============================================================================
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set_theme(style="whitegrid", palette="muted")

# ==============================================================================
# KONFIGURASI HALAMAN
# ==============================================================================

st.set_page_config(
    page_title="Tugas Kelompok 6 - Analisis Sekolah",
    page_icon="🏫",
    layout="wide"
)

st.title("🏫 Analisis Pengaruh Penduduk Usia Sekolah & Luas Wilayah terhadap Jumlah Sekolah")

st.markdown(
    """
    <div style="background-color: rgba(99,102,241,0.08); border-left: 4px solid #6366f1;
                border-radius: 8px; padding: 14px 18px; margin-bottom: 10px;">
    <b>Mata Kuliah:</b> Analisis Data Statistik<br>
    <b>Dosen:</b> Tri Aji Nugroho<br>
    <b>Kelompok:</b> Kelompok 6
    </div>
    """,
    unsafe_allow_html=True
)

# ==============================================================================
# NAVIGASI (TAB)
# ==============================================================================

tab_masalah, tab_data, tab_cleaning, tab_eda, tab_model, tab_kesimpulan = st.tabs(
    ["❓ Problem Definition", "📥 Data Collection", "🧹 Data Cleaning",
     "📊 EDA", "🧮 Statistical Modeling", "✅ Interpret & Communicate"]
)

# ------------------------------------------------------------------------------
# TAHAP 1 — PROBLEM DEFINITION
# ------------------------------------------------------------------------------
with tab_masalah:
    st.header("❓ Problem Definition")

    st.subheader("Rumusan Masalah")
    st.markdown(
        """
        1. Bagaimana karakteristik persebaran penduduk usia sekolah, luas
           wilayah, dan jumlah sekolah di Indonesia?
        2. Apakah terdapat hubungan linear yang kuat antara jumlah penduduk
           usia sekolah dan luas wilayah terhadap jumlah sekolah?
        3. Faktor mana yang memberikan kontribusi paling signifikan terhadap
           pembangunan sekolah baru?
        """
    )

    st.subheader("Hipotesis")
    col_h0, col_h1 = st.columns(2)
    with col_h0:
        st.info(
            "**H0:** Tidak ada pengaruh signifikan antara penduduk usia "
            "sekolah dan luas wilayah terhadap jumlah sekolah."
        )
    with col_h1:
        st.success(
            "**H1:** Terdapat pengaruh positif dan signifikan dari penduduk "
            "usia sekolah dan luas wilayah terhadap jumlah sekolah."
        )

# ------------------------------------------------------------------------------
# TAHAP 2 — DATA COLLECTION
# ------------------------------------------------------------------------------
with tab_data:
    st.header("📥 Data Collection")

    st.caption(
        "Dataset pada tahap ini disimulasikan (bukan data riil) mengikuti "
        "pola hubungan penduduk usia sekolah dan luas wilayah terhadap "
        "jumlah sekolah, sesuai skrip asli tugas kelompok."
    )

    n_prov = st.sidebar.slider("Jumlah Provinsi (simulasi)", min_value=10, max_value=50, value=38)
    seed = st.sidebar.number_input("Random Seed", min_value=0, value=42, step=1)

    @st.cache_data
    def buat_dataset(n_prov, seed):
        np.random.seed(seed)
        total_pop = np.random.normal(3000000, 1000000, n_prov)
        school_age = total_pop * np.random.uniform(0.18, 0.22, n_prov)
        area = np.random.normal(50000, 15000, n_prov)
        schools = (school_age * 0.003) + (area * 0.01) + np.random.normal(0, 100, n_prov)

        df = pd.DataFrame({
            "provinsi": [f"Provinsi_{i+1}" for i in range(n_prov)],
            "penduduk_usia_sekolah": school_age.astype(int),
            "total_penduduk": total_pop.astype(int),
            "luas_wilayah": area.astype(int),
            "jumlah_sekolah": schools.astype(int),
        })
        return df

    df = buat_dataset(n_prov, seed)

    st.subheader("Inspeksi Awal Data")
    st.write(f"**Dimensi Dataset:** {df.shape[0]} Baris (Provinsi), {df.shape[1]} Kolom")
    st.dataframe(df.head(10), use_container_width=True)

# ------------------------------------------------------------------------------
# TAHAP 3 — DATA CLEANING
# ------------------------------------------------------------------------------
with tab_cleaning:
    st.header("🧹 Data Cleaning")

    kolom_numerik = ["penduduk_usia_sekolah", "total_penduduk", "luas_wilayah", "jumlah_sekolah"]

    st.subheader("Pengecekan Missing Values")
    st.dataframe(df.isnull().sum().rename("jumlah_missing"), use_container_width=True)

    st.subheader("Pengecekan Duplikasi Data")
    st.write(f"Jumlah baris duplikat: **{df.duplicated().sum()}**")

    df_bersih = df.copy()
    for col in kolom_numerik:
        df_bersih = df_bersih[df_bersih[col] >= 0]

    st.subheader("Validasi Logis")
    st.write("Data numerik divalidasi agar tidak bernilai negatif.")
    c1, c2 = st.columns(2)
    c1.metric("Baris Sebelum Cleaning", len(df))
    c2.metric("Baris Setelah Cleaning", len(df_bersih))

# ------------------------------------------------------------------------------
# TAHAP 4 — EDA
# ------------------------------------------------------------------------------
with tab_eda:
    st.header("📊 Exploratory Data Analysis (EDA)")

    st.subheader("Tabel Statistik Deskriptif Nasional")
    deskriptif = df_bersih[kolom_numerik].describe().T[["mean", "std", "min", "50%", "max"]]
    deskriptif.columns = ["Rata-rata", "Std Deviasi", "Minimum", "Median", "Maksimum"]
    st.dataframe(deskriptif, use_container_width=True)

    st.subheader("Matriks Korelasi Pearson")
    corr_matrix = df_bersih[kolom_numerik].corr(method="pearson")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.dataframe(corr_matrix.style.format("{:.2f}"), use_container_width=True)
    with col_b:
        fig_corr, ax_corr = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Blues",
                    cbar=True, annot_kws={"size": 10}, ax=ax_corr)
        ax_corr.set_title("Matriks Korelasi Pearson Antar Variabel", fontweight="bold")
        st.pyplot(fig_corr)

    st.info(
        """
        **Kesimpulan Analisis Matriks Korelasi**
        1. **Multikolinieritas Indikatif:** Korelasi sangat tinggi (mendekati 1.00)
           antara `total_penduduk` dan `penduduk_usia_sekolah`, karena usia sekolah
           adalah subset dari total penduduk. Memasukkan keduanya sekaligus dalam
           regresi akan memicu kolinieritas.
        2. **Korelasi Target:** `penduduk_usia_sekolah` berkorelasi positif sangat
           kuat dengan `jumlah_sekolah`, mengonfirmasi bahwa demografi usia
           spesifik berjalan beriringan dengan jumlah fasilitas pendidikan.
        """
    )

# ------------------------------------------------------------------------------
# TAHAP 5 — STATISTICAL MODELING
# ------------------------------------------------------------------------------
with tab_model:
    st.header("🧮 Statistical Modeling")

    st.subheader("Uji Asumsi Multikolinieritas (VIF)")
    X_check = df_bersih[["penduduk_usia_sekolah", "total_penduduk", "luas_wilayah"]]
    X_check_const = sm.add_constant(X_check)

    vif_table = pd.DataFrame()
    vif_table["Variabel Independen"] = X_check_const.columns
    vif_table["Nilai VIF"] = [
        variance_inflation_factor(X_check_const.values, i)
        for i in range(X_check_const.shape[1])
    ]
    st.dataframe(vif_table, use_container_width=True)
    st.warning(
        "**Keputusan Analitik:** Variabel `total_penduduk` di-drop pada pemodelan "
        "karena memiliki nilai VIF sangat tinggi (redundan dengan `penduduk_usia_sekolah`)."
    )

    st.subheader("Model Regresi OLS Final")
    X_final = df_bersih[["penduduk_usia_sekolah", "luas_wilayah"]]
    X_final = sm.add_constant(X_final)
    y = df_bersih["jumlah_sekolah"]

    model_regresi = sm.OLS(y, X_final).fit()

    with st.expander("📄 Lihat Ringkasan Model Regresi (model.summary())", expanded=True):
        st.text(model_regresi.summary())

    df_bersih["prediksi"] = model_regresi.predict(X_final)
    df_bersih["residual"] = y - df_bersih["prediksi"]

    st.subheader("Plot Residual vs Fitted Values (Uji Homoskedastisitas)")
    fig_res, ax_res = plt.subplots(figsize=(9, 5))
    sns.scatterplot(x=df_bersih["prediksi"], y=df_bersih["residual"], color="red", s=70, alpha=0.7, ax=ax_res)
    ax_res.axhline(y=0, color="black", linestyle="--")
    ax_res.set_title("Plot Residual vs Fitted Values", fontweight="bold")
    ax_res.set_xlabel("Nilai Prediksi Jumlah Sekolah")
    ax_res.set_ylabel("Residual (Error)")
    st.pyplot(fig_res)

    st.info(
        """
        **Kesimpulan Grafik Residual vs Fitted Values**
        1. **Asumsi Homoskedastisitas Terpenuhi:** titik-titik residual tersebar
           acak di atas dan di bawah garis nol tanpa membentuk pola tertentu
           (tidak mekar seperti terompet).
        2. **Makna:** varians galat dari model konstan. Model regresi OLS stabil,
           seimbang, dan valid digunakan untuk memprediksi kebutuhan infrastruktur
           pendidikan.
        """
    )

# ------------------------------------------------------------------------------
# TAHAP 6 — INTERPRET & COMMUNICATE
# ------------------------------------------------------------------------------
with tab_kesimpulan:
    st.header("✅ Interpret & Communicate")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("R²", round(model_regresi.rsquared, 3))
    k2.metric("Adjusted R²", round(model_regresi.rsquared_adj, 3))
    k3.metric("F Statistic", round(model_regresi.fvalue, 3))
    k4.metric("Prob (F)", f"{model_regresi.f_pvalue:.5f}")

    st.subheader("Validasi Hipotesis")
    if model_regresi.f_pvalue < 0.05:
        st.success(
            "Model regresi berhasil menolak H0 dan mengonfirmasi H1. Penduduk "
            "usia sekolah dan luas wilayah secara simultan berpengaruh signifikan "
            "terhadap sebaran jumlah sekolah di Indonesia."
        )
    else:
        st.warning("Model regresi tidak berhasil menolak H0 (tidak signifikan secara statistik).")

    st.subheader("Insight Pemodelan")
    st.write(
        """
        Koefisien regresi menunjukkan bahwa pertumbuhan populasi usia sekolah
        memiliki daya dorong yang jauh lebih absolut terhadap penambahan sekolah
        baru dibandingkan ukuran geografis wilayah tersebut.
        """
    )

    st.subheader("Rekomendasi Kebijakan Publik")
    st.write(
        """
        Dalam mengalokasikan Dana Alokasi Khusus (DAK) Pendidikan, pemerintah
        sebaiknya:
        - Menjadikan tren demografi usia sekolah (bukan sekadar rasio penduduk
          umum) sebagai bobot utama distribusi anggaran.
        - Menggunakan faktor luas wilayah murni sebagai penyeimbang guna mencegah
          terjadinya *blank-spot* aksesibilitas pendidikan di provinsi yang
          sangat luas.
        """
    )

    st.subheader("Keterbatasan & Saran Eksplorasi")
    st.write(
        """
        Penelitian ini belum mengontrol aspek fiskal daerah (contoh: APBD) dan
        tipografi geografis (kepulauan vs kontinental). Penelitian selanjutnya
        direkomendasikan menyertakan variabel ekonomi makro untuk meningkatkan
        akurasi R-squared model.
        """
    )

    st.caption("Dibuat dengan Streamlit • Pandas • Statsmodels • Seaborn")
