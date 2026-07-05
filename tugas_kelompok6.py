"""
==============================================================================
ANALISIS HUBUNGAN JUMLAH PENDUDUK USIA SEKOLAH DENGAN PERSEBARAN SEKOLAH DI INDONESIA
==============================================================================
Proyek Akhir — Analisis Data Statistik
Dashboard interaktif Streamlit dengan alur:
Beranda -> Masalah -> Data -> EDA -> Modeling -> Kesimpulan

Cara menjalankan:
    streamlit run app.py

Dataset default: complete_data.csv (letakkan di folder yang sama dengan
app.py), atau unggah file CSV lain lewat sidebar.
==============================================================================
"""

import os
import glob

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
    page_title="Analisis Sebaran Sekolah per Provinsi",
    page_icon="🏫",
    layout="wide"
)

# ==============================================================================
# DATA KELOMPOK
# ==============================================================================

NAMA_MATA_KULIAH = "Analisis Data Statistik"
NAMA_DOSEN = "Nama Dosen"  # ganti sesuai dosen pengampu
JUDUL_PROYEK = "Analisis Hubungan Penduduk Usia Sekolah dan Jumlah Sekolah per Provinsi"

anggota_kelompok = [
    {"nama": "Fariz Firmansyah", "nim": "0102525005"},
    {"nama": "Sayid Muhammad Al Husain", "nim": "0102525018"},
    {"nama": "Muhammad Farrel Inzaghi Santoso", "nim": "0102525023"},
    {"nama": "Azzindan Zulvan", "nim": "0102525701"},
]


def buat_inisial(nama):
    kata = nama.split()
    if len(kata) >= 2:
        return (kata[0][0] + kata[1][0]).upper()
    return nama[:2].upper()


CSS_GLOBAL = """
<style>
.kartu-anggota {
    display: flex;
    align-items: center;
    gap: 12px;
    background-color: rgba(128,128,128,0.08);
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 8px;
    height: 100%;
}
.avatar-inisial {
    flex-shrink: 0;
    width: 42px;
    height: 42px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 15px;
}
.info-anggota .nama { font-weight: 600; font-size: 14px; line-height: 1.3; }
.info-anggota .nim { font-size: 12.5px; opacity: 0.7; }
.kotak-info {
    background-color: rgba(99,102,241,0.08);
    border-left: 4px solid #6366f1;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
</style>
"""
st.markdown(CSS_GLOBAL, unsafe_allow_html=True)


def tampilkan_kartu_kelompok():
    st.markdown("#### 👥 Kelompok")
    kolom_anggota = st.columns(len(anggota_kelompok))
    for kolom, anggota in zip(kolom_anggota, anggota_kelompok):
        with kolom:
            st.markdown(
                f"""
                <div class="kartu-anggota">
                    <div class="avatar-inisial">{buat_inisial(anggota['nama'])}</div>
                    <div class="info-anggota">
                        <div class="nama">{anggota['nama']}</div>
                        <div class="nim">{anggota['nim']}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ==============================================================================
# FUNGSI BANTUAN DATA
# ==============================================================================

@st.cache_data
def muat_data_default():
    """Coba cari complete_data.csv di folder yang sama dengan app.py."""
    kandidat = glob.glob("complete_data*.csv")
    if kandidat:
        return pd.read_csv(kandidat[0]), kandidat[0]
    return None, None


@st.cache_data
def bersihkan_data(df):
    df = df.drop_duplicates()
    df = df.dropna(subset=[
        "province_name", "school_name",
        "total_population", "total_education_age_population"
    ])
    return df


@st.cache_data
def buat_data_provinsi(df):
    agg_dict = dict(
        jumlah_sekolah=("school_name", "count"),
        total_penduduk=("total_population", "max"),
        penduduk_usia_sekolah=("total_education_age_population", "max"),
    )
    if "city_name" in df.columns:
        agg_dict["jumlah_kabupaten_kota"] = ("city_name", "nunique")

    provinsi = df.groupby("province_name", as_index=False).agg(**agg_dict)
    return provinsi


# ==============================================================================
# SIDEBAR — SUMBER DATA
# ==============================================================================

st.sidebar.header("⚙️ Sumber Data")

df_default, nama_file_default = muat_data_default()

file_upload = st.sidebar.file_uploader(
    "Unggah file CSV lain (opsional)",
    type=["csv"]
)

if file_upload is not None:
    df = pd.read_csv(file_upload)
    sumber_data = file_upload.name
elif df_default is not None:
    df = df_default
    sumber_data = nama_file_default
    st.sidebar.success(f"Memuat otomatis: {nama_file_default}")
else:
    st.sidebar.warning("complete_data.csv tidak ditemukan. Silakan unggah file CSV.")
    st.stop()

kolom_wajib = {
    "province_name", "school_name", "total_population", "total_education_age_population"
}
if not kolom_wajib.issubset(set(df.columns)):
    st.error(
        "File CSV tidak memiliki kolom yang dibutuhkan: "
        f"{', '.join(sorted(kolom_wajib))}"
    )
    st.stop()

df_bersih = bersihkan_data(df)
provinsi = buat_data_provinsi(df_bersih)

# ==============================================================================
# NAVIGASI (TAB)
# ==============================================================================

tab_beranda, tab_masalah, tab_data, tab_eda, tab_model, tab_kesimpulan = st.tabs(
    ["🏠 Beranda", "❓ Masalah", "📂 Data", "📊 EDA", "🧮 Modeling", "✅ Kesimpulan"]
)

# ------------------------------------------------------------------------------
# TAB 1 — BERANDA
# ------------------------------------------------------------------------------
with tab_beranda:
    st.title(f"🏫 {JUDUL_PROYEK}")
    st.markdown(
        f"""
        <div class="kotak-info">
        <b>Mata Kuliah:</b> {NAMA_MATA_KULIAH}<br>
        <b>Dosen Pengampu:</b> {NAMA_DOSEN}
        </div>
        """,
        unsafe_allow_html=True
    )
    tampilkan_kartu_kelompok()

    st.markdown("---")
    st.subheader("Ringkasan Proyek")
    st.write(
        """
        Ketersediaan sekolah yang merata di setiap provinsi merupakan salah satu
        indikator penting pemerataan akses pendidikan di Indonesia. Proyek ini
        menganalisis apakah jumlah sekolah yang berdiri di suatu provinsi
        berhubungan dengan jumlah penduduk usia sekolah dan total penduduk di
        provinsi tersebut, menggunakan data sebaran sekolah (jenjang SD hingga
        SMK/SMA/SLB) di seluruh provinsi Indonesia.
        """
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Jumlah Sekolah (baris data)", f"{len(df_bersih):,}")
    m2.metric("Jumlah Provinsi", df_bersih["province_name"].nunique())
    if "city_name" in df_bersih.columns:
        m3.metric("Jumlah Kabupaten/Kota", df_bersih["city_name"].nunique())
    if "stage" in df_bersih.columns:
        m4.metric("Jenjang Pendidikan", df_bersih["stage"].nunique())

# ------------------------------------------------------------------------------
# TAB 2 — MASALAH
# ------------------------------------------------------------------------------
with tab_masalah:
    st.header("❓ Definisi Masalah")
    st.write(
        """
        Pemerataan jumlah sekolah menjadi salah satu tolok ukur keberhasilan
        pembangunan pendidikan di suatu wilayah. Provinsi dengan jumlah penduduk
        usia sekolah yang besar idealnya diimbangi dengan jumlah sekolah yang
        memadai. Namun demikian, kondisi geografis, kepadatan penduduk, dan
        kebijakan daerah dapat menyebabkan ketimpangan antara jumlah penduduk
        usia sekolah dan ketersediaan sekolah.
        """
    )

    st.subheader("Pertanyaan Riset")
    st.markdown(
        """
        - **Q1:** Apakah jumlah penduduk usia sekolah berhubungan secara
          signifikan dengan jumlah sekolah di suatu provinsi?
        - **Q2:** Apakah total penduduk provinsi turut memengaruhi jumlah
          sekolah, atau penduduk usia sekolah adalah faktor yang lebih dominan?
        """
    )

    st.subheader("Hipotesis")
    col_h0, col_h1 = st.columns(2)
    with col_h0:
        st.info(
            "**H0:** Tidak ada hubungan signifikan antara penduduk usia "
            "sekolah/total penduduk dengan jumlah sekolah per provinsi."
        )
    with col_h1:
        st.success(
            "**H1:** Terdapat hubungan signifikan antara penduduk usia "
            "sekolah/total penduduk dengan jumlah sekolah per provinsi."
        )

# ------------------------------------------------------------------------------
# TAB 3 — DATA
# ------------------------------------------------------------------------------
with tab_data:
    st.header("📂 Pengumpulan & Inspeksi Data")
    st.caption(f"Sumber data: `{sumber_data}`")

    st.subheader("Pratinjau Data Mentah")
    st.dataframe(df.head(10), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Baris (sebelum cleaning)", f"{len(df):,}")
    c2.metric("Baris (setelah cleaning)", f"{len(df_bersih):,}")
    c3.metric("Baris terhapus", f"{len(df) - len(df_bersih):,}")

    if "stage" in df_bersih.columns:
        st.subheader("Distribusi Jenjang Sekolah")
        colA, colB = st.columns([1, 1])
        with colA:
            st.dataframe(
                df_bersih["stage"].value_counts().rename("jumlah"),
                use_container_width=True
            )
        with colB:
            fig_jenjang, ax_jenjang = plt.subplots(figsize=(6, 4))
            df_bersih["stage"].value_counts().plot(kind="bar", ax=ax_jenjang, color="#6366f1")
            ax_jenjang.set_ylabel("Jumlah Sekolah")
            ax_jenjang.set_title("Jumlah Sekolah per Jenjang")
            st.pyplot(fig_jenjang)

    if "status" in df_bersih.columns:
        st.subheader("Distribusi Status Sekolah (Negeri/Swasta)")
        st.bar_chart(df_bersih["status"].value_counts())

    st.subheader("Data Agregat per Provinsi")
    st.dataframe(provinsi, use_container_width=True)

    if len(provinsi) < 3:
        st.warning(
            "⚠️ Jumlah provinsi kurang dari 3, hasil uji statistik mungkin "
            "kurang bermakna, namun perhitungan tetap ditampilkan."
        )

# ------------------------------------------------------------------------------
# TAB 4 — EDA
# ------------------------------------------------------------------------------
with tab_eda:
    st.header("📊 Exploratory Data Analysis (EDA)")

    st.subheader("Statistik Deskriptif")
    st.dataframe(provinsi.describe(), use_container_width=True)

    st.subheader("Boxplot")
    fig1, ax1 = plt.subplots(1, 2, figsize=(10, 5))
    sns.boxplot(y=provinsi["jumlah_sekolah"], ax=ax1[0])
    ax1[0].set_title("Jumlah Sekolah")
    sns.boxplot(y=provinsi["penduduk_usia_sekolah"], ax=ax1[1])
    ax1[1].set_title("Penduduk Usia Sekolah")
    plt.tight_layout()
    st.pyplot(fig1)
    st.caption(
        "Boxplot menunjukkan sebaran nilai antar provinsi serta provinsi-"
        "provinsi yang berpotensi menjadi outlier (misalnya provinsi dengan "
        "jumlah sekolah atau penduduk usia sekolah jauh di atas rata-rata)."
    )

    st.subheader("Scatter Plot")
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=provinsi, x="penduduk_usia_sekolah", y="jumlah_sekolah", s=80, ax=ax2)
    ax2.set_title("Penduduk Usia Sekolah vs Jumlah Sekolah")
    ax2.grid(True)
    st.pyplot(fig2)
    st.caption(
        "Pola sebaran titik memperlihatkan kecenderungan arah hubungan "
        "(positif/negatif) antara penduduk usia sekolah dan jumlah sekolah."
    )

    st.subheader("Heatmap Korelasi Pearson")
    kolom_korelasi = ["jumlah_sekolah", "penduduk_usia_sekolah", "total_penduduk"]
    corr = provinsi[kolom_korelasi].corr(method="pearson")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.dataframe(corr.style.format("{:.3f}"), use_container_width=True)
    with col_b:
        fig3, ax3 = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=True, cmap="RdYlBu", fmt=".3f", ax=ax3)
        ax3.set_title("Heatmap Korelasi Pearson")
        st.pyplot(fig3)

# ------------------------------------------------------------------------------
# TAB 5 — MODELING
# ------------------------------------------------------------------------------
with tab_model:
    st.header("🧮 Modeling")

    st.subheader("Uji Signifikansi Korelasi Pearson")
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
        st.write(f"r = {r1:.4f}  |  p = {p1:.5f}")
        st.success("Hubungan signifikan") if p1 < 0.05 else st.warning("Hubungan tidak signifikan")
    with col_y:
        st.markdown("**Total Penduduk vs Jumlah Sekolah**")
        st.write(f"r = {r2:.4f}  |  p = {p2:.5f}")
        st.success("Hubungan signifikan") if p2 < 0.05 else st.warning("Hubungan tidak signifikan")

    st.subheader("Regresi Linear Berganda")
    X = provinsi[["penduduk_usia_sekolah", "total_penduduk"]].copy()
    Y = provinsi["jumlah_sekolah"].copy()
    X = X.apply(pd.to_numeric)
    Y = pd.to_numeric(Y)
    X = sm.add_constant(X)

    model = sm.OLS(Y, X).fit()

    with st.expander("📄 Lihat Ringkasan Model Regresi (model.summary())", expanded=True):
        st.text(model.summary())

    provinsi["prediksi"] = model.predict(X)
    provinsi["residual"] = Y - provinsi["prediksi"]

    st.subheader("Prediksi vs Aktual")
    st.dataframe(
        provinsi[["province_name", "jumlah_sekolah", "prediksi", "residual"]],
        use_container_width=True
    )

    col_p, col_r = st.columns(2)
    with col_p:
        fig4, ax4 = plt.subplots(figsize=(6, 5))
        ax4.scatter(Y, provinsi["prediksi"], s=70)
        ax4.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color="red")
        ax4.set_xlabel("Data Aktual")
        ax4.set_ylabel("Prediksi")
        ax4.set_title("Aktual vs Prediksi")
        ax4.grid(True)
        st.pyplot(fig4)
    with col_r:
        fig5, ax5 = plt.subplots(figsize=(6, 5))
        sns.histplot(provinsi["residual"], bins=8, kde=True, ax=ax5)
        ax5.set_title("Distribusi Residual")
        st.pyplot(fig5)

# ------------------------------------------------------------------------------
# TAB 6 — KESIMPULAN
# ------------------------------------------------------------------------------
with tab_kesimpulan:
    st.header("✅ Interpretasi & Kesimpulan")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("R²", round(model.rsquared, 3))
    k2.metric("Adjusted R²", round(model.rsquared_adj, 3))
    k3.metric("F Statistic", round(model.fvalue, 3))
    k4.metric("Prob (F)", f"{model.f_pvalue:.5f}")

    if model.f_pvalue < 0.05:
        st.success("Model regresi **signifikan** secara statistik (p < 0.05).")
    else:
        st.warning("Model regresi **tidak signifikan** secara statistik (p ≥ 0.05).")

    st.subheader("Jawaban Pertanyaan Riset")
    st.markdown(
        f"""
        - **Q1 (Penduduk usia sekolah vs jumlah sekolah):** r = {r1:.3f},
          p = {p1:.5f} → {"hubungan signifikan" if p1 < 0.05 else "hubungan tidak signifikan"}.
        - **Q2 (Faktor dominan):** berdasarkan koefisien regresi, variabel
          dengan nilai koefisien dan p-value terbaik pada ringkasan model di
          atas menunjukkan faktor yang paling berpengaruh terhadap jumlah
          sekolah per provinsi.
        """
    )

    st.subheader("Rekomendasi")
    st.write(
        """
        Provinsi dengan jumlah penduduk usia sekolah tinggi namun jumlah
        sekolah relatif rendah (dapat dilihat dari residual negatif pada
        tabel prediksi) perlu menjadi prioritas pembangunan sekolah baru
        agar rasio ketersediaan sekolah terhadap penduduk usia sekolah lebih
        merata di seluruh Indonesia.
        """
    )

    st.caption("Dibuat dengan Streamlit • Pandas • Statsmodels • Seaborn")
