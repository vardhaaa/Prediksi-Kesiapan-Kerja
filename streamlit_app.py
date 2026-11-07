import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================

st.set_page_config(
    page_title="Prediksi Kesiapan Kerja",
    page_icon="🎓",
    layout="centered"
)

st.markdown("""
<style>
/* Semua wrapper radio harus selebar penuh */
div[data-testid="stRadio"],
div[data-testid="stRadio"] > div,
div[data-testid="stRadio"] > div > div {
    width: 100% !important;
}

/* Container elemen tiap pertanyaan (class st-key-pertanyaan_N) */
div[class*="st-key-pertanyaan_"],
div[class*="st-key-pertanyaan_"] > div,
div[class*="st-key-pertanyaan_"] > div > div {
    width: 100% !important;
    max-width: 100% !important;
}

div[role="radiogroup"] {
    display: flex !important;
    flex-direction: row !important;
    justify-content: space-between !important;
    align-items: flex-start !important;
    width: 100% !important;
    gap: 4px !important;
    padding: 0 !important;
}

/* Sembunyikan lingkaran radio bawaan */
div[role="radiogroup"] label > div:first-child { display: none !important; }

/* Tiap pilihan = 1/5 lebar */
div[role="radiogroup"] > label {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    display: flex !important;
    justify-content: center !important;
    align-items: flex-start !important;
    padding: 12px 0 !important;
    margin: 0 !important;
    cursor: pointer !important;
}

/* Teks: rata tengah, bintang di atas teks */
div[role="radiogroup"] label > div:last-child {
    width: 100%;
    display: flex;
    justify-content: center;
}
div[role="radiogroup"] label p {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    text-align: center !important;
    font-size: 12px !important;
    line-height: 1.3 !important;
    margin: 0 !important;
}

/* Bintang */
div[role="radiogroup"] label p::before {
    content: "★";
    font-size: 34px;
    line-height: 1.1;
    margin-bottom: 4px;
    color: #D1D5DB;
}

/* Bintang terpilih + semua bintang sebelumnya */
div[role="radiogroup"] label:has(input:checked) p::before,
div[role="radiogroup"] label:has(~ label input:checked) p::before {
    color: #F5B301 !important;
}

/* Hover */
div[role="radiogroup"] label:hover p::before,
div[role="radiogroup"] label:has(~ label:hover) p::before {
    color: #FBBF24 !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD MODEL DAN SCALER
# ==========================================

@st.cache_resource
def load_model():
    model = joblib.load("model_knn_perbaikan.pkl")
    scaler = joblib.load("scaler_knn_perbaikan.pkl")
    return model, scaler


try:
    knn, scaler = load_model()

except FileNotFoundError as e:
    st.error(f"File tidak ditemukan: {e.filename}")
    st.info(
        "Pastikan file model_knn_perbaikan.pkl dan "
        "scaler_knn_perbaikan.pkl berada dalam folder "
        "yang sama dengan streamlit_app.py."
    )
    st.stop()


# ==========================================
# 3. DAFTAR 19 PERTANYAAN
# ==========================================

feature_names = [
    "Saya sudah memiliki rencana karier yang jelas setelah lulus kuliah",
    "Saya memahami langkah-langkah yang perlu saya tempuh untuk mencapai tujuan karier saya",
    "Saya tetap semangat dan pantang menyerah meski menghadapi kesulitan",
    "Saya menganggap tantangan sebagai peluang untuk berkembang",
    "Saya percaya diri dengan kemampuan saya menyelesaikan tugas pekerjaan",
    "Saya yakin bisa sukses dalam dunia kerja dengan kemampuan yang saya miliki",
    "Saya merasa optimis dengan masa depan karier saya",
    "Saya memiliki kemampuan komunikasi yang baik",
    "Saya mampu bekerja sama dengan orang lain dalam tim",
    "Saya mampu mengelola waktu dan menyelesaikan tugas tepat waktu",
    "Saya merasa sangat termotivasi untuk segera bekerja setelah lulus",
    "Saya secara aktif mencari peluang kerja atau pelatihan yang bisa mendukung karier saya",
    "Saya mampu mengatur dan mengelola waktu serta prioritas pekerjaan",
    "Saya memiliki kemampuan dalam memimpin dan mengoordinasi tim",
    "Saya pernah mengikuti program magang/PKL yang relevan dengan jurusan saya",
    "Saya memiliki keterampilan teknis yang sesuai dengan bidang studi saya",
    "Saya telah mengikuti pelatihan atau kursus tambahan yang mendukung keahlian teknis saya",
    "Saya memiliki sertifikasi yang diakui dalam bidang keahlian saya",
    "Saya telah menerapkan keterampilan teknis saya dalam proyek akademik atau non-akademik"
]

LABEL_SKALA = {
    1: "Sangat Tidak Setuju",
    2: "Tidak Setuju",
    3: "Netral",
    4: "Setuju",
    5: "Sangat Setuju"
}


# ==========================================
# 4. TAMPILAN UTAMA
# ==========================================

st.title("🎓 Prediksi Kesiapan Kerja")

st.write(
    """
    Aplikasi ini digunakan untuk memprediksi tingkat kesiapan
    kerja mahasiswa berdasarkan jawaban survei menggunakan
    algoritma **K-Nearest Neighbors (KNN)**.
    """
)

st.divider()

st.subheader("Petunjuk Pengisian")

st.write(
    "Silakan jawab seluruh pertanyaan sesuai dengan kondisi "
    "dan pengalaman yang kamu miliki."
)

st.markdown(
    """
    **Skala Penilaian:**

    - 1 = Sangat Tidak Setuju
    - 2 = Tidak Setuju
    - 3 = Netral
    - 4 = Setuju
    - 5 = Sangat Setuju
    """
)

st.divider()


# ==========================================
# 5. FORMULIR PERTANYAAN
# ==========================================

jawaban = {}

with st.form("form_prediksi"):

    st.subheader("Kuesioner Kesiapan Kerja")

    for i, pertanyaan in enumerate(feature_names):

        # Card pertanyaan
        st.markdown(
            f"""
            <div style="
                background-color: rgba(37, 99, 235, 0.10);
                padding: 20px;
                border-radius: 12px;
                margin-bottom: 15px;
                text-align: center;
            ">
                <p style="
                    font-weight: bold;
                    font-size: 17px;
                    margin: 0;
                    text-align: center;
                ">
                    {i + 1}. {pertanyaan}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        nilai = st.radio(
            label=f"Jawaban pertanyaan {i + 1}",
            options=[1, 2, 3, 4, 5],
            index=None,  # tidak ada jawaban default
            horizontal=True,
            format_func=lambda x: LABEL_SKALA[x],
            key=f"pertanyaan_{i}",
            label_visibility="collapsed"
        )

        jawaban[pertanyaan] = nilai

        st.divider()

    submitted = st.form_submit_button(
        "Prediksi Kesiapan Kerja",
        use_container_width=True
    )


# ==========================================
# 6. PROSES PREDIKSI
# ==========================================

if submitted:

    # Validasi: semua pertanyaan harus dijawab
    belum = [
        i + 1 for i, q in enumerate(feature_names)
        if jawaban[q] is None
    ]

    if belum:
        st.warning(
            "Mohon jawab semua pertanyaan. "
            f"Belum dijawab: nomor {', '.join(map(str, belum))}"
        )
        st.stop()

    # Mengubah jawaban menjadi DataFrame
    data_input = pd.DataFrame(
        [[jawaban[col] for col in feature_names]],
        columns=feature_names
    )

    try:

        # Proses scaling
        data_scaled = scaler.transform(data_input)

        # Prediksi menggunakan KNN
        hasil_prediksi = knn.predict(data_scaled)[0]

        st.divider()
        st.subheader("Hasil Prediksi")

        if hasil_prediksi == "Siap":

            st.success("🎉 KATEGORI KESIAPAN KERJA: SIAP")
            st.write(
                "Berdasarkan jawaban yang diberikan, "
                "kamu termasuk dalam kategori siap kerja."
            )

        elif hasil_prediksi == "Cukup Siap":

            st.warning("📘 KATEGORI KESIAPAN KERJA: CUKUP SIAP")
            st.write(
                "Berdasarkan jawaban yang diberikan, "
                "kamu termasuk dalam kategori cukup siap kerja."
            )

        elif hasil_prediksi == "Tidak Siap":

            st.error("📌 KATEGORI KESIAPAN KERJA: TIDAK SIAP")
            st.write(
                "Berdasarkan jawaban yang diberikan, "
                "kamu termasuk dalam kategori tidak siap kerja."
            )

        else:

            st.info(f"Kategori hasil prediksi: {hasil_prediksi}")

        # ==================================
        # 7. PROBABILITAS PREDIKSI
        # ==================================

        if hasattr(knn, "predict_proba"):

            st.divider()
            st.subheader("Probabilitas Prediksi")

            probabilitas = knn.predict_proba(data_scaled)[0]

            hasil_probabilitas = pd.DataFrame({
                "Kategori": knn.classes_,
                "Probabilitas": probabilitas
            })

            for _, row in hasil_probabilitas.iterrows():

                persentase = row["Probabilitas"] * 100

                st.write(
                    f"**{row['Kategori']}**: "
                    f"{persentase:.2f}%"
                )

                st.progress(float(row["Probabilitas"]))

        st.caption(
            "Hasil prediksi diperoleh berdasarkan pola data "
            "yang telah dipelajari oleh model KNN."
        )

    except Exception as e:

        st.error(
            f"Terjadi kesalahan saat melakukan prediksi: {e}"
        )