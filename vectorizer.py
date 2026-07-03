import os
import joblib
import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logger = logging.getLogger(__name__)

# Jalur tempat model TF-IDF akan disimpan
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

def fit_and_save_vectorizer(corpus_list):
    """
    Melatih TfidfVectorizer pada seluruh korpus kalimat paket wisata aktif,
    kemudian menyimpan model vectorizer yang telah dilatih ke berkas .pkl.
    
    Return: (vectorizer, tfidf_matrix)
    """
    # Membuat direktori 'models' jika belum ada
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        logger.info(f"Direktori '{MODEL_DIR}' berhasil dibuat.")
        
    logger.info("Mulai melakukan pelatihan (fitting) TfidfVectorizer...")
    
    # INIBAGIAN TF (TERM FREQUENCY)
    # Inisialisasi vectorizer dengan parameter yang ditentukan
    # sublinear_tf=True membantu menormalkan kata-kata yang muncul sangat sering
    # ngram_range=(1, 2) mengizinkan kombinasi 2 kata (Bigram) agar konteks angka tidak tertukar
    # contoh: "3 destinasi" dan "3 hari" akan dibaca berbeda, tidak hanya memisahkan angka "3".
    vectorizer = TfidfVectorizer(
        analyzer='word', 
        ngram_range=(1, 2), 
        min_df=1, 
        sublinear_tf=True
    )
    
    # Melatih model dan mentransformasikannya ke representasi matriks
    # INI BAGIAN IDF (INVERSE DOCUMENT FREQUENCY)
    tfidf_matrix = vectorizer.fit_transform(corpus_list)
    
    # Menyimpan model ke berkas .pkl
    joblib.dump(vectorizer, MODEL_PATH)
    logger.info(f"Model TfidfVectorizer berhasil disimpan di: {MODEL_PATH}")
    
    return vectorizer, tfidf_matrix

def load_vectorizer():
    """
    Memuat model TfidfVectorizer dari berkas .pkl yang sudah disimpan.
    Akan melempar FileNotFoundError jika berkas model tidak ditemukan.
    
    Return: TfidfVectorizer
    """
    if not os.path.exists(MODEL_PATH):
        error_msg = f"Berkas model vectorizer tidak ditemukan di '{MODEL_PATH}'. Silakan jalankan endpoint /vectorize terlebih dahulu untuk melatih model."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    logger.info("Memuat model TfidfVectorizer dari berkas pkl...")
    vectorizer = joblib.load(MODEL_PATH)
    return vectorizer

def transform_preference(vectorizer, preference_text):
    """
    Mentransformasikan preferensi pengguna menjadi vektor TF-IDF
    berdasarkan ruang kosa kata (vocabulary) dari model yang telah dilatih.
    PENTING: Menggunakan transform(), bukan fit_transform()!
    
    Return: numpy 2D array (1, n_features)
    """
    logger.info("Melakukan transformasi teks preferensi pengguna menjadi vektor...")
    # Masukkan sebagai list agar dapat diproses oleh vectorizer
    pref_matrix = vectorizer.transform([preference_text])
    # Mengembalikan representasi numpy array baris pertama
    return pref_matrix.toarray()[0]

def calculate_similarity(preference_vector, package_vectors_list):
    """
    Menghitung skor Cosine Similarity antara satu vektor preferensi 
    dengan daftar vektor paket wisata yang ada.
    
    Return: list of float (skor kemiripan tiap paket wisata)
    """
    logger.info("Mulai menghitung nilai Cosine Similarity...")
    
    # Konversi list vektor dari DB menjadi 2D numpy array
    # Bentuk dimensi: (jumlah_paket, jumlah_fitur)
    pkg_matrix = np.array(package_vectors_list)
    
    if len(pkg_matrix) == 0:
        logger.warning("Daftar vektor paket wisata kosong.")
        return []
        
    # Memastikan format vektor preferensi berdimensi 2D (1, jumlah_fitur)
    pref_vector_2d = preference_vector.reshape(1, -1)
    
    # Hitung cosine similarity
    # Hasil berupa matriks berukuran (1, jumlah_paket)
    similarity_matrix = cosine_similarity(pref_vector_2d, pkg_matrix)
    
    # Ambil baris pertama dan konversikan menjadi list float
    scores = similarity_matrix[0].tolist()
    
    logger.info("Perhitungan Cosine Similarity selesai.")
    return scores

def get_top_n(similarity_scores, package_ids, n=5):
    """
    Memasangkan ID paket wisata dengan skor kemiripannya, mengurutkannya 
    secara menurun (descending), lalu mengambil N teratas.
    
    Return: list of dict berisi hasil pemeringkatan
    """
    # Menggabungkan ID paket wisata dengan skor kemiripan
    package_scores = []
    for pkg_id, score in zip(package_ids, similarity_scores):
        package_scores.append({
            "package_id": int(pkg_id),
            "similarity_score": round(float(score), 4)  # Membulatkan 4 desimal agar rapi
        })
        
    # Urutkan berdasarkan skor kemiripan secara menurun (descending)
    sorted_packages = sorted(package_scores, key=lambda x: x["similarity_score"], reverse=True)
    
    # Berikan peringkat (rank) dan ambil N teratas
    ranked_packages = []
    for i, item in enumerate(sorted_packages[:n], start=1):
        item["rank"] = i
        ranked_packages.append(item)
        
    return ranked_packages
