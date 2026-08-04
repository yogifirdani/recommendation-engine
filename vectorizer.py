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
    
    
    vectorizer = TfidfVectorizer(
        analyzer='word', 
        ngram_range=(1, 2), 
        min_df=1, 
        sublinear_tf=True
    )
    
    tfidf_matrix = vectorizer.fit_transform(corpus_list)
    
    # Menyimpan model ke berkas .pkl
    joblib.dump(vectorizer, MODEL_PATH)
    logger.info(f"Model TfidfVectorizer berhasil disimpan di: {MODEL_PATH}")
    
    return vectorizer, tfidf_matrix

def load_vectorizer():
    if not os.path.exists(MODEL_PATH):
        logger.warning(f"Berkas model vectorizer tidak ditemukan di '{MODEL_PATH}'. Melakukan inisialisasi otomatis...")
        try:
            from database import get_active_destinations
            from preprocessor import build_combined_features
            
            # Ambil destinasi aktif
            df_dest = get_active_destinations()
            if df_dest.empty:
                raise RuntimeError("Tidak ada destinasi aktif di database untuk melatih model.")
                
            # Bentuk korpus
            corpus_list = df_dest.apply(build_combined_features, axis=1).tolist()
            
            # Latih dan simpan
            vectorizer, _ = fit_and_save_vectorizer(corpus_list)
            logger.info("Inisialisasi model otomatis berhasil diselesaikan.")
            return vectorizer
        except Exception as e:
            error_msg = f"Gagal menginisialisasi model secara otomatis: {str(e)}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
    logger.info("Memuat model TfidfVectorizer dari berkas pkl...")
    vectorizer = joblib.load(MODEL_PATH)
    return vectorizer

def transform_preference(vectorizer, preference_text):
    
    logger.info("Melakukan transformasi teks preferensi pengguna menjadi vektor...")
    # Masukkan sebagai list agar dapat diproses oleh vectorizer
    pref_matrix = vectorizer.transform([preference_text])
    # Mengembalikan representasi numpy array baris pertama
    return pref_matrix.toarray()[0]

def calculate_similarity(preference_vector, package_vectors_list):
    
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

def get_top_n(similarity_scores, package_ids, n=3):
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