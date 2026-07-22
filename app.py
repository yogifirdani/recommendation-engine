import logging
import os
import hashlib
import json
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

from database import (
    get_active_packages,
    get_preference_by_id,
    get_all_vectors,
    save_package_vector,
    save_recommendation_result,
    clear_package_vectors
)
from preprocessor import (
    build_combined_features,
    build_preference_features
)
from vectorizer import (
    fit_and_save_vectorizer,
    load_vectorizer,
    transform_preference,
    calculate_similarity,
    get_top_n,
    MODEL_PATH
)
from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG


# Konfigurasi logging dasar ke console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# PENTING: Mengembalikan JSON dengan format yang rapi saat didebug
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

@app.route('/')
@app.route('/vectorize', methods=['POST'])
def vectorize():
    """
    Endpoint 1: POST /vectorize
    Dipanggil Laravel saat admin melakukan tambah/edit/hapus paket wisata.
    Melakukan ekstraksi teks, pembobotan TF-IDF baru, dan menyimpan ke database.
    """
    logger.info("Menerima permintaan /vectorize untuk melatih ulang model TF-IDF...")
    try:
        # 1. Ambil paket aktif dari database
        df_packages = get_active_packages()
        if df_packages.empty:
            logger.warning("Tidak ada paket wisata aktif yang ditemukan di database.")
            return jsonify({
                "status": "error",
                "message": "Tidak ada data paket wisata aktif untuk divektorisasi."
            }), 404
            
        # 2. Proses dan gabungkan fitur teks untuk setiap paket
        corpus = []
        package_ids = []
        for idx, row in df_packages.iterrows():
            combined_feat = build_combined_features(row)
            corpus.append(combined_feat)
            package_ids.append(int(row['id']))
            
        # 3. Latih TfidfVectorizer baru dan simpan model
        vectorizer, tfidf_matrix = fit_and_save_vectorizer(corpus)
        
        # 4. Hitung hash dari vocabulary untuk deteksi konsistensi
        vocab_json_str = json.dumps(vectorizer.vocabulary_, sort_keys=True)
        vocab_hash = hashlib.sha256(vocab_json_str.encode('utf-8')).hexdigest()
        
        # 5. Hapus semua vektor lama agar tidak tercampur dengan vektor baru yang berbeda dimensi
        clear_package_vectors()
        
        # 6. Konversi matriks TF-IDF sparse menjadi list rapat dan simpan ke database (Upsert)
        for i, pkg_id in enumerate(package_ids):
            # Ubah baris sparse matrix ke dense list
            vector_dense = tfidf_matrix[i].toarray()[0].tolist()
            combined_text = corpus[i]
            save_package_vector(pkg_id, combined_text, vector_dense, vocab_hash)
            
        # 6. Kirim respon sukses
        return jsonify({
            "status": "success",
            "message": "Proses vektorisasi dan pelatihan ulang TF-IDF selesai dengan sukses.",
            "total_paket": len(package_ids),
            "vocabulary_size": len(vectorizer.vocabulary_),
            "packages_vectorized": package_ids
        }), 200
        
    except Exception as e:
        logger.error(f"Error pada endpoint /vectorize: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Terjadi kesalahan sistem saat vektorisasi: {str(e)}"
        }), 500


@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Endpoint 2: POST /recommend
    Dipanggil Laravel saat wisatawan memasukkan preferensi pencarian.
    Menerima preference_id, menghitung Cosine Similarity, dan menyimpan hasil rekomendasi.
    """
    logger.info("Menerima permintaan rekomendasi paket wisata...")
    
    # 1. Validasi input request body
    data = request.get_json(silent=True) or {}
    preference_id = data.get('preference_id')
    
    if not preference_id:
        logger.warning("Permintaan ditolak: parameter 'preference_id' tidak ada dalam request.")
        return jsonify({
            "status": "error",
            "message": "Parameter 'preference_id' wajib diisi."
        }), 400
        
    try:
        # 2. Ambil preferensi pengguna berdasarkan ID
        preference = get_preference_by_id(preference_id)
        if not preference:
            logger.warning(f"Data preferensi dengan ID {preference_id} tidak ditemukan.")
            return jsonify({
                "status": "error",
                "message": f"Data preferensi dengan ID {preference_id} tidak ditemukan."
            }), 404
            
        # 3. Muat model TF-IDF dari penyimpanan PKL
        try:
            vectorizer = load_vectorizer()
        except FileNotFoundError as fnf_err:
            return jsonify({
                "status": "error",
                "message": f"Model belum diinisialisasi. Silakan jalankan /vectorize terlebih dahulu. Error: {str(fnf_err)}"
            }), 503
            
        # 4. Ambil semua vektor paket wisata yang ada di database
        df_vectors = get_all_vectors()
        if df_vectors.empty:
            logger.warning("Tabel vektor paket wisata kosong.")
            return jsonify({
                "status": "error",
                "message": "Proses vektorisasi paket belum pernah dijalankan. Jalankan /vectorize terlebih dahulu."
            }), 404
            
        # 5. Gabungkan fitur preferensi pengguna dan preprocess
        preference_text = build_preference_features(preference)
        logger.info(f"Teks preferensi terproses: '{preference_text}'")
        
        # 6. Transformasikan teks preferensi ke bentuk Vektor TF-IDF
        preference_vector = transform_preference(vectorizer, preference_text)
        
        # 7. Siapkan data vektor paket untuk kalkulasi
        package_ids = []
        package_vectors = []
        
        for idx, row in df_vectors.iterrows():
            vec_data = row['tfidf_vector']
            # Kadang SQLAlchemy mem-parsing JSON menjadi list Python secara otomatis, kadang string.
            if isinstance(vec_data, str):
                vec_list = json.loads(vec_data)
            else:
                vec_list = list(vec_data)
            
            package_vectors.append(vec_list)
            package_ids.append(int(row['package_id']))
            
        # 8. Hitung Cosine Similarity (Tahap 1: Pencarian Kemiripan Deskripsi)
        similarity_scores = calculate_similarity(preference_vector, package_vectors)
        
        # 9. Terapkan Penyaringan Mutlak (Tahap 2, 3) & Pencocokan (Tahap 4)
        df_active = get_active_packages()
        active_packages_dict = df_active.set_index('id').to_dict(orient='index')
        
        pref_budget = float(preference.get('budget') or 0)
        pref_category = str(preference.get('tour_category') or '').lower()
        pref_duration = str(preference.get('preferred_duration') or '').lower()
        pref_facilities = str(preference.get('preferred_facilities') or '').lower()
        
        filtered_packages = []
        for pkg_id, score in zip(package_ids, similarity_scores):
            if pkg_id not in active_packages_dict:
                continue
            pkg_info = active_packages_dict[pkg_id]
            
            # TAHAP 2: Penyaringan Budget (Buang jika over budget)
            pkg_price = float(pkg_info['pax1']) if pkg_info['pax1'] is not None else 0.0
            if pref_budget > 0 and pkg_price > pref_budget:
                continue
                
            # TAHAP 3: Penyaringan Kategori (Buang jika beda kategori)
            pkg_category = str(pkg_info.get('tour_category') or '').lower()
            if pref_category and pref_category != 'semua kategori' and pkg_category != pref_category:
                continue
                
            import re
            
            # TAHAP 4: Penyaringan Durasi
            pkg_duration = str(pkg_info.get('duration') or '').lower()
            if pref_duration and pref_duration != 'semua durasi' and pref_duration not in pkg_duration:
                continue
                
            # Jika lolos saringan mutlak (Budget, Kategori, Durasi), simpan kandidat
            filtered_packages.append({
                "package_id": pkg_id,
                "similarity_score": round(float(score), 4)
            })
            
        # 10. Pengurutan Akhir (Ranking) berdasarkan skor Cosine Similarity
        sorted_packages = sorted(filtered_packages, key=lambda x: x["similarity_score"], reverse=True)
        
        # Ambil Top-3
        top_recommendations = []
        for i, item in enumerate(sorted_packages[:3], start=1):
            item["rank"] = i
            top_recommendations.append(item)
        
        enriched_data = []
        results_list = []      # Untuk kolom 'results' JSON array
        scores_dict = {}       # Untuk kolom 'similarity_scores' JSON object
        
        for item in top_recommendations:
            pkg_id = item['package_id']
            score = item['similarity_score']
            rank = item['rank']
            
            results_list.append(pkg_id)
            scores_dict[str(pkg_id)] = score
            
            if pkg_id in active_packages_dict:
                pkg_info = active_packages_dict[pkg_id]
                enriched_data.append({
                    "rank": rank,
                    "package_id": pkg_id,
                    "name": pkg_info['name'],
                    "category": pkg_info['category_name'],
                    "pax1": float(pkg_info['pax1']) if pkg_info['pax1'] is not None else 0.0,
                    "duration": pkg_info['duration'],
                    "location": pkg_info['location'],
                    "similarity_score": score,
                    "slug": pkg_info['slug']
                })
                
        # 11. Simpan hasil rekomendasi kembali ke tabel MySQL 'recommendations'
        session_id = preference.get('session_id', 'unknown')
        save_recommendation_result(preference_id, session_id, results_list, scores_dict)
        
        # 12. Kirim respon hasil rekomendasi
        return jsonify({
            "status": "success",
            "preference": {
                "category": preference.get('tour_category') or "Semua Kategori",
                "budget": float(preference.get('budget') or 0),
                "duration": preference.get('preferred_duration') or "Semua Durasi",
                "facilities": preference.get('preferred_facilities') or "",
                "description": preference.get('description') or ""
            },
            "data": enriched_data
        }), 200
        
    except Exception as e:
        logger.error(f"Error pada endpoint /recommend: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Terjadi kesalahan sistem saat pemrosesan rekomendasi: {str(e)}"
        }), 500


@app.route('/health', methods=['GET'])
def health():
    """
    Endpoint 3: GET /health
    Memeriksa kesehatan sistem (health check) dan ketersediaan model TF-IDF.
    """
    try:
        model_exists = os.path.exists(MODEL_PATH)
        df_vectors = get_all_vectors()
        total_vectors = len(df_vectors)
        
        return jsonify({
            "status": "ok",
            "model_exists": model_exists,
            "total_vectors_in_db": total_vectors
        }), 200
    except Exception as e:
        logger.error(f"Error pada endpoint /health: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Layanan database atau sistem bermasalah: {str(e)}"
        }), 500


@app.route('/evaluate', methods=['POST'])
def evaluate():
    """
    Endpoint 4: POST /evaluate (Opsional / Tambahan)
    Mengukur performa rekomendasi (Precision, Recall, F1-Score) secara simulasi
    menggunakan minimal 3 skenario preferensi wisatawan yang berbeda.
    """
    logger.info("Memulai proses evaluasi performa sistem rekomendasi...")
    try:
        # 1. Ambil seluruh paket aktif dari database untuk dibandingkan
        df_packages = get_active_packages()
        if df_packages.empty:
            return jsonify({
                "status": "error",
                "message": "Tidak ada data paket wisata aktif untuk dievaluasi."
            }), 404
            
        # 2. Muat model TF-IDF
        try:
            vectorizer = load_vectorizer()
        except FileNotFoundError:
            return jsonify({
                "status": "error",
                "message": "Model belum dilatih. Lakukan /vectorize terlebih dahulu."
            }), 503
            
        # 3. Ambil seluruh vektor paket dari DB
        df_vectors = get_all_vectors()
        if df_vectors.empty:
            return jsonify({
                "status": "error",
                "message": "Tidak ada vektor paket wisata tersimpan."
            }), 404
            
        package_ids = []
        package_vectors = []
        for idx, row in df_vectors.iterrows():
            vec = row['tfidf_vector']
            vec_list = json.loads(vec) if isinstance(vec, str) else list(vec)
            package_vectors.append(vec_list)
            package_ids.append(int(row['package_id']))

        # 4. Definisikan minimal 3 skenario preferensi wisatawan sebagai data uji
        scenarios = [
            {
                "id": 1,
                "skenario": "Wisatawan Ekonomis Budget Terbatas",
                "tour_category": "Nature Trip",
                "budget": 300000.0,
                "preferred_duration": "7 Jam",
                "preferred_facilities": "jeep tiket masuk guide jeep",
                "description": "ingin melihat matahari terbit di bromo dengan budget hemat"
            },
            {
                "id": 2,
                "skenario": "Wisatawan Snorkeling Petualang",
                "tour_category": "Adventure Trip",
                "budget": 500000.0,
                "preferred_duration": "1 Day",
                "preferred_facilities": "snorkeling kapal konsumsi guide",
                "description": "mencari petualangan snorkeling dan menyeberang dengan kapal"
            },
            {
                "id": 3,
                "skenario": "Wisatawan Premium Tour Lengkap",
                "tour_category": "Culture Trip",
                "budget": 3000000.0,
                "preferred_duration": "2D1N",
                "preferred_facilities": "mobil driver bbm ijen guide air mineral senter",
                "description": "wisata sejarah dan budaya banyuwangi dengan mobil dan driver lengkap"
            }
        ]
        
        evaluation_results = []
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        
        # 5. Evaluasi masing-masing skenario
        for sc in scenarios:
            # A. Tentukan Himpunan Ground Truth Paket Wisata RELEVAN (R)
            # Aturan relevansi:
            # 1. tour_category paket SAMA dengan tour_category preferensi, ATAU
            # 2. harga paket (pax1) <= budget preferensi * 1.3 (toleransi 30% di atas budget)
            relevant_package_ids = set()
            for idx, pkg in df_packages.iterrows():
                pkg_id = int(pkg['id'])
                pkg_tour_cat = pkg.get('tour_category')
                pax1_val = pkg.get('pax1')
                pax1_float = float(pax1_val) if pd.notna(pax1_val) else 0.0
                
                # Cek kriteria relevansi
                match_category = (pkg_tour_cat is not None and pkg_tour_cat == sc['tour_category'])
                match_budget = (pax1_float <= sc['budget'] * 1.3)
                
                if match_category or match_budget:
                    relevant_package_ids.add(pkg_id)
            
            # B. Jalankan Rekomendasi TF-IDF + Cosine Similarity untuk Skenario Ini
            pref_text = build_preference_features(sc)
            pref_vector = transform_preference(vectorizer, pref_text)
            scores = calculate_similarity(pref_vector, package_vectors)
            top_rec = get_top_n(scores, package_ids, n=5)
            recommended_ids = [item['package_id'] for item in top_rec]
            
            # C. Hitung Metrik Kebenaran Evaluasi (TP, FP, FN)
            recommended_set = set(recommended_ids)
            
            # TP (True Positives): Paket direkomendasikan yang MEMANG RELEVAN
            tp = len(recommended_set.intersection(relevant_package_ids))
            
            # FP (False Positives): Paket direkomendasikan tapi TIDAK RELEVAN
            fp = len(recommended_set.difference(relevant_package_ids))
            
            # FN (False Negatives): Paket RELEVAN tapi TIDAK direkomendasikan
            fn = len(relevant_package_ids.difference(recommended_set))
            
            # D. Hitung Precision, Recall, dan F1-Score
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            total_precision += precision
            total_recall += recall
            total_f1 += f1
            
            evaluation_results.append({
                "skenario_id": sc['id'],
                "nama_skenario": sc['skenario'],
                "preferensi": {
                    "kategori": sc['tour_category'],
                    "budget": sc['budget'],
                    "fasilitas": sc['preferred_facilities'],
                    "deskripsi": sc['description']
                },
                "jumlah_relevan_aktual": len(relevant_package_ids),
                "paket_relevan_aktual": list(relevant_package_ids),
                "paket_direkomendasikan": recommended_ids,
                "metrik": {
                    "true_positives_tp": tp,
                    "false_positives_fp": fp,
                    "false_negatives_fn": fn,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1_score": round(f1, 4)
                }
            })
            
        # 6. Hitung rata-rata keseluruhan (Average Metrics)
        avg_precision = total_precision / len(scenarios)
        avg_recall = total_recall / len(scenarios)
        avg_f1 = total_f1 / len(scenarios)
        
        return jsonify({
            "status": "success",
            "message": "Proses evaluasi sistem rekomendasi selesai.",
            "metrik_rata_rata": {
                "average_precision": round(avg_precision, 4),
                "average_recall": round(avg_recall, 4),
                "average_f1_score": round(avg_f1, 4)
            },
            "detail_skenario": evaluation_results
        }), 200
        
    except Exception as e:
        logger.error(f"Error pada endpoint /evaluate: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Terjadi kesalahan sistem saat evaluasi: {str(e)}"
        }), 500


if __name__ == '__main__':
    # Menjalankan aplikasi Flask
    logger.info(f"Memulai server Flask di http://{FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)