import json
import logging
from sqlalchemy import create_engine, text
import pandas as pd
from config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Membuat URL koneksi database MySQL menggunakan PyMySQL
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Membuat engine SQLAlchemy
engine = create_engine(DATABASE_URL, pool_recycle=3600)

def get_connection():
    """Mengembalikan koneksi langsung dari engine SQLAlchemy"""
    return engine.connect()

def get_active_packages():
    """
    Mengambil semua data paket wisata yang aktif (is_active = 1)
    dan melakukan JOIN dengan tabel categories serta package_types.
    
    Return: pandas DataFrame
    """
    query = """
        SELECT 
            tp.id,
            tp.package_name AS name,
            tp.package_name_en AS name_en,
            tp.slug,
            LEAST(
                COALESCE(NULLIF(tp.price_10pax, 0), 9999999999),
                COALESCE(NULLIF(tp.price_8pax, 0), 9999999999),
                COALESCE(NULLIF(tp.price_5pax, 0), 9999999999),
                COALESCE(NULLIF(tp.price_4pax, 0), 9999999999),
                COALESCE(NULLIF(tp.price_3pax, 0), 9999999999),
                COALESCE(NULLIF(tp.price_2pax, 0), 9999999999),
                COALESCE(NULLIF(tp.price_1pax, 0), 9999999999)
            ) AS pax1,                  -- Menggunakan harga termurah dari semua varian pax
            tp.duration,
            tp.tour_category,                        -- Kriteria Kategori Wisata (alam, budaya, dll.)
            tp.destination AS destinations,          -- Menggunakan destination riil AS destinations
            tp.description AS highlight,             -- Menggunakan description riil AS highlight (Deskripsi Paket)
            tp.facilities_included,
            tp.city AS location,                     -- Menggunakan city riil AS location
            tp.category_id,
            c.category_name,
            pt.type_name
        FROM tour_packages tp
        LEFT JOIN categories c ON tp.category_id = c.id
        LEFT JOIN package_types pt ON tp.package_type_id = pt.id
        WHERE tp.is_active = 1
    """
    try:
        with get_connection() as conn:
            df = pd.read_sql(query, conn)
            logger.info(f"Berhasil mengambil {len(df)} paket wisata aktif dari database.")
            return df
    except Exception as e:
        logger.error(f"Gagal mengambil paket wisata aktif: {str(e)}")
        raise e

def get_preference_by_id(preference_id):
    """
    Mengambil data preferensi pengguna berdasarkan ID.
    
    Return: dict satu baris atau None jika tidak ditemukan
    """
    query = text("""
        SELECT 
            up.id,
            up.session_id,
            up.category_id,
            up.tour_category,
            up.description,
            up.budget,
            up.preferred_duration,
            up.preferred_facilities,
            up.created_at,
            up.updated_at,
            c.category_name          -- JOIN ke kategori untuk melengkapi nama kategori
        FROM user_preferences up
        LEFT JOIN categories c ON up.category_id = c.id
        WHERE up.id = :preference_id
    """)
    try:
        with get_connection() as conn:
            result = conn.execute(query, {"preference_id": preference_id}).mappings().fetchone()
            if result:
                # Konversi hasil RowMapping menjadi dictionary biasa
                pref_dict = dict(result)
                # Konversi budget ke tipe float jika tidak None
                if pref_dict.get('budget') is not None:
                    pref_dict['budget'] = float(pref_dict['budget'])
                logger.info(f"Berhasil mengambil preferensi untuk ID: {preference_id}")
                return pref_dict
            else:
                logger.warning(f"Preferensi dengan ID: {preference_id} tidak ditemukan.")
                return None
    except Exception as e:
        logger.error(f"Gagal mengambil preferensi dengan ID {preference_id}: {str(e)}")
        raise e

def get_all_vectors():
    """
    Mengambil semua data package_id, combined_features, dan tfidf_vector
    dari tabel package_vectors.
    
    Return: pandas DataFrame
    """
    query = "SELECT package_id, combined_features, tfidf_vector FROM package_vectors"
    try:
        with get_connection() as conn:
            df = pd.read_sql(query, conn)
            logger.info(f"Berhasil mengambil {len(df)} data vektor dari database.")
            return df
    except Exception as e:
        logger.error(f"Gagal mengambil data vektor paket wisata: {str(e)}")
        raise e

def clear_package_vectors():
    """Menghapus semua data di tabel package_vectors sebelum dilakukan vektorisasi ulang."""
    query = text("DELETE FROM package_vectors")
    try:
        with get_connection() as conn:
            conn.execute(query)
            conn.commit()
        logger.info("Berhasil mengosongkan tabel package_vectors.")
    except Exception as e:
        logger.error(f"Gagal mengosongkan tabel package_vectors: {str(e)}")
        raise e

def save_package_vector(package_id, combined_features, tfidf_vector_list, vocabulary_hash=None):
    """
    Menyimpan atau mengupdate vektor TF-IDF dari paket wisata ke database (UPSERT).
    """
    query = text("""
        INSERT INTO package_vectors (package_id, combined_features, tfidf_vector, vocabulary_hash)
        VALUES (:package_id, :combined_features, :tfidf_vector, :vocabulary_hash)
        ON DUPLICATE KEY UPDATE
            combined_features = VALUES(combined_features),
            tfidf_vector = VALUES(tfidf_vector),
            vocabulary_hash = VALUES(vocabulary_hash),
            last_updated = CURRENT_TIMESTAMP
    """)
    try:
        tfidf_json_str = json.dumps(tfidf_vector_list)
        with get_connection() as conn:
            conn.execute(query, {
                "package_id": package_id,
                "combined_features": combined_features,
                "tfidf_vector": tfidf_json_str,
                "vocabulary_hash": vocabulary_hash
            })
            conn.commit()
        logger.info(f"Berhasil menyimpan/mengupdate vektor paket ID: {package_id}")
    except Exception as e:
        logger.error(f"Gagal menyimpan vektor paket ID {package_id}: {str(e)}")
        raise e

def save_recommendation_result(preference_id, session_id, results_list, scores_list):
    """
    Menyimpan hasil rekomendasi ke tabel recommendations.
    Jika preference_id sudah ada di tabel, lakukan UPDATE pada kolom results & similarity_scores.
    Jika belum ada record, lakukan INSERT.
    """
    # 1. Cek apakah record sudah ada
    check_query = text("SELECT id FROM recommendations WHERE preference_id = :preference_id")
    
    results_json = json.dumps(results_list)
    scores_json = json.dumps(scores_list)
    
    try:
        with get_connection() as conn:
            row = conn.execute(check_query, {"preference_id": preference_id}).fetchone()
            
            if row:
                # Lakukan UPDATE
                update_query = text("""
                    UPDATE recommendations
                    SET 
                        results = :results,
                        similarity_scores = :scores,
                        session_id = :session_id,
                        updated_at = NOW()
                    WHERE preference_id = :preference_id
                """)
                conn.execute(update_query, {
                    "preference_id": preference_id,
                    "session_id": session_id,
                    "results": results_json,
                    "scores": scores_json
                })
                logger.info(f"Berhasil memperbarui hasil rekomendasi untuk preference_id: {preference_id}")
            else:
                # Lakukan INSERT
                insert_query = text("""
                    INSERT INTO recommendations (preference_id, session_id, results, similarity_scores, created_at, updated_at)
                    VALUES (:preference_id, :session_id, :results, :scores, NOW(), NOW())
                """)
                conn.execute(insert_query, {
                    "preference_id": preference_id,
                    "session_id": session_id,
                    "results": results_json,
                    "scores": scores_json
                })
                logger.info(f"Berhasil membuat/menyisipkan hasil rekomendasi baru untuk preference_id: {preference_id}")
            
            conn.commit()
    except Exception as e:
        logger.error(f"Gagal menyimpan hasil rekomendasi untuk preference_id {preference_id}: {str(e)}")
        raise e
