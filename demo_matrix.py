import pandas as pd
import numpy as np
from preprocessor import build_preference_features, build_combined_features
from database import get_active_packages
from vectorizer import load_vectorizer

def generate_tfidf_matrix():
    print("Mengambil data paket dari database...")
    target_ids = [1, 3, 12, 7, 9]
    df_packages = get_active_packages()
    df_packages = df_packages[df_packages['id'].isin(target_ids)]
    
    # 1. Siapkan teks dokumen
    corpus = []
    doc_labels = []
    
    # Input Preferensi
    dummy_preference = {
        'tour_category': 'Nature Trip',
        'budget': 300000,
        'preferred_duration': '1 Hari',
        'preferred_facilities': 'Mobil, tiket masuk, guide',
        'description': 'saya ingin melihat sunrise di pagi hari'
    }
    pref_text = build_preference_features(dummy_preference)
    corpus.append(pref_text)
    doc_labels.append("Pref(Anda)")
    
    # Input Paket Wisata
    for idx, row in df_packages.iterrows():
        corpus.append(build_combined_features(row))
        # Singkat nama agar tabel tidak terlalu lebar
        short_name = row['name'].split()[0]
        if len(short_name) > 8:
            short_name = short_name[:8]
        doc_labels.append(f"P{row['id']}_{short_name}")
        
    # 2. Vektorisasi
    print("Memuat vectorizer dan mentransformasi teks...")
    vectorizer = load_vectorizer()
    tfidf_matrix = vectorizer.transform(corpus)
    
    # 3. Ambil daftar kata (fitur)
    feature_names = vectorizer.get_feature_names_out()
    
    # 4. Buat DataFrame
    # Baris (index) = feature_names
    # Kolom = doc_labels
    df_matrix = pd.DataFrame(
        tfidf_matrix.toarray().T, 
        index=feature_names, 
        columns=doc_labels
    )
    
    # Tampilkan hanya kata yang setidaknya muncul di salah satu dari 6 dokumen tersebut
    df_matrix = df_matrix[(df_matrix != 0).any(axis=1)]
    
    # Urutkan secara alfabetis berdasarkan kata
    df_matrix = df_matrix.sort_index()
    
    # 5. Simpan ke file teks agar rapi
    output_filename = "matrix_output.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("=== MATRIKS TF-IDF (KATA vs DOKUMEN) ===\n")
        f.write("Kolom: Dokumen (Pref = Preferensi User, P = Paket Wisata)\n")
        f.write("Baris: Kata (Term) yang diekstrak setelah preprocessing\n")
        f.write("Nilai: Bobot TF-IDF\n")
        f.write("=" * 85 + "\n\n")
        
        # Opsi pandas agar tidak terpotong saat dicetak
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        
        f.write(df_matrix.round(4).to_string())
        
    print(f"Selesai! Matriks berhasil disimpan ke {output_filename}")

if __name__ == "__main__":
    generate_tfidf_matrix()
