import preprocessor
import database
import vectorizer
import warnings

# Mengabaikan warning agar output rapi
warnings.filterwarnings("ignore")

def write_cosine_table(preference_text, package_names, scores, output_file="cosine_output.txt"):
    """
    Fungsi untuk menulis tabel hasil Cosine Similarity ke dalam file teks.
    """
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, output_file)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n" + "="*80 + "\n")
        f.write("=== PERHITUNGAN COSINE SIMILARITY ===\n")
        f.write("="*80 + "\n")
        f.write(f"KEINGINAN (PREFERENSI) WISATAWAN:\n{preference_text}\n\n")
        
        # Gabungkan nama paket dengan skornya
        results = []
        for i in range(len(package_names)):
            results.append((package_names[i], scores[i]))
            
        # Urutkan berdasarkan skor tertinggi (Ranking)
        results = sorted(results, key=lambda x: x[1], reverse=True)
        
        col1_width = 10
        col2_width = 45
        col3_width = 15
        border = f"+-{'-' * col1_width}-+-{'-' * col2_width}-+-{'-' * col3_width}-+\n"
        
        f.write(border)
        f.write(f"| {'RANKING'.ljust(col1_width)} | {'NAMA PAKET WISATA'.ljust(col2_width)} | {'SKOR COSINE'.ljust(col3_width)} |\n")
        f.write(border)
        
        for i, (name, score) in enumerate(results, 1):
            f.write(f"| {str(i).ljust(col1_width)} | {name.ljust(col2_width)} | {f'{score:.4f}'.ljust(col3_width)} |\n")
        
        f.write(border)
        f.write("\n* Catatan: Skor semakin mendekati 1.0000 berarti semakin cocok.\n\n")
        
    print(f"Berhasil menyimpan hasil Cosine Similarity ke dalam file: {output_file}")

def run_demo_cosine():
    # 1. Kita buat data seolah-olah wisatawan sedang mencari paket di website
    # Wisatawan ingin: "Melihat blue fire di kawah ijen, dan menikmati pemandangan savana baluran"
    # Budget: Rp 3.000.000 (akan menghasilkan budgetbracket)
    # Durasi: "3 Hari" (akan menjadi durasi3d2n)
    
    dummy_preference = {
    'tour_category': 'Nature Trip',  # <-- HANYA INI YANG DIKIRIM OLEH WEBSITE KE DB
    'budget': 300000,
    'preferred_duration': '1 Hari',
    'preferred_facilities': 'Mobil, tiket masuk, guide',
    'description': 'saya ingin melihat sunrise di pagi hari'
}

    
    # 2. Proses teks preferensi wisatawan menggunakan fungsi yang sama di preprocessor.py
    pref_text_clean = preprocessor.build_preference_features(dummy_preference)
    
    # 3. Muat "Otak" AI (Model TF-IDF)
    print("Memuat model TF-IDF dari sistem Anda...")
    try:
        tfidf_model = vectorizer.load_vectorizer()
    except Exception as e:
        print("Gagal memuat model. Pastikan Anda sudah menjalankan train_model.py!")
        return
        
    # 4. Ubah preferensi wisatawan jadi angka (Vektor)
    pref_vector = vectorizer.transform_preference(tfidf_model, pref_text_clean)
    
    # 5. Mengambil 3 paket (Kawah Ijen, Tabuhan, Djawatan/Pulau Merah) dari database
    print("Mengambil data vektor 3 paket dari database...")
    df_packages = database.get_active_packages()
    target_ids = [1, 3, 12, 7, 9]
    filtered_df = df_packages[df_packages['id'].isin(target_ids)]
    
    # Kumpulkan vektor paket-paket tersebut
    package_vectors = []
    package_names = []
    
    for index, row in filtered_df.iterrows():
        # Untuk demo, kita transform langsung agar cepat tanpa harus query tabel package_vectors
        combined_text = preprocessor.build_combined_features(row)
        pkg_vector = tfidf_model.transform([combined_text]).toarray()[0]
        
        package_vectors.append(pkg_vector)
        package_names.append(row['name'])
        
    # 6. Hitung Cosine Similarity menggunakan fungsi yang ada di vectorizer.py
    # preference_vector vs daftar_vektor_paket
    scores = vectorizer.calculate_similarity(pref_vector, package_vectors)
    
    # 7. Tampilkan tabel hasilnya
    raw_preference_str = f"Kategori: {dummy_preference['tour_category']}, Budget: Rp {dummy_preference['budget']}\n"
    raw_preference_str += f"Durasi: {dummy_preference['preferred_duration']}, Fasilitas: {dummy_preference['preferred_facilities']}\n"
    raw_preference_str += f"Deskripsi: {dummy_preference['description']}"
    
    write_cosine_table(raw_preference_str, package_names, scores)

if __name__ == "__main__":
    run_demo_cosine()
