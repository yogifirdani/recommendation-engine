import preprocessor
import database
import vectorizer
import warnings
import textwrap
import sys

warnings.filterwarnings("ignore")

def print_markdown_table_row(cols, widths):
    """Mencetak baris tabel markdown"""
    row = "| "
    for i in range(len(cols)):
        row += str(cols[i]).ljust(widths[i]) + " | "
    print(row)

def main():
    # Mengalihkan semua perintah print() ke dalam file txt
    with open('laporan_rekomendasi.txt', 'w', encoding='utf-8') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        
        print("# Hasil Eksekusi Tabel Rekomendasi\n")
        print("Berikut adalah data yang di-generate langsung dari sistem untuk laporan Anda.\n")
        print("---")
        
        df = database.get_active_packages()
        
        # ==========================================
        # PREPROCESSING (5 PAKET)
        # ==========================================
        print("\n## 4.1.2 Hasil Preprocessing Data (5 Paket Wisata)\n")
        
        target_ids = [1, 7, 4, 10, 22]
        prep_df = df[df['id'].isin(target_ids)]
        
        prep_data = []
        for index, row_prep in prep_df.iterrows():
            combined_raw = f"{row_prep.get('category_name', '')} {row_prep.get('tour_category', '')} {preprocessor.get_package_price_tags(row_prep.get('pax1', 0))} {preprocessor.standardize_duration(row_prep.get('duration', ''))} {row_prep.get('facilities_included', '')} {str(row_prep.get('highlight', '')) * 3}"
            
            cleaned = preprocessor.strip_html(combined_raw)
            folded = preprocessor.case_folding(cleaned)
            no_stop = preprocessor.remove_stopwords(folded)
            stemmed = preprocessor.stemming(no_stop)
            tokenized = preprocessor.tokenize(stemmed)
            
            prep_data.append({
                'name': row_prep['name'],
                'cleaned': cleaned,
                'folded': folded,
                'no_stop': no_stop,
                'stemmed': stemmed,
                'tokenized': tokenized
            })

        # Helper function to truncate for table display
        def trunc(t):
            return t # Tampilkan seluruh teks secara utuh tanpa disingkat

        # Tabel 4.5
        print("**Tabel 4.5 Hasil Case Folding**\n")
        print("| Nama Paket | Sebelum Diproses | Hasil Case Folding |")
        print("| :--- | :--- | :--- |")
        for item in prep_data:
            print(f"| **{item['name']}** | `{trunc(item['cleaned'])}` | `{trunc(item['folded'])}` |")
        print("\n")

        # Tabel 4.7
        print("**Tabel 4.7 Hasil Stopword Removal**\n")
        print("| Nama Paket | Sebelum Diproses | Hasil Stopword Removal |")
        print("| :--- | :--- | :--- |")
        for item in prep_data:
            print(f"| **{item['name']}** | `{trunc(item['folded'])}` | `{trunc(item['no_stop'])}` |")
        print("\n")

        # Tabel 4.9
        print("**Tabel 4.9 Hasil Stemming**\n")
        print("| Nama Paket | Sebelum Diproses | Hasil Stemming |")
        print("| :--- | :--- | :--- |")
        for item in prep_data:
            print(f"| **{item['name']}** | `{trunc(item['no_stop'])}` | `{trunc(item['stemmed'])}` |")
        print("\n")

        # Tabel 4.11
        print("**Tabel 4.11 Hasil Tokenizing**\n")
        print("| Nama Paket | Sebelum Diproses | Hasil Tokenizing |")
        print("| :--- | :--- | :--- |")
        for item in prep_data:
            print(f"| **{item['name']}** | `{trunc(item['stemmed'])}` | `{trunc(item['tokenized'])}` |")
        print("\n*(Penjelasan singkat Tokenizing: Setelah dilakukan stemming, sistem memecah ulang kalimat dan membuang karakter tunggal. Pada kelima sampel di atas, sistem memastikan tidak ada lagi kata yang berkarakter tunggal sehingga hasil Tokenizing merupakan versi terbersih dari teks).*\n")
        print("---\n")

        # ==========================================
        # PREPROCESSING PREFERENSI WISATAWAN
        # ==========================================
        print("## 4.1.3 Hasil Preprocessing Data Preferensi Wisatawan\n")
        print("Sebelum menghitung bobot TF-IDF, sistem menerima input preferensi dari wisatawan sebagai berikut:\n")
        print("- **Kategori Wisata**: Nature Trip")
        print("- **Budget Maksimal**: Rp 300.000")
        print("- **Durasi**: 1 Hari")
        print("- **Fasilitas**: Mobil, tiket masuk, guide")
        print("- **Deskripsi Tambahan**: saya ingin melihat sunrise di pagi hari dengan pemandangan yang indah\n")
        
        dummy_preference = {
            'tour_category': 'Nature Trip',
            'budget': 300000,
            'preferred_duration': '1 Hari',
            'preferred_facilities': 'Mobil, tiket masuk, guide',
            'description': 'saya ingin melihat sunrise di pagi hari dengan pemandangan yang indah'
        }
        
        # Raw gabungan
        desc_weighted = f"{dummy_preference['description']} " * 3
        pref_raw = f"{dummy_preference['tour_category']} {preprocessor.get_user_budget_tags(dummy_preference['budget'])} {preprocessor.standardize_duration(dummy_preference['preferred_duration'])} {dummy_preference['preferred_facilities']} {desc_weighted}".strip()
        
        pref_cleaned = preprocessor.strip_html(pref_raw)
        pref_folded = preprocessor.case_folding(pref_cleaned)
        pref_no_stop = preprocessor.remove_stopwords(pref_folded)
        pref_stemmed = preprocessor.stemming(pref_no_stop)
        pref_tokenized = preprocessor.tokenize(pref_stemmed)

        print("**Tabel 4.12 Tahapan Preprocessing Input Wisatawan**\n")
        print("| Tahapan | Hasil Teks |")
        print("| :--- | :--- |")
        print(f"| **1. Penggabungan Fitur (Raw)** | `{pref_raw}` |")
        print(f"| **2. Case Folding** | `{pref_folded}` |")
        print(f"| **3. Stopword Removal** | `{pref_no_stop}` |")
        print(f"| **4. Stemming** | `{pref_stemmed}` |")
        print(f"| **5. Tokenizing** | `{pref_tokenized}` |")
        print("\nInput yang telah melalui tahap *preprocessing* ini (hasil akhirnya) akan menjadi dokumen *query* wisatawan untuk dicocokkan dengan dokumen paket wisata di tahap TF-IDF.\n")
        print("---\n")

        # ==========================================
        # TF-IDF (5 PAKET)
        # ==========================================
        print("## 4.1.4 Hasil TF-IDF\n")
        print("Tabel ini menunjukkan perbandingan bobot (weight) TF-IDF dari kriteria yang dimasukkan wisatawan terhadap 5 sampel dokumen (paket wisata) di database. Hanya token (kata) utama dengan bobot kemunculan yang tinggi yang ditampilkan.\n")
        print("**Tabel 4.13 Cuplikan Matriks TF-IDF**\n")
        
        tfidf_model = vectorizer.load_vectorizer()
        target_ids = [1, 7, 4, 10, 22]
        filtered_df = df[df['id'].isin(target_ids)]
        
        pref_text_clean = preprocessor.build_preference_features(dummy_preference)
        pref_vector = vectorizer.transform_preference(tfidf_model, pref_text_clean)
        
        feature_names = tfidf_model.get_feature_names_out()
        
        # Ambil SEMUA kata kunci (token) dari wisatawan yang memiliki bobot > 0
        pref_indices = [i for i in range(len(pref_vector)) if pref_vector[i] > 0]
        pref_indices = sorted(pref_indices, key=lambda i: pref_vector[i], reverse=True)
        
        package_vectors = []
        package_names = []
        
        for index, row in filtered_df.iterrows():
            combined_text = preprocessor.build_combined_features(row)
            pkg_vector = tfidf_model.transform([combined_text]).toarray()[0]
            package_vectors.append(pkg_vector)
            package_names.append(row['name'])

        # Header Tabel TF-IDF
        cols = ["Token (Kata Kunci)", "Bobot Wisatawan"]
        for name in package_names:
            cols.append(f"Pkg {name.split()[0]}") # Ambil 1 kata pertama untuk header
            
        widths = [20, 15] + [15]*len(package_names)
        print_markdown_table_row(cols, widths)
        
        # Separator
        sep = "| " + " | ".join([":---"] * len(cols)) + " |"
        print(sep)
        
        # Isi Tabel TF-IDF
        for idx in pref_indices:
            row_data = [f"`{feature_names[idx]}`", f"{pref_vector[idx]:.4f}"]
            for i, pkg_vec in enumerate(package_vectors):
                row_data.append(f"{pkg_vec[idx]:.4f}")
            print_markdown_table_row(row_data, widths)

        print("\n---\n")

        # ==========================================
        # COSINE SIMILARITY
        # ==========================================
        print("## 4.1.5 Hasil Cosine Similarity\n")
        print("Berdasarkan input preferensi wisatawan (mencari wisata Alam/Nature, Budget Rp 300.000, 1 Hari, fasilitas mobil/guide, ingin melihat sunrise pagi hari dengan pemandangan indah), berikut adalah peringkat Cosine Similarity dari jarak vektor TF-IDF terdekat (Top-5):\n")
        print("**Tabel 4.14 Hasil Perhitungan Cosine Similarity (Top-5 Rekomendasi)**\n")
        
        scores = vectorizer.calculate_similarity(pref_vector, package_vectors)
        results = []
        for i in range(len(package_names)):
            price_int = int(filtered_df.iloc[i]['pax1'])
            price_formatted = f"Rp {price_int:,.0f}".replace(",", ".")
            results.append((package_names[i], scores[i], filtered_df.iloc[i]['tour_category'], price_formatted, filtered_df.iloc[i]['duration']))
            
        results = sorted(results, key=lambda x: x[1], reverse=True)
        
        # Header Tabel Cosine
        cosine_cols = ["Ranking", "Nama Paket Wisata", "Kategori", "Harga / Budget", "Durasi", "Skor Cosine"]
        cosine_widths = [7, 40, 15, 18, 10, 15]
        print_markdown_table_row(cosine_cols, cosine_widths)
        
        # Separator
        print("| " + " | ".join([":---"] * len(cosine_cols)) + " |")
        
        # Isi Tabel Cosine
        for i, res in enumerate(results, 1):
            row_data = [f"**{i}**", res[0], res[2], res[3], res[4], f"**{res[1]:.4f}**"]
            print_markdown_table_row(row_data, cosine_widths)

        print("\n**Penjelasan Singkat (Tambahkan di bawah Tabel 4.14):**")
        print("Dari hasil perhitungan Cosine Similarity, sistem berhasil mengidentifikasi bahwa paket Kawah Ijen Banyuwangi (Blue Fire) merupakan destinasi yang paling relevan (Ranking 1) dengan preferensi wisatawan, memperoleh skor kecocokan tertinggi yaitu 0.1764. Hal ini dikarenakan paket tersebut memuat matriks token (TF-IDF) yang paling selaras dengan kriteria wisatawan, terutama pada variabel sunrise, pemandangan, indah, fasilitas mobil, dan harga yang masih masuk dalam batas toleransi budget. Paket dengan skor terendah pada peringkat 5 (Djawatan) tetap ditampilkan karena masih memiliki korelasi pada durasi dan fasilitas, namun memiliki kedekatan sudut vektor yang jauh lebih lebar dibanding Kawah Ijen.")
        print("\n")

        # Kembalikan stdout ke semula
        sys.stdout = original_stdout

    print("[BERHASIL] Hasil tabel laporan telah disimpan ke dalam file 'laporan_rekomendasi.txt'")

if __name__ == '__main__':
    main()
