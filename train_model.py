import json
import hashlib
from database import get_active_packages, save_package_vector
from preprocessor import build_combined_features
from vectorizer import fit_and_save_vectorizer

def train():
    """
    Fungsi utama untuk melatih model TF-IDF dan menyimpan hasilnya ke database.
    Ini adalah "Dapur Pembelajaran" (Training Phase) dari sistem rekomendasi.
    """
    
    # 1. MENGAMBIL DATA
    # Mengambil semua data paket wisata yang berstatus aktif dari database MySQL
    df_packages = get_active_packages()
    
    corpus = []        # Array untuk menampung seluruh teks gabungan dari semua paket
    package_ids = []   # Array untuk menyimpan ID paket agar pasangannya tidak tertukar
    
    # 2. TAHAP PREPROCESSING
    # Looping (mengulang) satu per satu setiap paket wisata yang diambil dari DB
    for idx, row in df_packages.iterrows():
        # Membersihkan teks dan menggabungkan kriteria (Case Folding, Stemming, Tokenizing, dll)
        combined_feat = build_combined_features(row)
        
        # Memasukkan teks yang sudah bersih ke dalam wadah (corpus)
        corpus.append(combined_feat)
        
        # Menyimpan ID paket yang bersangkutan
        package_ids.append(int(row['id']))
        
    # 3. MELATIH ALGORITMA TF-IDF (TRAINING)
    # Menyuapkan seluruh 'corpus' ke algoritma TF-IDF.
    # Algoritma akan menghitung bobot setiap kata (TF dan IDF), mengubah teks menjadi matriks angka,
    # dan membekukan/menyimpan rumusnya menjadi file .pkl (Pickle).
    vectorizer, tfidf_matrix = fit_and_save_vectorizer(corpus)
    
    # 4. MEMBUAT KODE UNIK (HASH) KAMUS KOSA KATA
    # Berfungsi sebagai penanda/versi kamus, agar sistem tahu jika sewaktu-waktu ada penambahan kata baru.
    vocab_json_str = json.dumps(vectorizer.vocabulary_, sort_keys=True)
    vocab_hash = hashlib.sha256(vocab_json_str.encode('utf-8')).hexdigest()
    
    # 5. MENYIMPAN HASIL KE DATABASE
    # Membedah matriks raksasa hasil TF-IDF menjadi potongan per-paket wisata
    for i, pkg_id in enumerate(package_ids):
        # Mengubah format matriks menjadi deretan angka biasa (dense list) agar muat disimpan di MySQL
        vector_dense = tfidf_matrix[i].toarray()[0].tolist()
        
        # Teks gabungan yang sudah bersih
        combined_text = corpus[i]
        
        # Menyimpan semuanya ke dalam tabel `package_vectors` (UPSERT)
        save_package_vector(pkg_id, combined_text, vector_dense, vocab_hash)
        
    print("Training Berhasil! Model AI sudah selesai belajar dari data terbaru.")

if __name__ == "__main__":
    # Baris ini memastikan bahwa fungsi train() langsung dijalankan
    # apabila kita mengetik perintah `python train_model.py` di terminal.
    train()
