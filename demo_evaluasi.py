import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from preprocessor import build_preference_features, build_combined_features
from database import get_active_packages
from vectorizer import load_vectorizer
from sklearn.metrics.pairwise import cosine_similarity

def run_evaluation():
    # 1. Mengambil data paket dari database
    df_packages = get_active_packages()
    
    # ID Paket yang tersedia:
    # 1: Kawah Ijen Banyuwangi (Blue Fire)
    # 7: Baluran (Afrika Van Java)
    # 4: Djawatan (Lord Of The Rings)
    # 22: Hutan Djawatan, Taman Nasional Baluran, Kawah Ijen, Kawah Wurung
    # 10: Kawah Wurung & Djawatan
    package_ids = df_packages['id'].tolist()
    
    # Mengambil teks fitur untuk paket (seperti yang dilakukan server)
    corpus = [build_combined_features(row) for _, row in df_packages.iterrows()]
    
    # Memuat vectorizer yang sudah dilatih
    vectorizer = load_vectorizer()
    pkg_matrix = vectorizer.transform(corpus).toarray()
    
    # 2. Definisikan 5 Skenario dan Ground Truth (ID Paket Relevan)
    scenarios = [
        {
            "nama": "Sunrise Pagi Hari",
            "kategori": "Nature Trip",
            "budget": 300000,
            "durasi": "1 Hari",
            "fasilitas": "Mobil, tiket masuk, guide",
            "deskripsi": "saya ingin melihat sunrise di pagi hari",
            # Ground truth: Kawah Ijen & Kawah Wurung (keduanya memiliki sunrise)
            "ground_truth_ids": [1, 9] 
        },
        {
            "nama": "Savana Ala Afrika",
            "kategori": "Nature Trip",
            "budget": 350000,
            "durasi": "1 Hari",
            "fasilitas": "jeep, tiket masuk",
            "deskripsi": "melihat padang savana luas seperti afrika",
            # Ground truth: Baluran & Kawah Wurung & Djawatan (memiliki savana)
            "ground_truth_ids": [7, 10, 22] 
        },
        {
            "nama": "Hutan Trembesi Magis",
            "kategori": "Nature Trip",
            "budget": 350000,
            "durasi": "1 Hari",
            "fasilitas": "Mobil, tiket masuk, guide",
            "deskripsi": "berjalan di bawah pohon trembesi raksasa dengan suasana magis",
            # Ground truth: Djawatan utama & alternatifnya
            "ground_truth_ids": [4, 12] 
        },
        {
            "nama": "Padang Savana Bukit Hijau",
            "kategori": "Nature Trip",
            "budget": 650000,
            "durasi": "1 Hari",
            "fasilitas": "Mobil, tiket masuk",
            "deskripsi": "melihat hamparan padang savana luas dengan bukit hijau dan udara sejuk",
            # Ground truth: Kawah Wurung & Kawah Ijen
            "ground_truth_ids": [10, 1] 
        },
        {
            "nama": "Eksplorasi Lengkap 2 Hari",
            "kategori": "Nature Trip",
            "budget": 3500000,
            "durasi": "2D1N",
            "fasilitas": "hiace, driver, bbm, tiket",
            "deskripsi": "liburan panjang mengunjungi hutan djawatan, taman nasional baluran, kawah ijen, dan kawah wurung",
            # Ground truth: Paket ID 22 & Paket 2 Hari lainnya (Baluran ID 7)
            "ground_truth_ids": [22, 7] 
        }
    ]
    
    results = []
    
    # Threshold Top-N (Sistem mengembalikan 5 teratas sebagai "Relevan" oleh sistem)
    N = 5
    
    print("=" * 80)
    print("PROSES EVALUASI SKENARIO (TANPA AKURASI)")
    print("=" * 80)
    
    for sc in scenarios:
        # Proses preferensi
        pref_dict = {
            'tour_category': sc['kategori'],
            'budget': sc['budget'],
            'preferred_duration': sc['durasi'],
            'preferred_facilities': sc['fasilitas'],
            'description': sc['deskripsi']
        }
        pref_text = build_preference_features(pref_dict)
        pref_vector = vectorizer.transform([pref_text]).toarray()[0]
        
        # Hitung Cosine Similarity
        similarity_scores = cosine_similarity(pref_vector.reshape(1, -1), pkg_matrix)[0]
        
        # Urutkan dan ambil Top N
        pkg_scores = [(package_ids[i], similarity_scores[i]) for i in range(len(package_ids))]
        pkg_scores.sort(key=lambda x: x[1], reverse=True)
        top_n_ids = [p[0] for p in pkg_scores[:N]]
        
        # Buat array Biner untuk evaluasi (1 = Relevan, 0 = Tidak Relevan)
        y_true = [1 if pid in sc['ground_truth_ids'] else 0 for pid in package_ids]
        y_pred = [1 if pid in top_n_ids else 0 for pid in package_ids]
        
        # Hitung True Positive, dll (Manual untuk log)
        TP = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        FP = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        FN = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
        
        # Hitung Metrics dengan sklearn
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        results.append({
            "Query/Skenario": sc['nama'],
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        })
        
        print(f"Skenario: {sc['nama']}")
        print(f"Top-{N} Rekomendasi Sistem: {top_n_ids}")
        print(f"Ground Truth Relevan     : {sc['ground_truth_ids']}")
        print(f"TP: {TP}, FP: {FP}, FN: {FN}")
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}\n")
        
    # Buat DataFrame
    df_results = pd.DataFrame(results)
    
    # Hitung Baris Total dan Rata-rata
    total_precision = df_results["Precision"].sum()
    total_recall = df_results["Recall"].sum()
    total_f1 = df_results["F1-Score"].sum()
    
    avg_precision = df_results["Precision"].mean() * 100
    avg_recall = df_results["Recall"].mean() * 100
    avg_f1 = df_results["F1-Score"].mean() * 100
    
    df_results.loc[len(df_results)] = ["Total", total_precision, total_recall, total_f1]
    df_results.loc[len(df_results)] = ["Rata-rata (%)", avg_precision, avg_recall, avg_f1]
    
    # 3. Simpan ke Teks File
    with open("evaluasi_output.txt", "w", encoding="utf-8") as f:
        f.write("=== DAFTAR REFERENSI PAKET WISATA ===\n")
        f.write("ID 1 : Kawah Ijen Banyuwangi (Blue Fire)\n")
        f.write("ID 7 : Baluran (Afrika Van Java)\n")
        f.write("ID 4 : Djawatan (Lord Of The Rings)\n")
        f.write("ID 22: Hutan Djawatan, Taman Nasional Baluran, Kawah Ijen, Kawah Wurung\n")
        f.write("ID 10: Kawah Wurung & Djawatan\n")
        f.write("=" * 80 + "\n\n")

        f.write("=== TABEL EVALUASI METRIK ===\n\n")
        # Format angka agar seragam
        formatted_df = df_results.copy()
        for col in ["Precision", "Recall", "F1-Score"]:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.2f}")
        
        f.write(formatted_df.to_string(index=False))
        
        # Narasi Kesimpulan (TANPA AKURASI)
        f.write("\n\n\n=== DRAFT NARASI KESIMPULAN ===\n")
        narasi = (f"Dari pengujian terhadap 5 skenario preferensi, dihasilkan matriks evaluasi "
                  f"yang cukup bervariasi dan realistis. Sistem mencatat skor rata-rata Precision sebesar {avg_precision:.2f}%, "
                  f"Recall sebesar {avg_recall:.2f}%, dan F1-Score {avg_f1:.2f}%. "
                  f"Fluktuasi nilai metrik (khususnya Recall yang tidak selalu 100%) membuktikan "
                  f"bahwa performa rekomendasi algoritma TF-IDF sangat bergantung pada seberapa presisi kata kunci "
                  f"input wisatawan beririsan dengan ketersediaan paket-paket referensi di dalam database.")
        f.write(narasi)
        
    # 4. Generate Grafik Bar Menggunakan Matplotlib (TANPA AKURASI)
    # Ambil hanya data skenario (tanpa baris Total & Rata-rata)
    df_plot = df_results.iloc[:-2].copy()
    
    x = np.arange(len(df_plot))
    width = 0.25 # Lebar bar dibesarkan sedikit karena jumlah metrik berkurang jadi 3
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars (Untuk persentase kita kalikan 100 agar range 0-100)
    rects1 = ax.bar(x - width, df_plot["Precision"] * 100, width, label='Precision (%)', color='#36a2eb')
    rects2 = ax.bar(x, df_plot["Recall"] * 100, width, label='Recall (%)', color='#ffcd56')
    rects3 = ax.bar(x + width, df_plot["F1-Score"] * 100, width, label='F1-Score (%)', color='#ff6384')
    
    # Tambahkan Label
    ax.set_ylabel('Persentase Skor (%)')
    ax.set_title('Hasil Evaluasi Metrik Rekomendasi (Precision, Recall, F1-Score)')
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["Query/Skenario"])
    ax.legend()
    ax.set_ylim([0, 110]) # Range Y dari 0 s/d 110 (agar ada ruang untuk legenda)
    
    # Label nilai di atas bar
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.0f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
                        
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    fig.tight_layout()
    plt.savefig('grafik_evaluasi.png', dpi=300)
    
    print("\nFile 'evaluasi_output.txt' dan 'grafik_evaluasi.png' berhasil diperbarui (tanpa Akurasi)!")

if __name__ == "__main__":
    run_evaluation()
