import preprocessor
import database
import vectorizer
import warnings
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore")

def run_demo_breakdown():
    print("=== PERSIAPAN DATA ===")
    df = database.get_active_packages()
    target_ids = [1, 3, 12, 7, 9]
    filtered_df = df[df['id'].isin(target_ids)]
    
    # MUAT MODEL SISTEM
    tfidf_model = vectorizer.load_vectorizer()
    feature_names = tfidf_model.get_feature_names_out()
    
    for _, row in filtered_df.iterrows():
        pkg_id = row['id']
        pkg_name = row['name']
        combined_text = preprocessor.build_combined_features(row)
        
        print(f"\n{'='*80}")
        print(f"ANALISIS PAKET ID: {pkg_id} ({pkg_name})")
        print(f"{'='*80}")
        print(f"Teks Setelah Preprocessing:\n{combined_text[:150]}...\n")
        
        # ---------------------------------------------------------
        # PROSES 1 & 2: MENGHITUNG TF LOKAL DAN IDF GLOBAL
        # ---------------------------------------------------------
        tf_vectorizer = CountVectorizer(vocabulary=feature_names, ngram_range=(1, 2))
        tf_matrix = tf_vectorizer.transform([combined_text])
        tf_array = tf_matrix.toarray()[0]
        
        # 1. Siapkan daftar TF (Kemunculan Terbanyak)
        words_by_tf = [(feature_names[i], tf_array[i]) for i in range(len(tf_array)) if tf_array[i] > 0]
        words_by_tf = sorted(words_by_tf, key=lambda x: x[1], reverse=True)[:10] 
        
        # 2. Siapkan daftar IDF (Kelangkaan Tertinggi)
        idf_weights = tfidf_model.idf_
        words_by_idf = [(feature_names[i], idf_weights[i]) for i in range(len(tf_array)) if tf_array[i] > 0]
        words_by_idf = sorted(words_by_idf, key=lambda x: x[1], reverse=True)[:10]

        print("PROSES 1 & 2: MENGHITUNG NILAI TF LOKAL & IDF GLOBAL")
        print("=" * 95)
        print(f"{'Hasil Nilai TF Lokal (Kemunculan)'.ljust(35)} | {'Jumlah'.center(8)} | {'Hasil Nilai IDF Global (Kelangkaan)'.ljust(35)} | {'Bobot IDF'.center(9)}")
        print("-" * 36 + "+" + "-" * 10 + "+" + "-" * 37 + "+" + "-" * 11)
        
        for i in range(len(words_by_tf)):
            tf_word, tf_val = words_by_tf[i]
            idf_word, idf_val = words_by_idf[i]
            print(f"{tf_word.ljust(35)} | {str(tf_val).center(8)} | {idf_word.ljust(35)} | {str(round(idf_val, 4)).center(9)}")

        # ---------------------------------------------------------
        # PROSES 3: MENGHITUNG TF-IDF (TF x IDF dengan Normalisasi)
        # ---------------------------------------------------------
        print("\nPROSES 3: PEMBENTUKAN MATRIKS FINAL TF-IDF")
        print("=" * 60)
        
        tfidf_matrix = tfidf_model.transform([combined_text])
        tfidf_array = tfidf_matrix.toarray()[0]
        
        # 3. Siapkan daftar Final TF-IDF (Bobot Tertinggi)
        words_by_tfidf = [(feature_names[i], tfidf_array[i]) for i in range(len(tf_array)) if tfidf_array[i] > 0]
        words_by_tfidf = sorted(words_by_tfidf, key=lambda x: x[1], reverse=True)[:10]

        print(f"{'Kata (Token)'.ljust(35)} | {'Bobot Final TF-IDF'.center(20)}")
        print("-" * 36 + "+" + "-" * 22)
        for word, final_val in words_by_tfidf:
            print(f"{word.ljust(35)} | {str(round(final_val, 4)).center(20)}")

if __name__ == "__main__":
    run_demo_breakdown()
