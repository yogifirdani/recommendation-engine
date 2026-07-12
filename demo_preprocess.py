import preprocessor
import database
import textwrap
import warnings
from bs4 import MarkupResemblesLocatorWarning

# Menyembunyikan warning dari BeautifulSoup agar tabel tidak berantakan
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

def print_table(package_name, data):
    col1_width = 23
    col2_width = 90  # Lebar kolom teks
    
    border = f"+-{'-' * col1_width}-+-{'-' * col2_width}-+"
    
    print("\n")
    print(f"=== PAKET: {package_name} ===")
    print(border)
    print(f"| {'TAHAPAN'.ljust(col1_width)} | {'HASIL TEKS'.ljust(col2_width)} |")
    print(border)
    
    for tahapan, teks in data:
        # Memotong teks menjadi beberapa baris jika terlalu panjang
        wrapped_teks = textwrap.wrap(teks, width=col2_width)
        if not wrapped_teks:
            wrapped_teks = [""]
            
        print(f"| {tahapan.ljust(col1_width)} | {wrapped_teks[0].ljust(col2_width)} |")
        
        for line in wrapped_teks[1:]:
            print(f"| {''.ljust(col1_width)} | {line.ljust(col2_width)} |")
            
        print(border)
    print("\n")

# Mengambil data dari database
print("Mengambil data dari database...")
df = database.get_active_packages()

# Filter untuk mencari 3 paket yang Anda sebutkan
# ID 1: Kawah Ijen Banyuwangi
# ID 3: Tabuhan & Menjangan Island
# ID 12: Djawatan, Green Island, Pulau Merah
target_ids = [1, 3, 12, 7, 9]
filtered_df = df[df['id'].isin(target_ids)]

for index, row in filtered_df.iterrows():
    # 1. Menggabungkan fitur persis seperti pada preprocessor.build_combined_features
    category_name = str(row.get('category_name') or '')
    tour_category = str(row.get('tour_category') or '')
    price_tags = preprocessor.get_package_price_tags(row.get('pax1'))
    duration = preprocessor.standardize_duration(row.get('duration'))
    facilities_included = str(row.get('facilities_included') or '')
    highlight = str(row.get('highlight') or '')
    
    # Pada kode Anda, highlight (deskripsi) diulang 3x untuk memberi bobot TF-IDF lebih besar
    highlight_weighted = f"{highlight} " * 3
    
    # Ini adalah teks mentah dari database sebelum diproses (Kategori, Harga, Durasi, Fasilitas, Deskripsi)
    combined_raw = f"{category_name} {tour_category} {price_tags} {duration} {facilities_included} {highlight_weighted}"
    
    # 2. Menjalankan tahap-tahap preprocessing secara berurutan
    cleaned = preprocessor.strip_html(combined_raw)
    folded = preprocessor.case_folding(cleaned)
    no_stop = preprocessor.remove_stopwords(folded)
    stemmed = preprocessor.stemming(no_stop)
    tokenized = preprocessor.tokenize(stemmed)
    
    # Menyiapkan data untuk tabel
    data = [
        (" Teks Raw (Gabungan)", combined_raw.strip()),
        (" Case Folding", folded),
        (" Stopword Removal", no_stop),
        (" Stemming", stemmed),
        (" Tokenizing (Final)", tokenized)
    ]
    
    # Menampilkan tabel per paket
    print_table(row['name'], data)
