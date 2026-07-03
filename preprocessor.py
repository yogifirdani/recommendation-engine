import re
import logging
from bs4 import BeautifulSoup
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd

# Setup logging
logger = logging.getLogger(__name__)

# ==========================================================
# INISIALISASI SASTRAWI SEKALI SAJA DI LEVEL MODUL
# ==========================================================
logger.info("Menginisialisasi Sastrawi Stemmer dan StopWordRemover...")
try:
    stemmer = StemmerFactory().create_stemmer()
    stop_words_factory = StopWordRemoverFactory()
    base_stopwords = stop_words_factory.get_stop_words()
except Exception as e:
    logger.error(f"Gagal menginisialisasi PySastrawi: {str(e)}")
    raise e

# Kata-kata stopword kustom tambahan dalam konteks pariwisata
custom_stopwords = [
    "hari", "malam", "jam", "dll", "dsb", "atau", 
    "dan", "dengan", "untuk", "yang", "ini", "itu",
    "ke", "di", "dari", "pada", "oleh", "akan",
    "pax", "rp", "trip", "tour", "open", "day",
    "saya", "ingin", "bisa", "dapat", "mau", "buat",
    "menikmati", "nikmat", "indahnya", "indah",
    "melihat", "lihat", "merasakan", "rasa", "sangat"
]

# Menggabungkan stopword bawaan dan kustom menjadi satu set pencarian unik
all_stopwords = set(base_stopwords + custom_stopwords)
logger.info(f"Sastrawi berhasil diinisialisasi. Total stopword: {len(all_stopwords)}")

def strip_html(text):
    """
    Menghapus tag HTML dari teks dan menyisakan teks biasa.
    """
    if text is None:
        return ""
    # Pastikan diubah ke string
    text_str = str(text).strip()
    if not text_str:
        return ""
    try:
        soup = BeautifulSoup(text_str, "html.parser")
        return soup.get_text(separator=" ")
    except Exception as e:
        logger.warning(f"Gagal melakukan strip HTML: {str(e)}")
        return text_str

def case_folding(text):
    """
    Mengubah teks menjadi huruf kecil dan menghapus semua karakter 
    selain huruf, angka, dan spasi. Serta memangkas spasi berlebih.
    """
    if not text:
        return ""
    # 1. Ubah ke lowercase
    text = text.lower()
    # 2. Hapus karakter non-alfanumerik (pertahankan huruf, angka, spasi)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # 3. Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def remove_stopwords(text):
    """
    Menghapus kata-kata yang tidak memiliki makna penting (stopword).
    """
    if not text:
        return ""
    words = text.split()
    filtered_words = [word for word in words if word not in all_stopwords]
    return " ".join(filtered_words)

# Daftar kata pengecualian yang tidak boleh di-stem (nama destinasi/tempat)
stemming_exemptions = set([
    "baluran", "djawatan", "tabuhan", "menjangan", 
    "ijen", "wurung", "merah", "banyuwangi"
])

def stemming(text):
    """
    Mengubah kata berimbuhan menjadi kata dasar menggunakan Sastrawi.
    Pengecualian diberikan untuk kata-kata khusus (nama destinasi).
    """
    if not text:
        return ""
    try:
        # Pisahkan kata dan cek satu per satu
        words = text.split()
        stemmed_words = []
        for word in words:
            if word in stemming_exemptions:
                stemmed_words.append(word)
            else:
                stemmed_words.append(stemmer.stem(word))
        return " ".join(stemmed_words)
    except Exception as e:
        logger.warning(f"Gagal melakukan stemming: {str(e)}")
        return text


def tokenize(text):
    """
    Memotong teks menjadi token (kata), membuang kata kosong dan kata 
    yang panjang karakternya kurang dari 2.
    """
    if not text:
        return ""
    words = text.split()
    filtered_tokens = [word for word in words if len(word) >= 2]
    return " ".join(filtered_tokens)

def preprocess(text):
    """
    Pipeline prapemrosesan teks lengkap:
    strip_html -> case_folding -> remove_stopwords -> stemming -> tokenize
    """
    if not text:
        return ""
    
    cleaned = strip_html(text)
    folded = case_folding(cleaned)
    no_stop = remove_stopwords(folded)
    stemmed = stemming(no_stop)
    tokenized = tokenize(stemmed)
    
    return tokenized

def get_package_price_tags(price_val):
    """
    Mengubah harga riil paket menjadi HANYA SATU tag batas budget yang sesuai.
    Ini mencegah paket murah mendapatkan skor yang tidak wajar akibat overlap tag yang banyak.
    Contoh: Harga 250.000 -> menghasilkan HANYA 'budgetbracket250k'
    """
    try:
        price = float(price_val)
    except (ValueError, TypeError):
        return ""
    
    thresholds = [
        150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000,
        600000, 700000, 800000, 900000, 1000000, 1500000, 2000000, 
        2500000, 3000000, 4000000, 5000000
    ]
    for limit in thresholds:
        if price <= limit:
            return f"budgetbracket{limit // 1000}k"
            
    return "budgetbracketabove5000k"

def get_user_budget_tags(budget_val):
    """
    Mengubah budget pencarian wisatawan menjadi deretan tag batas budget yang mampu dibelinya.
    Karena paket hanya memiliki 1 tag, overlap (kecocokan) maksimal selalu 1 tag untuk semua rentang harga.
    Contoh: Budget 500.000 -> menghasilkan tag:
    'budgetbracket150k budgetbracket200k ... budgetbracket500k'
    """
    try:
        budget = float(budget_val)
    except (ValueError, TypeError):
        return ""
    
    thresholds = [
        150000, 200000, 250000, 300000, 350000, 400000, 450000, 500000,
        600000, 700000, 800000, 900000, 1000000, 1500000, 2000000, 
        2500000, 3000000, 4000000, 5000000
    ]
    tags = []
    for limit in thresholds:
        if limit <= budget:
            tags.append(f"budgetbracket{limit // 1000}k")
    if budget > 5000000:
        tags.append("budgetbracketabove5000k")
    return " ".join(tags)

def standardize_duration(duration_str):
    """
    Menyeragamkan penulisan durasi agar durasi bahasa Inggris (Day) 
    dan bahasa Indonesia (Hari/Jam) memiliki kata kunci yang cocok secara teks.
    """
    if not duration_str:
        return ""
    s = str(duration_str).lower()
    tags = []
    if "7 jam" in s or "7 hour" in s or "7" in s:
        tags.append("durasi7jam")
    if "1 day" in s or "1 hari" in s or "one day" in s or "1d" in s:
        tags.append("durasi1day")
    if "3d2n" in s or "3 hari" in s or "3 day" in s or "3d" in s:
        tags.append("durasi3d2n")
    return " ".join(tags) if tags else s

def build_combined_features(row):
    """
    Menerima baris data dari tour_packages DataFrame, menggabungkan atribut-atributnya,
    dan menerapkan pembersihan preprocessing lengkap sesuai dengan kriteria resmi penelitian:
    Kategori Relasi, Kategori Wisata, Harga (dalam bentuk tag), Durasi (standar), Fasilitas, dan Deskripsi Paket.
    """
    category_name = str(row.get('category_name') or '')
    tour_category = str(row.get('tour_category') or '')
    
    # Konversi harga ke tag budget TF-IDF
    price_tags = get_package_price_tags(row.get('pax1'))
    
    # Standarisasi durasi
    duration = standardize_duration(row.get('duration'))
    
    facilities_included = str(row.get('facilities_included') or '')
    highlight = str(row.get('highlight') or '')
    # Berikan bobot lebih besar pada 'highlight' (deskripsi inti) dengan mengulanginya 3x
    # Ini memastikan kata kunci unik wisata (seperti 'pantai', 'gunung') mengalahkan kata umum fasilitas.
    highlight_weighted = f"{highlight} " * 3
    
    # Gabungkan kriteria resmi penelitian
    combined = (
        f"{category_name} {tour_category} {price_tags} {duration} {facilities_included} {highlight_weighted}"
    )
    
    return preprocess(combined)

def build_preference_features(preference_dict):
    """
    Menerima dictionary data preferensi wisatawan, menggabungkan atributnya,
    dan menerapkan pembersihan preprocessing lengkap.
    """
    tour_category = str(preference_dict.get('tour_category') or '')
    
    # Konversi budget ke tag budget TF-IDF
    budget_tags = get_user_budget_tags(preference_dict.get('budget'))
    
    # Standarisasi durasi preferensi
    preferred_duration = standardize_duration(preference_dict.get('preferred_duration'))
    
    preferred_facilities = str(preference_dict.get('preferred_facilities') or '')
    
    description = str(preference_dict.get('description') or '')
    # Berikan bobot lebih besar pada 'description' (keinginan utama user) dengan mengulanginya 3x
    # Ini memastikan intensi utama user (misal: 'snorkeling', 'laut') menang atas fasilitas umum (misal: 'mobil')
    description_weighted = f"{description} " * 3
    
    # Gabungkan semua fitur preferensi menjadi satu string teks
    pref = (
        f"{tour_category} {budget_tags} {preferred_duration} {preferred_facilities} {description_weighted}"
    )
    
    return preprocess(pref)
