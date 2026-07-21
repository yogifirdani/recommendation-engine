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

# Kata-kata stopword kustom tambahan dalam konteks pariwisata + Tala Stopwords
custom_stopwords = [
    # --- Stopword Konteks Pariwisata ---
    "pax", "rp", "trip", "tour", "open", "day", "menikmati", "nikmat", 
    "indahnya", "indah", "melihat", "lihat", "merasakan", "rasa", "jam", "dll", "dsb", "malam", "hari",

    # --- Tala Stopwords Indonesia ---
    "ada", "adalah", "adanya", "adapun", "agak", "agaknya", "agar", "akan", "akankah", "akhir", 
    "akhiri", "akhirnya", "aku", "akulah", "amat", "amatlah", "anda", "andalah", "antar", "antara", 
    "antaranya", "apa", "apaan", "apabila", "apakah", "apalagi", "apatah", "artinya", "asal", "asalkan", 
    "atas", "atau", "ataukah", "ataupun", "awal", "awalnya", "bagai", "bagaikan", "bagaimana", "bagaimanakah", 
    "bagaimanapun", "bagi", "bagian", "bahkan", "bahwa", "bahwasanya", "baik", "bakal", "bakalan", "balik", 
    "banyak", "bapak", "baru", "bawah", "beberapa", "begini", "beginian", "beginikah", "beginilah", "begitu", 
    "begitukah", "begitulah", "begitupun", "bekerja", "belakang", "belakangan", "belum", "belumlah", "benar", 
    "benarkah", "benarlah", "berada", "berakhir", "berakhirlah", "berakhirnya", "berapa", "berapakah", "berapalah", 
    "berapapun", "berarti", "berawal", "berbagai", "berdatangan", "beri", "berikan", "berikut", "berikutnya", 
    "berjumlah", "berkali-kali", "berkata", "berkehendak", "berkeinginan", "berkenaan", "berlainan", "berlalu", 
    "berlangsung", "berlebihan", "bermacam", "bermacam-macam", "bermaksud", "bermula", "bersama", "bersama-sama", 
    "bersiap", "bersiap-siap", "bertanya", "bertanya-tanya", "berturut", "berturut-turut", "bertutur", "berujar", 
    "berupa", "besar", "betul", "betulkah", "biasa", "biasanya", "bila", "bilakah", "bisa", "bisakah", "boleh", 
    "bolehkah", "bolehlah", "buat", "bukan", "bukankah", "bukanlah", "bukannya", "bulan", "bung", "cara", "caranya", 
    "cukup", "cukupkah", "cukuplah", "cuma", "dahulu", "dalam", "dan", "dapat", "dari", "daripada", "datang", "dekat", 
    "demi", "demikian", "demikianlah", "dengan", "depan", "di", "dia", "diakhiri", "diakhirinya", "dialah", "diantara", 
    "diantaranya", "diberi", "diberikan", "diberikannya", "dibuat", "dibuatnya", "didapat", "didatangkan", "digunakan", 
    "diibaratkan", "diibaratkannya", "diingat", "diingatkan", "diinginkan", "dijawab", "dijelaskan", "dijelaskannya", 
    "dikarenakan", "dikatakan", "dikatakannya", "dikerjakan", "diketahui", "diketahuinya", "dikira", "dilakukan", 
    "dilalui", "dilihat", "dimaksud", "dimaksudkan", "dimaksudkannya", "dimaksudnya", "diminta", "dimintai", 
    "dimisalkan", "dimulai", "dimulailah", "dimulainya", "dimungkinkan", "dini", "dipastikan", "diperbuat", 
    "diperbuatnya", "dipergunakan", "diperkirakan", "diperlihatkan", "diperlukan", "diperlukannya", "dipersoalkan", 
    "dipertanyakan", "dipunyai", "diri", "dirinya", "disampaikan", "disebut", "disebutkan", "disebutkannya", "disini", 
    "disinilah", "ditambahkan", "ditandaskan", "ditanya", "ditanyai", "ditanyakan", "ditegaskan", "ditujukan", "ditunjuk", 
    "ditunjuki", "ditunjukkan", "ditunjukkannya", "ditunjuknya", "dituturkan", "dituturkannya", "diucapkan", "diucapkannya", 
    "diungkapkan", "dong", "dua", "dulu", "empat", "enggak", "enggaknya", "entah", "entahlah", "guna", "gunakan", "hal", 
    "hampir", "hanya", "hanyalah", "hari", "harus", "haruslah", "harusnya", "hendak", "hendaklah", "hendaknya", "hingga", 
    "ia", "ialah", "ibarat", "ibaratkan", "ibaratnya", "ibu", "ikut", "ingat", "ingat-ingat", "ingin", "inginkah", "inginkan", 
    "ini", "inikah", "inilah", "itu", "itukah", "itulah", "jadi", "jadilah", "jadinya", "jangan", "jangankan", "janganlah", 
    "jauh", "jawab", "jawaban", "jawabnya", "jelas", "jelaskan", "jelaslah", "jelasnya", "jika", "jikalau", "juga", "jumlah", 
    "jumlahnya", "justru", "kala", "kalau", "kalaulah", "kalaupun", "kalian", "kami", "kamilah", "kamu", "kamulah", "kan", 
    "kapan", "kapankah", "kapanpun", "karena", "karenanya", "kasus", "kata", "katakan", "katakanlah", "katanya", "ke", 
    "keadaan", "kebetulan", "kecil", "kedua", "keduanya", "keinginan", "kelamaan", "kelihatan", "kelihatannya", "kelima", 
    "keluar", "kembali", "kemudian", "kemungkinan", "kemungkinannya", "kenapa", "kepada", "kepadanya", "kesampaian", 
    "keseluruhan", "keseluruhannya", "keterlaluan", "ketika", "khususnya", "kini", "kinilah", "kira", "kira-kira", 
    "kiranya", "kita", "kitalah", "kok", "kurang", "lagi", "lagian", "lah", "lain", "lainnya", "lalu", "lama", "lamanya", 
    "lanjut", "lanjutnya", "lebih", "lewat", "lima", "luar", "macam", "maka", "makanya", "makin", "malah", "malahan", 
    "mampu", "mampukah", "mana", "manakala", "manalagi", "masa", "masalah", "masalahnya", "masih", "masihkah", "masing", 
    "masing-masing", "mau", "maupun", "melainkan", "melakukan", "melalui", "melihat", "melihatnya", "memang", "memastikan", 
    "memberi", "memberikan", "membuat", "memerlukan", "memihak", "meminta", "memintakan", "memisalkan", "memperbuat", 
    "mempergunakan", "memperkirakan", "memperlihatkan", "mempersiapkan", "mempersoalkan", "mempertanyakan", "mempunyai", 
    "memulai", "memungkinkan", "menaiki", "menambahkan", "menandaskan", "menanti", "menanti-nanti", "menantikan", 
    "menanya", "menanyai", "menanyakan", "mendapat", "mendapatkan", "mendatang", "mendatangi", "mendatangkan", 
    "menegaskan", "mengakhiri", "mengapa", "mengatakan", "mengatakannya", "mengenai", "mengerjakan", "mengetahui", 
    "menggunakan", "menghendaki", "mengibaratkan", "mengibaratkannya", "mengingat", "mengingatkan", "menginginkan", 
    "mengira", "mengucapkan", "mengucapkannya", "mengungkapkan", "menjadi", "menjawab", "menjelaskan", "menuju", "menunjuk", 
    "menunjuki", "menunjukkan", "menunjuknya", "menurut", "menuturkan", "menyampaikan", "menyangkut", "menyatakan", 
    "menyebutkan", "menyeluruh", "menyiapkan", "merasa", "mereka", "merekalah", "merupakan", "meski", "meskipun", 
    "meyakini", "meyakinkan", "minta", "mirip", "misal", "misalkan", "misalnya", "mula", "mulai", "mulailah", "mulanya", 
    "mungkin", "mungkinkah", "nah", "naik", "namun", "nanti", "nantinya", "nyaris", "nyatanya", "oleh", "olehnya", "pada", 
    "padahal", "padanya", "pak", "paling", "panjang", "pantas", "para", "pasti", "pastilah", "penting", "pentingnya", "per", 
    "percuma", "perlu", "perlukah", "perlunya", "pernah", "persoalan", "pertama", "pertama-tama", "pertanyaan", "pertanyakan", 
    "pihak", "pihaknya", "pukul", "pula", "pun", "punya", "rasa", "rasanya", "rata", "rupanya", "saat", "saatnya", "saja", 
    "sajalah", "saling", "sama", "sama-sama", "sambil", "sampai", "sampai-sampai", "sampaikan", "sana", "sangat", 
    "sangatlah", "satu", "saya", "sayalah", "se", "sebab", "sebabnya", "sebagai", "sebagaimana", "sebagainya", "sebagian", 
    "sebaik", "sebaik-baiknya", "sebaiknya", "sebaliknya", "sebanyak", "sebegini", "sebegitu", "sebelum", "sebelumnya", 
    "sebenarnya", "seberapa", "sebesar", "sebetulnya", "sebisanya", "sebuah", "sebut", "sebutlah", "sebutnya", "secara", 
    "secukupnya", "sedang", "sedangkan", "sedemikian", "sedikit", "sedikitnya", "seenaknya", "segala", "segalanya", 
    "segera", "seharusnya", "sehingga", "seingat", "sejak", "sejauh", "sejenak", "sejumlah", "sekadar", "sekadarnya", 
    "sekali", "sekali-kali", "sekalian", "sekaligus", "sekalipun", "sekarang", "sekecil", "seketika", "sekiranya", 
    "sekitar", "sekitarnya", "sekurang-kurangnya", "sekurangnya", "sela", "selain", "selaku", "selalu", "selama", 
    "selama-lamanya", "selamanya", "selanjutnya", "seluruh", "seluruhnya", "semacam", "semakin", "semampu", "semampunya", 
    "semasa", "semasih", "semata", "semata-mata", "semaunya", "sementara", "semisal", "semisalnya", "sempat", "semua", 
    "semuanya", "semula", "sendiri", "sendirian", "sendirinya", "seolah", "seolah-olah", "seorang", "sepanjang", 
    "sepantasnya", "sepantasnyalah", "seperlunya", "seperti", "sepertinya", "sepihak", "sering", "seringnya", "serta", 
    "serupa", "sesaat", "sesama", "sesampai", "sesegera", "sesekali", "seseorang", "sesuatu", "sesuatunya", "sesudah", 
    "sesudahnya", "setelah", "setempat", "setengah", "seterusnya", "setiap", "setiba", "setibanya", "setidak-tidaknya", 
    "setidaknya", "setinggi", "seusai", "sewaktu", "siap", "siapa", "siapakah", "siapapun", "sini", "sinilah", "soal", 
    "soalnya", "suatu", "sudah", "sudahkah", "sudahlah", "supaya", "tadi", "tadinya", "tahu", "tahun", "tak", "tambah", 
    "tambahnya", "tampak", "tampaknya", "tandas", "tandasnya", "tanpa", "tanya", "tanyakan", "tanyanya", "tapi", "tegas", 
    "tegasnya", "telah", "tempat", "tengah", "tentang", "tentu", "tentulah", "tentunya", "tepat", "terakhir", "terasa", 
    "terbanyak", "terdahulu", "terdapat", "terdiri", "terhadap", "terhadapnya", "teringat", "teringat-ingat", "terjadi", 
    "terjadilah", "terjadinya", "terkira", "terlalu", "terlebih", "terlihat", "termasuk", "ternyata", "tersampaikan", 
    "tersebut", "tersebutlah", "tertentu", "tertuju", "terus", "terutama", "tetap", "tetapi", "tiap", "tiba", "tiba-tiba", 
    "tidak", "tidakkah", "tidaklah", "tiga", "tinggi", "toh", "tunjuk", "turut", "tutur", "tuturnya", "ucap", "ucapnya", 
    "ujar", "ujarnya", "umum", "umumnya", "ungkap", "ungkapnya", "untuk", "usah", "usai", "waduh", "wah", "wahai", 
    "waktu", "waktunya", "walau", "walaupun", "wong", "yaitu", "yakin", "yakni", "yang"
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
    if "2d1n" in s or "2 hari" in s or "2 day" in s or "2d" in s:
        tags.append("durasi2d1n")
    if "3d2n" in s or "3 hari" in s or "3 day" in s or "3d" in s:
        tags.append("durasi3d2n")
    if "4d3n" in s or "4 hari" in s or "4 day" in s or "4d" in s:
        tags.append("durasi4d3n")
    return " ".join(tags) if tags else s

def build_combined_features(row):
    """
    Menerima baris data dari tour_packages DataFrame, dan HANYA mengambil
    Deskripsi Paket (highlight) untuk diproses menjadi teks TF-IDF (Tahap 1).
    Parameter lain (Budget, Kategori, Fasilitas) akan diproses terpisah di Tahap 2 (Filtering).
    """
    highlight = str(row.get('highlight') or '')
    
    # Berikan bobot lebih besar pada 'highlight' (deskripsi inti) dengan mengulanginya 3x
    # Ini memastikan kata kunci unik wisata (seperti 'pantai', 'gunung') sangat dipertimbangkan
    highlight_weighted = f"{highlight} " * 3
    
    return preprocess(highlight_weighted)

def build_preference_features(preference_dict):
    """
    Menerima dictionary data preferensi wisatawan, dan HANYA mengambil
    Deskripsi untuk diproses menjadi teks TF-IDF (Tahap 1).
    """
    description = str(preference_dict.get('description') or '')
    
    # Berikan bobot lebih besar pada 'description' (keinginan utama user) dengan mengulanginya 3x
    description_weighted = f"{description} " * 3
    
    return preprocess(description_weighted)