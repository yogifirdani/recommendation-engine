import PyPDF2

def extract_text_from_pdf(pdf_path, out_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            text = f"Total pages: {num_pages}\n"
            for i in range(num_pages):
                page = reader.pages[i]
                text += f"\n--- Page {i+1} ---\n"
                text += page.extract_text()
                
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(text)
    except Exception as e:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(f"Error reading PDF: {e}")

if __name__ == "__main__":
    pdf_file = "Minggu 10 Aplikasi Data Mining untuk deteksi infeksi sistem Gastro Usus (1).pdf"
    out_file = "pdf_output.txt"
    extract_text_from_pdf(pdf_file, out_file)
