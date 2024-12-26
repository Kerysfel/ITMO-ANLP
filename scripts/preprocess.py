import fitz  # PyMuPDF
from PIL import Image
import os

PDF_FOLDER = "archive/"
OUTPUT_FOLDER = "data/output/"

# Убедимся, что директория для результатов существует
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Преобразование PDF в изображения
def convert_pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)  # Высокое разрешение
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        output_path = os.path.join(
            OUTPUT_FOLDER,
            f"{os.path.basename(pdf_path).replace('.pdf', '')}_page_{page_num}.png"
        )
        img.save(output_path, "PNG")
        print(f"Изображение сохранено: {output_path}")

# Преобразование всех PDF
def process_all_pdfs():
    for root, _, files in os.walk(PDF_FOLDER):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                print(f"Конвертация: {pdf_path}")
                convert_pdf_to_images(pdf_path)

process_all_pdfs()