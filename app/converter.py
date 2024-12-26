import fitz  # PyMuPDF
from PIL import Image
import os

OUTPUT_FOLDER = "data/output/"

def convert_pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img.save(os.path.join(OUTPUT_FOLDER, f"{os.path.basename(pdf_path)}_page_{page_num}.png"), "PNG")