import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import fitz  # PyMuPDF
from PIL import Image

# Import from app.converter
from app.converter import convert_pdf_to_images

PDF_FOLDER = "archive/"
OUTPUT_FOLDER = "data/output/"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def process_all_pdfs():
    """
    Finds all PDF files in PDF_FOLDER and converts them to images
    using the convert_pdf_to_images() function in app.converter.
    """
    for root, _, files in os.walk(PDF_FOLDER):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                print(f"Converting: {pdf_path}")
                convert_pdf_to_images(pdf_path)

if __name__ == "__main__":
    process_all_pdfs()