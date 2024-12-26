import fitz  # PyMuPDF
from app.models.colpali import ColPaliModel
from pymilvus import Collection
import numpy as np
from PIL import Image

colpali = ColPaliModel()

async def process_and_store_pdf(file):
    pdf_path = f"data/output/{file.filename}"
    with open(pdf_path, "wb") as f:
        f.write(file.file.read())

    doc = fitz.open(pdf_path)
    collection = Collection("cheat_sheets")

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()

        # Векторизация текста
        text_embedding = colpali.embed_text(text)

        # Векторизация изображений
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = fitz.Pixmap(doc, xref)
            pil_image = Image.frombytes("RGB", [base_image.width, base_image.height], base_image.samples)
            img_embedding = colpali.embed_image(pil_image)

            data = {
                "embedding": img_embedding.tolist(),
                "text": text,
                "pdf_path": pdf_path,
                "page_number": page_num
            }
            collection.insert(data)