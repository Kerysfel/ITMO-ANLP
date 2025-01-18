import fitz  # PyMuPDF
from app.models.colpali import ColPaliModel
from pymilvus import Collection
import numpy as np
from PIL import Image
import os

colpali = ColPaliModel()

async def process_and_store_pdf(file):
    """
    Reads an uploaded PDF file, extracts text and images from each page,
    and inserts embeddings into the 'cheat_sheets' Milvus collection.
    """
    pdf_path = f"data/output/{file.filename}"
    with open(pdf_path, "wb") as f:
        f.write(file.file.read())

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    collection = Collection("cheat_sheets")

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        text_embedding = colpali.embed_text(text)

        # 1) Insert text as a separate record
        data_text = {
            "embedding": text_embedding.tolist(),
            "text": text,
            "pdf_path": pdf_path,
            "page_number": page_num
        }
        collection.insert(data_text)

        # 2) Insert each image as a separate record
        images = page.get_images(full=True)
        for img_index, img_info in enumerate(images):
            xref = img_info[0]
            base_image = fitz.Pixmap(doc, xref)
            pil_image = Image.frombytes(
                "RGB", [base_image.width, base_image.height],
                base_image.samples
            )
            img_embedding = colpali.embed_image(pil_image)

            data_img = {
                "embedding": img_embedding.tolist(),
                "text": text,
                "pdf_path": pdf_path,
                "page_number": page_num
            }
            collection.insert(data_img)