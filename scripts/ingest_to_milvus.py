import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType, utility
from models.colpali import ColPaliModel
import fitz  # PyMuPDF
import numpy as np
import yaml
from PIL import Image
import torch

MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "cheat_sheets"
DIMENSION = 128  # Matches ColPali's embedding dimension
PATHS_FILE = "configs/paths.yaml"

connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

colpali = ColPaliModel()

def create_milvus_collection():
    """
    Creates or retrieves an existing Milvus collection named 'cheat_sheets'.
    """
    if utility.has_collection(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        return Collection(COLLECTION_NAME)

    print(f"Creating collection '{COLLECTION_NAME}'...")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="pdf_path", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="page_number", dtype=DataType.INT64)
    ]

    schema = CollectionSchema(fields, description="Cheat sheet embeddings")
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
    collection.create_index(field_name="embedding", index_params=index_params)

    print(f"Collection '{COLLECTION_NAME}' created successfully.")
    return collection

def extract_text_and_images(pdf_path):
    """
    Opens a PDF, extracts text and images from each page,
    and returns a list of tuples (embedding, pdf_path, page_number, text).
    """
    doc = fitz.open(pdf_path)
    embeddings = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()

        text_embedding = colpali.embed_text(text)
        text_embedding = np.expand_dims(text_embedding, axis=0)  # shape (1,128)

        images = page.get_images(full=True)
        if images:
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = fitz.Pixmap(doc, xref)
                if base_image.n >= 4:
                    base_image = fitz.Pixmap(fitz.csRGB, base_image)

                pil_image = Image.frombytes(
                    "RGB", [base_image.width, base_image.height],
                    base_image.samples
                )
                img_embedding = colpali.embed_image(pil_image)

                embeddings.append((img_embedding, pdf_path, page_num, text))
        else:
            # No images, just store the text embedding
            embeddings.append((text_embedding, pdf_path, page_num, text))

    return embeddings

def update_paths(pdf_path, page_number, vector_id):
    """
    Updates configs/paths.yaml with the vector_id for each page of the PDF.
    """
    if os.path.exists(PATHS_FILE):
        with open(PATHS_FILE, "r") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {"documents": []}

    existing_pdf = next(
        (doc for doc in data.get("documents", []) if doc["pdf_path"] == pdf_path),
        None
    )

    if existing_pdf:
        existing_pdf["pages"].append({"page": page_number, "vector_id": vector_id})
    else:
        new_entry = {
            "pdf_path": pdf_path,
            "pages": [{"page": page_number, "vector_id": vector_id}]
        }
        data["documents"].append(new_entry)

    with open(PATHS_FILE, "w") as f:
        yaml.safe_dump(data, f)

def insert_to_milvus(pdf_folder):
    """
    Recursively finds PDFs in pdf_folder, processes each,
    and inserts embeddings into Milvus. Also updates paths.yaml.
    """
    collection = create_milvus_collection()
    for root, _, files in os.walk(pdf_folder):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                print(f"Processing: {pdf_path}")
                embeds = extract_text_and_images(pdf_path)

                data = {
                    "embedding": np.vstack([e[0] for e in embeds]).astype(np.float32),
                    "pdf_path": [e[1] for e in embeds],
                    "page_number": [e[2] for e in embeds],
                    "text": [e[3] for e in embeds]
                }

                insert_result = collection.insert(data)
                for i, inserted_id in enumerate(insert_result.primary_keys):
                    update_paths(pdf_path, embeds[i][2], inserted_id)

insert_to_milvus("archive/")