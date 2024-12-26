from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType
from transformers import AutoProcessor
from app.models.colpali import ColPaliModel
import fitz  # PyMuPDF
import numpy as np
import os
import yaml

# Константы для Milvus
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "cheat_sheets"
DIMENSION = 768  # Размер эмбеддинга
PATHS_FILE = "configs/paths.yaml"

# Подключение к Milvus
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

# Инициализация ColPali для эмбеддингов
colpali = ColPaliModel()

# Создание коллекции Milvus (если не существует)
def create_milvus_collection():
    collection = Collection(COLLECTION_NAME)
    if not collection.has_collection(COLLECTION_NAME):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="pdf_path", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="page_number", dtype=DataType.INT64)
        ]
        schema = CollectionSchema(fields, description="Cheat sheet embeddings")
        collection = Collection(COLLECTION_NAME, schema)
        collection.create_index("embedding", {"metric_type": "L2"})
    return collection

# Извлечение текста и изображений из PDF
def extract_text_and_images(pdf_path):
    doc = fitz.open(pdf_path)
    embeddings = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        
        # Векторизация текста
        text_embedding = colpali.embed_text(text)
        
        # Векторизация изображений (если есть)
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = fitz.Pixmap(doc, xref)
            img_embedding = colpali.embed_image(base_image)
            embeddings.append((img_embedding, pdf_path, page_num, text))

        # Если изображений нет — сохраняем только текст
        if not images:
            embeddings.append((text_embedding, pdf_path, page_num, text))

    return embeddings

# Обновление paths.yaml
def update_paths(pdf_path, page_number, vector_id):
    # Проверяем существование файла paths.yaml
    if os.path.exists(PATHS_FILE):
        with open(PATHS_FILE, "r") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {"documents": []}

    # Ищем существующую запись для PDF
    existing_pdf = next((doc for doc in data["documents"] if doc["pdf_path"] == pdf_path), None)

    if existing_pdf:
        existing_pdf["pages"].append({"page": page_number, "vector_id": vector_id})
    else:
        new_entry = {
            "pdf_path": pdf_path,
            "pages": [{"page": page_number, "vector_id": vector_id}]
        }
        data["documents"].append(new_entry)

    # Записываем обновлённые пути
    with open(PATHS_FILE, "w") as f:
        yaml.safe_dump(data, f)

# Загрузка в Milvus с обновлением paths.yaml
def insert_to_milvus(pdf_folder):
    collection = create_milvus_collection()
    for root, _, files in os.walk(pdf_folder):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                print(f"Обработка: {pdf_path}")
                embeds = extract_text_and_images(pdf_path)
                
                data = {
                    "embedding": [embed[0] for embed in embeds],
                    "pdf_path": [embed[1] for embed in embeds],
                    "page_number": [embed[2] for embed in embeds],
                    "text": [embed[3] for embed in embeds]
                }
                
                # Вставляем эмбеддинги в Milvus и сохраняем пути
                insert_result = collection.insert(data)
                for i, inserted_id in enumerate(insert_result.primary_keys):
                    update_paths(pdf_path, embeds[i][2], inserted_id)  # page_num и vector_id

insert_to_milvus("archive/")