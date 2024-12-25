import os
import json
from pdf2image import convert_from_path
import numpy as np
import requests
import faiss
import torch

# Пути к данным (измените на ваши локальные пути)
DATASET_PATH = 'archive/'  # Папка с PDF-файлами
OUTPUT_PATH = 'output/'    # Папка для сохранения file_paths.json
VECTORS_PATH = 'vectors/'  # Папка для сохранения embeddings.npy и FAISS индекса

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(VECTORS_PATH, exist_ok=True)

# API URL вашего локального сервера ColPali
API_URL = "http://127.0.0.1:5000/embed/image"

# Функция для обработки PDF в изображения
def pdf_to_images(pdf_path):
    try:
        images = convert_from_path(pdf_path, dpi=300)
        return images
    except Exception as e:
        print(f"Ошибка при конвертации PDF {pdf_path}: {e}")
        return []

# Функция для отправки изображения в ColPali API
def get_embedding(image):
    image.save("temp_image.png")
    with open("temp_image.png", "rb") as f:
        files = {'file': f}
        try:
            response = requests.post(API_URL, files=files, timeout=30)
            if response.status_code == 200:
                return np.array(response.json()["embedding"])
            else:
                print("Ошибка при обработке изображения:", response.json())
                return None
        except requests.exceptions.RequestException as e:
            print(f"Запрос к API не удался: {e}")
            return None

# Основная функция обработки всех PDF
def process_pdfs():
    embeddings = []
    file_paths = {}

    for foldername, subfolders, filenames in os.walk(DATASET_PATH):
        for filename in filenames:
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(foldername, filename)
                print(f"Обработка PDF: {pdf_path}")
                images = pdf_to_images(pdf_path)
                
                for i, image in enumerate(images):
                    print(f"  Обработка страницы {i+1}/{len(images)}")
                    embedding = get_embedding(image)
                    if embedding is not None:
                        embeddings.append(embedding)
                        # Сохраняем путь к странице PDF
                        file_paths[len(embeddings)-1] = {
                            "pdf": pdf_path,
                            "page": i
                        }

    # Сохранение векторов и путей в файлы
    if embeddings:
        embeddings = np.array(embeddings)
        np.save(os.path.join(VECTORS_PATH, "embeddings.npy"), embeddings)
        print(f"Эмбеддинги сохранены в {os.path.join(VECTORS_PATH, 'embeddings.npy')}")
    else:
        print("Эмбеддинги не были созданы.")

    if file_paths:
        with open(os.path.join(OUTPUT_PATH, "file_paths.json"), "w", encoding='utf-8') as f:
            json.dump(file_paths, f, ensure_ascii=False, indent=4)
        print(f"Пути к файлам сохранены в {os.path.join(OUTPUT_PATH, 'file_paths.json')}")
    else:
        print("Пути к файлам не были созданы.")

    # Создание и сохранение индекса FAISS
    if embeddings:
        d = embeddings.shape[1]  # Размерность эмбеддингов
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        faiss_index_path = os.path.join(VECTORS_PATH, "faiss_index.index")
        faiss.write_index(index, faiss_index_path)
        print(f"FAISS индекс создан и сохранён в {faiss_index_path}")
    else:
        print("Эмбеддинги не загружены. FAISS индекс не создан.")

    print("Обработка завершена! Векторы сохранены и FAISS индекс создан.")

process_pdfs()
