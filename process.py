import os
import json
import fitz  # PyMuPDF
import numpy as np
import requests
import faiss
from PIL import Image

# Пути к данным
DATASET_PATH = 'archive/'
OUTPUT_PATH = 'output/'
VECTORS_PATH = 'vectors/'

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(VECTORS_PATH, exist_ok=True)

API_URL = "http://127.0.0.1:5000/embed/image"

# Функция для обработки PDF в изображения (через PyMuPDF)
def pdf_to_images(pdf_path):
    images = []
    try:
        pdf_document = fitz.open(pdf_path)
        for page_number in range(len(pdf_document)):
            page = pdf_document.load_page(page_number)
            pix = page.get_pixmap(dpi=300)  # Устанавливаем DPI 300
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    except Exception as e:
        print(f"Ошибка при конвертации PDF {pdf_path}: {e}")
    return images

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

# Загрузка существующих путей, чтобы избежать повторной обработки
file_paths = {}
json_path = os.path.join(OUTPUT_PATH, "file_paths.json")

if os.path.exists(json_path):
    with open(json_path, "r", encoding='utf-8') as f:
        file_paths = json.load(f)

# Основная функция обработки PDF
def process_pdfs():
    global file_paths

    for foldername, subfolders, filenames in os.walk(DATASET_PATH):
        for filename in filenames:
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(foldername, filename)
                npy_path = os.path.join(VECTORS_PATH, f"{filename}_embeddings.npy")

                # Пропускаем обработку, если эмбеддинги уже существуют
                if os.path.exists(npy_path):
                    print(f"PDF {filename} уже обработан. Пропуск...")
                    continue

                print(f"Обработка PDF: {pdf_path}")
                
                images = pdf_to_images(pdf_path)
                pdf_embeddings = []

                for i, image in enumerate(images):
                    print(f"  Обработка страницы {i+1}/{len(images)}")
                    embedding = get_embedding(image)
                    if embedding is not None:
                        pdf_embeddings.append(embedding)
                        file_paths[len(file_paths)] = {
                            "pdf": pdf_path,
                            "page": i
                        }
                    
                        # Сохранение пути после каждой страницы
                        with open(json_path, "w", encoding='utf-8') as f:
                            json.dump(file_paths, f, ensure_ascii=False, indent=4)
                
                # Сохранение эмбеддингов после обработки PDF
                if pdf_embeddings:
                    pdf_embeddings = np.array(pdf_embeddings)
                    np.save(npy_path, pdf_embeddings)
                    print(f"Эмбеддинги для {filename} сохранены.")
                
    print("PDF обработаны. Начинаем создание FAISS индекса...")

    # Объединение всех эмбеддингов и создание FAISS индекса
    embeddings = []
    for npy_file in os.listdir(VECTORS_PATH):
        if npy_file.endswith("_embeddings.npy"):
            embeddings.append(np.load(os.path.join(VECTORS_PATH, npy_file)))

    if embeddings:
        embeddings = np.vstack(embeddings)
        np.save(os.path.join(VECTORS_PATH, "embeddings.npy"), embeddings)

        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        faiss_index_path = os.path.join(VECTORS_PATH, "faiss_index.index")
        faiss.write_index(index, faiss_index_path)
        print(f"FAISS индекс создан и сохранён в {faiss_index_path}")
    else:
        print("Эмбеддинги не загружены. FAISS индекс не создан.")

    print("Обработка завершена!")

process_pdfs()
