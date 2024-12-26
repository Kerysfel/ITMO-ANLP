import os
import json
import faiss
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import torch
import requests

# =====================
# РАЗРЕШАЕМ ДУБЛИРОВАНИЕ OpenMP
# =====================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
print("KMP_DUPLICATE_LIB_OK временно установлен в TRUE.")

try:
    # Пути к индексам и эмбеддингам
    VECTORS_PATH = 'vectors/'
    OUTPUT_PATH = 'output/'
    PDF_PATH = 'archive/'

    # =====================
    # НАСТРОЙКА КВАНТИЗАЦИИ (4-битная загрузка)
    # =====================
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    # Загрузка Qwen2-VL с Flash Attention 2
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2"  # Ускорение через Flash Attention 2
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    # Функция загрузки FAISS индексов
    def load_faiss_index(index_type):
        if index_type == "expanded":
            return faiss.read_index(os.path.join(VECTORS_PATH, "faiss_index_expanded.index"))
        else:
            return faiss.read_index(os.path.join(VECTORS_PATH, "faiss_index_pooled.index"))

    # Загрузка путей к PDF (file_paths.json)
    with open(os.path.join(OUTPUT_PATH, "file_paths.json"), "r", encoding='utf-8') as f:
        file_paths = json.load(f)

    PAGE_LIMIT = 5

    # Функция поиска в FAISS
    def search_faiss(query_embedding, index, top_k=50):
        if len(query_embedding.shape) > 2:
            query_embedding = query_embedding.reshape(1, -1)

        print("Размерность FAISS индекса:", index.d)
        print("Размерность текстового вектора (после reshape):", query_embedding.shape)

        if query_embedding.shape[1] != index.d:
            print(f"Размерности не совпадают: {query_embedding.shape[1]} vs {index.d}")
            query_embedding = query_embedding.reshape(-1, index.d)
            distances, indices = index.search(query_embedding, top_k)
            unique_indices = set(indices.flatten())
            results = [file_paths[str(idx)] for idx in unique_indices if str(idx) in file_paths]
            print(f"Найдено {len(results)} уникальных результатов.")
            return results[:PAGE_LIMIT]

        distances, indices = index.search(query_embedding, top_k)
        results = [file_paths[str(idx)] for idx in indices[0]]
        return results[:PAGE_LIMIT]

    # Векторизация текстового запроса через ColPali API
    def get_text_embedding(text):
        API_URL = "http://127.0.0.1:5000/embed/text"
        response = requests.post(API_URL, json={"queries": [text]})
        if response.status_code == 200:
            return np.array(response.json()["embeddings"]).astype(np.float32)
        else:
            print("Ошибка при векторизации текста:", response.text)
            return None

    # Извлечение изображения страницы из PDF
    def extract_page_image(pdf_path, page_number):
        pdf_document = fitz.open(pdf_path)
        page = pdf_document.load_page(page_number)
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img

    # Обработка страницы через Qwen2-VL
    def process_with_qwen2(image, query, pdf_path, page_number):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": query}
                ]
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda")

        print(f"Генерация ответа по файлу: {pdf_path}, страница {page_number}")
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256)  # Увеличиваем число токенов
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        torch.cuda.empty_cache()
        print("Ответ Qwen2-VL:", output_text[0])

    # Основной процесс
    def search_and_process(query, index_type="expanded"):
        query_embedding = get_text_embedding(query)
        if query_embedding is None:
            return

        index = load_faiss_index(index_type)
        results = search_faiss(query_embedding, index)

        for result in results[:PAGE_LIMIT]:
            pdf_path = result["pdf"]
            page_number = result["page"]
            image = extract_page_image(pdf_path, page_number)
            process_with_qwen2(image, query, pdf_path, page_number)

    search_and_process("Explain how Convolutional Neural Networks (CNN) work", index_type="expanded")

except Exception as e:
    print(f"Ошибка: {e}")

finally:
    os.environ.pop("KMP_DUPLICATE_LIB_OK", None)
    print("KMP_DUPLICATE_LIB_OK удалён после завершения программы.")