import os
import json
import faiss
import numpy as np
import requests

# Пути к индексам и файлам
VECTORS_PATH = 'vectors/'
OUTPUT_PATH = 'output/'

# Загрузка FAISS индекса (pooled)
index = faiss.read_index(os.path.join(VECTORS_PATH, "faiss_index_pooled.index"))

# Загрузка путей к PDF (file_paths.json)
with open(os.path.join(OUTPUT_PATH, "file_paths.json"), "r", encoding='utf-8') as f:
    file_paths = json.load(f)


# Векторизация текстового запроса через ColPali API
def get_text_embedding(text):
    API_URL = "http://127.0.0.1:5000/embed/text"
    response = requests.post(API_URL, json={"queries": [text]})
    if response.status_code == 200:
        return np.array(response.json()["embeddings"]).astype(np.float32)
    else:
        print("Ошибка при векторизации текста:", response.text)
        return None


# Функция поиска в FAISS и вывода результатов
def search_faiss_top_k(query_embedding, index, top_k=15):
    print("Размерность FAISS индекса:", index.d)
    print("Размерность текстового вектора:", query_embedding.shape)

    # Приведение к (1, 128)
    if len(query_embedding.shape) == 3:
        if query_embedding.shape[1] > 1:
            print(f"Вектор имеет {query_embedding.shape[1]} подвекторов. Усредняем...")
            query_embedding = query_embedding.mean(axis=1)  # (1, 128)
        else:
            query_embedding = query_embedding.reshape(1, -1)  # (1, 128)

    print("Форма вектора после приведения:", query_embedding.shape)

    # Поиск в FAISS
    distances, indices = index.search(query_embedding, top_k)
    
    print("\n=== Топ-15 страниц из FAISS ===")
    for i, idx in enumerate(indices[0]):
        if str(idx) in file_paths:
            result = file_paths[str(idx)]
            print(f"{i+1}. PDF: {result['pdf']}, Страница: {result['page']}, Distance: {distances[0][i]:.4f}")
        else:
            print(f"{i+1}. [ОШИБКА] Индекс {idx} не найден в file_paths.json")


# Основной процесс
def main():
    query = input("Введите текстовый запрос: ")
    query_embedding = get_text_embedding(query)

    if query_embedding is not None:
        search_faiss_top_k(query_embedding, index, top_k=15)
    else:
        print("Не удалось получить эмбеддинг запроса.")


if __name__ == "__main__":
    main()
