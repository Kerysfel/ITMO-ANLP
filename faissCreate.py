import os
import numpy as np
import faiss

VECTORS_PATH = 'vectors/'

def create_faiss_index():
    embeddings = []

    # Читаем все npy файлы и проверяем их форму
    for npy_file in os.listdir(VECTORS_PATH):
        if npy_file.endswith("_embeddings.npy"):
            data = np.load(os.path.join(VECTORS_PATH, npy_file))

            # Проверка размерности (выравниваем если необходимо)
            if len(data.shape) == 3:
                data = data.reshape(-1, data.shape[-1])  # Преобразуем в (N, D)
            
            embeddings.append(data)

    # Склеиваем все эмбеддинги в один массив
    if embeddings:
        embeddings = np.vstack(embeddings)
        np.save(os.path.join(VECTORS_PATH, "embeddings.npy"), embeddings)
        print(f"Общий размер эмбеддингов: {embeddings.shape}")

        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        faiss_index_path = os.path.join(VECTORS_PATH, "faiss_index.index")
        faiss.write_index(index, faiss_index_path)
        print(f"FAISS индекс создан и сохранён в {faiss_index_path}")
    else:
        print("Нет данных для создания FAISS индекса.")

create_faiss_index()