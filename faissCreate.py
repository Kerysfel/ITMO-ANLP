import os
import numpy as np
import faiss

VECTORS_PATH = 'vectors/'

def create_faiss_index():
    embeddings_expanded = []
    embeddings_pooled = []

    for npy_file in os.listdir(VECTORS_PATH):
        if npy_file.endswith("_embeddings.npy"):
            data = np.load(os.path.join(VECTORS_PATH, npy_file))

            # Вариант 1: Полное развертывание (все векторы)
            expanded_data = data.reshape(-1, data.shape[-1])
            embeddings_expanded.append(expanded_data)

            # Вариант 2: Средний вектор на страницу (Pooling)
            pooled_data = np.mean(data, axis=2).squeeze(axis=1)
            embeddings_pooled.append(pooled_data)

    # Создание FAISS индекса (развёрнутые векторы)
    if embeddings_expanded:
        embeddings_expanded = np.vstack(embeddings_expanded)
        np.save(os.path.join(VECTORS_PATH, "embeddings_expanded.npy"), embeddings_expanded)
        print(f"[INFO] Полное развертывание: {embeddings_expanded.shape}")

        d = embeddings_expanded.shape[1]
        index_expanded = faiss.IndexFlatL2(d)
        index_expanded.add(embeddings_expanded)
        
        faiss.write_index(index_expanded, os.path.join(VECTORS_PATH, "faiss_index_expanded.index"))
        print("FAISS индекс (развёрнутые векторы) создан и сохранён.")

    # Создание FAISS индекса (усреднённые векторы)
    if embeddings_pooled:
        embeddings_pooled = np.vstack(embeddings_pooled)
        np.save(os.path.join(VECTORS_PATH, "embeddings_pooled.npy"), embeddings_pooled)
        print(f"[INFO] Усреднённые векторы: {embeddings_pooled.shape}")

        d = embeddings_pooled.shape[1]
        index_pooled = faiss.IndexFlatL2(d)
        index_pooled.add(embeddings_pooled)
        
        faiss.write_index(index_pooled, os.path.join(VECTORS_PATH, "faiss_index_pooled.index"))
        print("FAISS индекс (усреднённые векторы) создан и сохранён.")
    else:
        print("Нет данных для создания FAISS индекса.")

create_faiss_index()
