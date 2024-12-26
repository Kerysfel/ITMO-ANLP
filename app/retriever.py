from pymilvus import Collection
from app.models.colpali import ColPaliModel

colpali = ColPaliModel()
COLLECTION_NAME = "cheat_sheets"

def search_milvus(query):
    collection = Collection(COLLECTION_NAME)
    query_embedding = colpali.embed_text(query)
    
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=5,
    )
    return results