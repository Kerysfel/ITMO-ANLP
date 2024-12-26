from app.retriever import search_milvus
from app.generator import generate_response

class RAGAgent:
    def __init__(self):
        self.history = []

    async def run(self, query: str):
        # 1. Поиск по Milvus
        search_results = search_milvus(query)
        
        if not search_results:
            return "К сожалению, ничего не найдено в cheat sheets."

        # 2. Сбор контекста из найденных документов
        context = "\n".join([result.entity.get("text", "") for result in search_results])

        # 3. Генерация ответа на основе контекста
        response = generate_response(context, query)
        
        # 4. Сохранение истории для дальнейшего анализа
        self.history.append({"query": query, "response": response})
        
        return response