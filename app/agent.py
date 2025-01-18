from app.retriever import search_milvus
from app.generator import generate_response

class RAGAgent:
    def __init__(self):
        self.history = []

    async def run(self, query: str):
        # 1. Search in Milvus
        search_results = search_milvus(query)

        if not search_results:
            return "Unfortunately, nothing was found in the cheat sheets."

        # 2. Collect context from retrieved documents
        context = "\n".join([
            result.entity.get("text", "") for result in search_results
        ])

        # 3. Generate answer
        response = generate_response(context, query)

        # 4. Save query-response history
        self.history.append({"query": query, "response": response})

        return response