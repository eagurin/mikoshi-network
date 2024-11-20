"""
title: Memory Module Pipeline
author: Evgeny A.
author_url: https://github.com/open-webui
funding_url: https://github.com/open-webui
version: 0.0.0.001a
requirements: r2r, sentence_transformers, sklearn
"""

from r2r import R2RClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class MemoryModule:
    def __init__(self):
        self.memory = (
            []
        )  # Здесь можно использовать базу данных или векторное хранилище
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def save_memory(self, text):
        embedding = self.embedding_model.encode([text])
        self.memory.append({"text": text, "embedding": embedding})

    def retrieve_memory(self, query, top_k=5):
        query_embedding = self.embedding_model.encode([query])
        similarities = []
        for item in self.memory:
            sim = cosine_similarity(query_embedding, item["embedding"])[0][0]
            similarities.append((sim, item["text"]))
        # Сортируем по убыванию похожести
        similarities.sort(reverse=True)
        # Возвращаем топ-k результатов
        top_texts = [text for _, text in similarities[:top_k]]
        return " ".join(top_texts)


# Инициализация клиентов
r2r_client = R2RClient()
memory_module = MemoryModule()


def process_query(query):
    # Извлекаем релевантную информацию из памяти
    past_info = memory_module.retrieve_memory(query)

    # Формируем расширенный контекст
    augmented_query = f"{past_info} {query}"

    # Получаем ответ от R2R
    response = r2r_client.rag(augmented_query)

    # Сохраняем новый контекст в память
    memory_module.save_memory(response)

    return response


# Пример использования
user_query = "Расскажи мне о нашем предыдущем разговоре о погоде."
answer = process_query(user_query)
print("Ответ:", answer)
