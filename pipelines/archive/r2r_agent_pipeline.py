# agent_pipeline.py

from typing import List, Dict, Any
import os
import asyncio
from r2r import R2RClient
from fastapi import FastAPI, Body

class R2RAgent:
    def __init__(self, client: R2RClient, max_iterations: int = 5):
        self.client = client
        self.max_iterations = max_iterations

    async def run(self, messages: List[Dict[str, str]]) -> str:
        user_message = messages[-1]['content']

        # Инициализируем контекст
        context = ''
        for iteration in range(self.max_iterations):
            print(f"Итерация {iteration + 1}")

            # Выполняем поиск (векторный, гибридный и графовый)
            search_results = await self.search(user_message)

            # Генерируем ответ на основе результатов поиска
            response = await self.generate_response(user_message, search_results)

            # Проверяем, достаточно ли ответа
            if self.check_completion(response):
                return response
            else:
                # Используем ответ как новый запрос пользователя
                user_message = response

        return response

    async def search(self, query: str) -> str:
        # Выполняем поиски
        vector_results = await self.vector_search(query)
        hybrid_results = await self.hybrid_search(query)
        graph_results = await self.graph_search(query)

        # Объединяем результаты
        combined_results = vector_results + hybrid_results + graph_results

        # Преобразуем результаты в текст
        documents = [doc.get('text', '') for doc in combined_results]
        return "\n".join(documents)

    async def vector_search(self, query: str) -> List[Dict[str, Any]]:
        response = self.client.search(
            query=query,
            vector_search_settings={
                'use_vector_search': True,
                'search_limit': 5,
            }
        )
        return response.get('results', {}).get('vector_search_results', [])

    async def hybrid_search(self, query: str) -> List[Dict[str, Any]]:
        response = self.client.search(
            query=query,
            hybrid_search_settings={
                'use_hybrid_search': True,
                'search_limit': 5,
                'full_text_weight': 1.0,
                'semantic_weight': 5.0,
            }
        )
        return response.get('results', {}).get('hybrid_search_results', [])

    async def graph_search(self, query: str) -> List[Dict[str, Any]]:
        response = self.client.search(
            query=query,
            kg_search_settings={
                'use_kg_search': True,
                'kg_search_type': 'local',
                'kg_search_level': 1,
                'search_limit': 5,
            }
        )
        return response.get('results', {}).get('kg_search_results', [])

    async def generate_response(self, query: str, context: str) -> str:
        prompt = f"""Используя приведенный контекст, ответь на вопрос пользователя.

Контекст:
{context}

Вопрос:
{query}

Ответ:
"""
        generation_config = {
            'model': 'openai/gpt-3.5-turbo',
            'max_tokens': 500,
            'temperature': 0.7,
        }
        completion = self.client.completion(prompt, generation_config)
        response_text = completion.get('choices', [{}])[0].get('text', '').strip()
        return response_text

    def check_completion(self, response: str) -> bool:
        # Простая проверка: завершаем после первой итерации
        return True

class Pipeline:
    def __init__(self):
        self.name = "Iterative R2R Agent Pipeline"
        # Инициализируем R2R клиент
        r2r_base_url = os.getenv('R2R_BASE_URL', 'http://localhost:7272')
        self.r2r_client = R2RClient(base_url=r2r_base_url)
        # Создаем экземпляр агента
        self.agent = R2RAgent(
            client=self.r2r_client,
            max_iterations=5,
        )

    async def on_startup(self):
        # Вызывается при запуске сервера
        print(f"Pipeline {self.name} запущен.")
        # Выполняем вход в R2R, если требуется авторизация
        self.r2r_client.login(
            email='admin@example.com',
            password='change_me_immediately',
        )

    async def on_shutdown(self):
        # Вызывается при остановке сервера
        print(f"Pipeline {self.name} остановлен.")

    async def run(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        # Используем агента для обработки сообщений
        assistant_response = await self.agent.run(messages)
        # Добавляем ответ ассистента в список сообщений
        messages.append({
            'role': 'assistant',
            'content': assistant_response,
        })
        return messages

# Инициализируем FastAPI приложение
app = FastAPI()
pipeline = Pipeline()

@app.on_event("startup")
async def startup_event():
    await pipeline.on_startup()

@app.on_event("shutdown")
async def shutdown_event():
    await pipeline.on_shutdown()

@app.post("/chat")
async def chat_endpoint(body: Dict[str, Any] = Body(...)):
    messages = body.get('messages', [])
    # Обрабатываем сообщения с помощью пайплайна
    response_messages = await pipeline.run(messages)
    # Возвращаем обновленные сообщения
    return {'messages': response_messages}
