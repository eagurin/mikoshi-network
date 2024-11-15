# agent_pipeline.py

from typing import List, Dict, Any
import os
import asyncio
from r2r import R2RClient
from r2r.config import R2RConfig
from r2r.providers.llm import OpenAICompletionProvider
from r2r.providers.database import PostgresDatabaseProvider
from r2r.agent import Agent
from fastapi import FastAPI, Body

class MyR2RAgent(Agent):
    def __init__(self, llm_provider, database_provider, config, max_iterations=5):
        super().__init__(
            llm_provider=llm_provider,
            database_provider=database_provider,
            config=config,
        )
        self.max_iterations = max_iterations
        self.r2r_client = R2RClient(base_url=config.api_base_url)
        self._register_tools()

    def _register_tools(self):
        # Регистрируем инструменты, которые агент может использовать
        self.add_tool(
            name="search",
            description="Выполняет векторный, гибридный и графовый поиск.",
            func=self.tool_search,
        )

    async def arun(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        # Основной асинхронный метод для обработки сообщений
        user_message = messages[-1]["content"]
        # Инициализируем контекст
        for iteration in range(self.max_iterations):
            print(f"Итерация {iteration + 1}")
            # Агент выполняет поиск
            search_results = await self.tool_search(user_message)
            # Генерируем ответ на основе результатов поиска
            response = await self.generate_response(user_message, search_results)
            # Проверяем, нужно ли завершить итерации
            if self.check_completion(response):
                return response
            else:
                # Используем ответ как новый запрос пользователя
                user_message = response
        return response

    async def process_llm_response(self, response: str) -> str:
        # Обрабатываем ответ LLM, если необходимо
        return response

    async def tool_search(self, query: str) -> str:
        # Выполняем векторный, гибридный и графовый поиск
        search_response = await self.r2r_client.search_async(
            query=query,
            vector_search_settings={
                "use_vector_search": True,
                "search_limit": 5,
            },
            hybrid_search_settings={
                "use_hybrid_search": True,
                "hybrid_search_settings": {
                    "full_text_weight": 1.0,
                    "semantic_weight": 5.0,
                    "full_text_limit": 200,
                    "rrf_k": 50,
                },
                "search_limit": 5,
            },
            kg_search_settings={
                "use_kg_search": True,
                "kg_search_type": "local",
                "kg_search_level": 1,
                "search_limit": 5,
            },
        )
        # Объединяем результаты
        results = search_response.get("results", {})
        combined_results = []
        for key in [
            "vector_search_results",
            "hybrid_search_results",
            "kg_search_results",
        ]:
            combined_results.extend(results.get(key, []))
        # Извлекаем текстовое содержимое
        documents = [doc.get("text", "") for doc in combined_results]
        return "\n".join(documents)

    async def generate_response(self, query: str, context: str) -> str:
        # Используем контекст для генерации ответа
        prompt = f"""Используя приведенный контекст, ответь на вопрос пользователя.

Контекст:
{context}

Вопрос:
{query}

Ответ:"""
        generation_config = {
            "model": "openai/gpt-3.5-turbo",
            "max_tokens": 500,
            "temperature": 0.7,
        }
        completion = await self.r2r_client.completion_async(prompt, generation_config)
        response_text = (
            completion.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        return response_text

    def check_completion(self, response: str) -> bool:
        # Простая реализация: завершаем после первой итерации
        return True

class Pipeline:
    def __init__(self):
        self.name = "Iterative R2R Agent Pipeline"
        # Загружаем конфигурацию R2R
        self.config = R2RConfig.from_toml("r2r.toml")
        # Инициализируем провайдеры
        self.llm_provider = OpenAICompletionProvider(self.config.completion)
        self.database_provider = PostgresDatabaseProvider(self.config.database)
        # Создаем экземпляр агента
        self.agent = MyR2RAgent(
            llm_provider=self.llm_provider,
            database_provider=self.database_provider,
            config=self.config,
            max_iterations=5,
        )

    async def on_startup(self):
        # Вызывается при запуске сервера
        print(f"Пайплайн {self.name} запущен.")
        # Выполняем вход в R2R, если требуется авторизация
        await self.agent.r2r_client.login_async(
            email="admin@example.com",
            password="change_me_immediately",
        )

    async def on_shutdown(self):
        # Вызывается при остановке сервера
        print(f"Пайплайн {self.name} остановлен.")

    async def run(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        # Используем агента для обработки сообщений
        assistant_response = await self.agent.arun(messages)
        # Добавляем ответ ассистента в список сообщений
        messages.append(
            {
                "role": "assistant",
                "content": assistant_response,
            }
        )
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
    messages = body.get("messages", [])
    # Обрабатываем сообщения с помощью пайплайна
    response_messages = await pipeline.run(messages)
    # Возвращаем обновленные сообщения
    return {"messages": response_messages}
