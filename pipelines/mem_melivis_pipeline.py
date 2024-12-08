"""
title: Memory Module Pipeline
author: Evgeny A.
author_url: https://github.com/open-webui
funding_url: https://github.com/open-webui
version: 0.0.0.001a
requirements: mem0ai
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import os
import json
import asyncio
from mem0 import Memory
from utils.pipelines.main import (
    get_last_user_message,
    add_or_update_system_message,
)
# Системный промпт для функции
DEFAULT_SYSTEM_PROMPT = (
    """Используй следующий контекст для помощи пользователю:\n{context}"""
)
class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = ["*"]  # Подключиться ко всем пайплайнам
        priority: int = 0
        # Конфигурация mem0 и локального хранилища
        MEM0_VECTOR_STORE: str = os.getenv("MEM0_VECTOR_STORE", "milvus")
        MEM0_VECTOR_STORE_CONFIG: Dict[str, Any] = {
            "collection_name": os.getenv("MEM0_COLLECTION_NAME", "test_collection"),
            "url": os.getenv("MEM0_MILVUS_URI", "http://localhost:19530"),
            "embedding_model_dims": int(os.getenv("MEM0_EMBEDDING_MODEL_DIMS", "768")),
            "metric_type": os.getenv("MEM0_METRIC_TYPE", "L2"),
        }
        MODEL: str = os.getenv("MODEL", "gpt-3.5-turbo")
        TEMPLATE: str = os.getenv(
            "TEMPLATE",
            """Используй следующий контекст для помощи пользователю:\n{context}""",
        )
        MEMORY_LIMIT: int = int(os.getenv("MEMORY_LIMIT", "10"))
        TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
        STREAM: bool = os.getenv("STREAM", "false").lower() == "true"
    def __init__(self, prompt: str | None = None) -> None:
        self.type = "filter"
        self.name = "Mem0 Local Storage Pipeline with Milvus"
        self.prompt = prompt or DEFAULT_SYSTEM_PROMPT
        self.valves = self.Valves()
        # Инициализируем mem0 с локальным хранилищем Milvus
        self.memory = Memory.from_config({
            "vector_store": {
                "provider": self.valves.MEM0_VECTOR_STORE,
                "config": self.valves.MEM0_VECTOR_STORE_CONFIG
            }
        })
        # Инициализируем шаблон промпта
        self.prompt_template = self.valves.TEMPLATE
    async def on_startup(self):
        print(f"Пайплайн '{self.name}' запускается.")
    async def on_shutdown(self):
        print(f"Пайплайн '{self.name}' останавливается.")
    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if body.get("title", False):
            return body
        print(f"Обработка сообщения через пайплайн '{self.name}'.")
        # Извлекаем последние N сообщений на основе лимита памяти
        past_messages = body["messages"][-self.valves.MEMORY_LIMIT :]
        # Получаем последнее сообщение пользователя
        user_message = get_last_user_message(past_messages)
        if not user_message:
            return body
        # Сохраняем сообщение пользователя в локальное хранилище mem0
        self.memory.add(user_message, user_id="user")
        # Ищем связанные воспоминания
        related_memories = self.memory.search(user_message, user_id="user")
        # Формируем контекст из полученных воспоминаний
        context = "\n".join([memory['text'] for memory in related_memories])
        # Строим промпт
        prompt = self.prompt_template.format(context=context)
        # Генерируем ответ (здесь используется ваша модель)
        agent_response = await self.generate_response(prompt)
        if not agent_response:
            return body
        # Добавляем или обновляем системное сообщение с ответом агента
        messages = add_or_update_system_message(agent_response, body["messages"])
        return {**body, "messages": messages}
    async def generate_response(self, prompt: str) -> Optional[str]:
        try:
            # Здесь вы вызываете вашу модель для генерации ответа
            # Например, если используете локальную модель:
            # import your_local_model
            # response = your_local_model.generate(prompt)
            # return response.strip()
            # В примере используется OpenAI API:
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            response = openai.ChatCompletion.create(
                model=self.valves.MODEL,
                messages=[{"role": "system", "content": prompt}],
                temperature=self.valves.TEMPERATURE,
                stream=self.valves.STREAM,
            )
            if self.valves.STREAM:
                content = ""
                for chunk in response:
                    content += chunk.choices[0].delta.get('content', '')
                return content.strip()
            else:
                return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Ошибка при генерации ответа: {e}")
            return None
