"""
title: Mem0 Pipeline for Open WebUI
author: Ваше Имя
version: 1.0
description: Pipeline для Open WebUI, реализующий возможности mem0 и соответствующий рекомендациям Open WebUI.
requirements: mem0ai
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import os
from mem0 import Memory

# Импорт необходимых функций из Open WebUI
from utils.pipelines.main import (
    get_last_user_message,
    add_or_update_system_message,
)

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = ["*"]  # Применяется ко всем пайплайнам
        priority: int = 0  # Приоритет выполнения

        # Настройки mem0
        MEM0_VECTOR_STORE: str = os.getenv("MEM0_VECTOR_STORE", "milvus")
        MEM0_VECTOR_STORE_CONFIG: Dict[str, Any] = {
            "collection_name": os.getenv("MEM0_COLLECTION_NAME", "mem_collection"),
            "url": os.getenv("MEM0_MILVUS_URI", "http://milvus:19530"),
            "embedding_model_dims": int(os.getenv("MEM0_EMBEDDING_MODEL_DIMS", "768")),
            "metric_type": os.getenv("MEM0_METRIC_TYPE", "L2"),
        }
        MODEL: str = os.getenv("MEM0_MODEL", "gpt-3.5-turbo")
        TEMPLATE: str = os.getenv(
            "MEM0_TEMPLATE",
            """Используй следующий контекст для помощи пользователю:\n{context}""",
        )
        MEMORY_LIMIT: int = int(os.getenv("MEM0_MEMORY_LIMIT", "10"))
        TEMPERATURE: float = float(os.getenv("MEM0_TEMPERATURE", "0.7"))
        STREAM: bool = os.getenv("MEM0_STREAM", "false").lower() == "true"
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    def __init__(self):
        # Указываем, что это фильтр-пайплайн для Open WebUI
        self.type = "filter"
        self.name = "Mem0 Integration Pipeline"
        self.valves = self.Valves()

        # Инициализация mem0 с заданной конфигурацией
        self.memory = Memory.from_config({
            "vector_store": {
                "provider": self.valves.MEM0_VECTOR_STORE,
                "config": self.valves.MEM0_VECTOR_STORE_CONFIG
            }
        })

        # Шаблон для системного сообщения
        self.prompt_template = self.valves.TEMPLATE

    async def on_startup(self):
        print(f"Пайплайн '{self.name}' был запущен.")

    async def on_shutdown(self):
        print(f"Пайплайн '{self.name}' был остановлен.")

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # Пропускаем обработку, если запрошена генерация заголовка
        if body.get("title", False):
            return body
        print(f"Обработка сообщения через пайплайн '{self.name}'.")
        # Извлекаем последние N сообщений на основе лимита памяти
        past_messages = body["messages"][-self.valves.MEMORY_LIMIT :]
        # Получаем последнее сообщение пользователя
        user_message = get_last_user_message(past_messages)
        if not user_message:
            return body
        user_message_content = user_message["content"]
        # Сохраняем сообщение пользователя в локальное хранилище mem0
        self.memory.add(user_message_content, user_id="user")
        # Ищем связанные воспоминания
        related_memories = self.memory.search(user_message_content, user_id="user")
        # Формируем контекст из полученных воспоминаний
        context = "\n".join([memory['text'] for memory in related_memories])
        # Строим промпт
        prompt = self.prompt_template.format(context=context)
        # Генерируем ответ с использованием модели
        agent_response = await self.generate_response(prompt)
        if not agent_response:
            return body
        # Добавляем или обновляем системное сообщение с ответом агента
        messages = add_or_update_system_message(agent_response, body["messages"])
        return {**body, "messages": messages}

    async def generate_response(self, prompt: str) -> Optional[str]:
        try:
            # Используем OpenAI API для генерации ответа
            openai.api_key = self.valves.OPENAI_API_KEY
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
