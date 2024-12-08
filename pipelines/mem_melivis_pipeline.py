"""
title: Mem0 Pipeline for Open WebUI
author: Ваше Имя
version: 1.0
description: Pipeline для Open WebUI, реализующий возможности mem0 и соответствующий рекомендациям Open WebUI.
# requirements: llama-index-memory-mem0
"""

from typing import List, Optional
from pydantic import BaseModel
from schemas import OpenAIChatMessage
import os
import requests
import json

# Импортируем необходимые модули для mem0
from llama_index.memory.mem0 import Mem0Memory

from utils.pipelines.main import (
    get_last_user_message,
    add_or_update_system_message,
    get_tools_specs,
)

class Pipeline:
    class Valves(BaseModel):
        # Список целевых pipeline IDs (моделей), к которым будет подключен этот фильтр.
        # Если вы хотите подключить этот фильтр ко всем pipelines, установите pipelines в ["*"]
        pipelines: List[str] = []

        # Назначьте уровень приоритета для фильтра pipeline.
        # Уровень приоритета определяет порядок, в котором выполняются фильтры pipeline.
        # Чем меньше число, тем выше приоритет.
        priority: int = 0

        # Параметры для вызова функций и mem0
        OPENAI_API_BASE_URL: str
        OPENAI_API_KEY: str
        TASK_MODEL: str
        TEMPLATE: str
        MEM0_API_KEY: str

    def __init__(self):
        # Фильтры pipeline совместимы только с Open WebUI
        # Фильтр pipeline можно рассматривать как middleware, которое можно использовать для редактирования данных перед отправкой в OpenAI API.
        self.type = "filter"

        # Опционально вы можете установить id и имя pipeline.
        # Идентификатор должен быть уникальным для всех pipelines.
        # self.id = "function_calling_with_mem0"
        self.name = "Function Calling with mem0 Integration"

        # Инициализируем valves
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],  # Подключить ко всем pipelines
                "OPENAI_API_BASE_URL": os.getenv(
                    "OPENAI_API_BASE_URL", "https://api.openai.com/v1"
                ),
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
                "TASK_MODEL": os.getenv("TASK_MODEL", "gpt-4o-mini"),
                "TEMPLATE": """Use the following context as your learned knowledge, inside <context></context> XML tags.
<context>
{{CONTEXT}}
</context>

When answering the user:
- If you don't know, just say that you don't know.
- If you are not sure, ask for clarification.
Avoid mentioning that you obtained the information from the context.
Answer in the same language as the user's question.""",
                "MEM0_API_KEY": os.getenv("MEM0_API_KEY", "YOUR_MEM0_API_KEY"),
            }
        )

        # Инициализируем Mem0Memory
        self.memory = Mem0Memory(
            api_key=self.valves.MEM0_API_KEY,
            collection_name=os.getenv("MEM0_COLLECTION_NAME", "mem_collection"),
            embedding_model_dims=int(os.getenv("MEM0_EMBEDDING_MODEL_DIMS", "768")),
            vector_store=os.getenv("MEM0_VECTOR_STORE", "milvus"),
            vector_store_kwargs={
                "milvus_uri": os.getenv("MEM0_MILVUS_URI", "http://localhost:19530"),
                "metric_type": os.getenv("MEM0_METRIC_TYPE", "L2"),
            },
        )

    async def on_startup(self):
        # Эта функция вызывается при запуске сервера.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # Эта функция вызывается при остановке сервера.
        print(f"on_shutdown:{__name__}")
        pass

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # Если запрошена генерация заголовка, пропустить фильтр вызова функций
        if body.get("title", False):
            return body

        print(f"pipe:{__name__}")
        print(user)

        # Получаем последнее сообщение пользователя
        user_message = get_last_user_message(body["messages"])

        # Получаем контекст из памяти mem0
        memory_context = ""
        try:
            mem_results = self.memory.query(user_message, top_k=5)
            if mem_results:
                memory_context = "\n".join([res["content"] for res in mem_results])
        except Exception as e:
            print(f"Memory retrieval error: {e}")

        # Обновляем TEMPLATE с контекстом из памяти
        system_prompt = self.valves.TEMPLATE.replace("{{CONTEXT}}", memory_context)

        # Обновляем сообщения
        messages = add_or_update_system_message(
            system_prompt, body["messages"]
        )

        # Получаем спецификации инструментов
        tools_specs = get_tools_specs(self.tools)

        # Системный prompt для вызова функций
        fc_system_prompt = (
            f"Tools: {json.dumps(tools_specs, indent=2)}"
            + """
If a function tool doesn't match the query, return an empty string. Else, pick a function tool, fill in the parameters from the function tool's schema, and return it in the format { "name": \"functionName\", "parameters": { "key": "value" } }. Only pick a function if the user asks. Only return the object. Do not return any other text."
"""
        )

        r = None
        try:
            # Вызываем OpenAI API для получения ответа функции
            r = requests.post(
                url=f"{self.valves.OPENAI_API_BASE_URL}/chat/completions",
                json={
                    "model": self.valves.TASK_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": fc_system_prompt,
                        },
                        {
                            "role": "user",
                            "content": "History:\n"
                            + "\n".join(
                                [
                                    f"{message['role']}: {message['content']}"
                                    for message in body["messages"][::-1][:4]
                                ]
                            )
                            + f"\nQuery: {user_message}",
                        },
                    ],
                },
                headers={
                    "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                stream=False,
            )
            r.raise_for_status()

            response = r.json()
            content = response["choices"][0]["message"]["content"]

            # Разбираем ответ функции
            if content != "":
                result = json.loads(content)
                print(result)

                # Вызываем функцию
                if "name" in result:
                    function = getattr(self.tools, result["name"])
                    function_result = None
                    try:
                        function_result = function(**result["parameters"])
                    except Exception as e:
                        print(e)

                    # Добавляем результат функции в память
                    if function_result:
                        # Добавляем в память mem0
                        self.memory.add(
                            content=function_result,
                            metadata={"user_id": user.get("id", "default_user")},
                        )

                        # Обновляем системный prompt с новым контекстом
                        system_prompt = self.valves.TEMPLATE.replace(
                            "{{CONTEXT}}", function_result
                        )

                        print(system_prompt)
                        messages = add_or_update_system_message(
                            system_prompt, messages
                        )

                        # Возвращаем обновленные сообщения
                        return {**body, "messages": messages}

        except Exception as e:
            print(f"Error: {e}")

            if r:
                try:
                    print(r.json())
                except:
                    pass

        return body
