from typing import List, Dict, Any, Optional
from schemas import OpenAIChatMessage
import asyncio
import json
import logging
from r2r import R2RClient

logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self):
        self.type = "manifold"  # Указываем, что это манифолдный pipeline
        self.name = "R2R Agent Pipeline"

    def pipelines(self):
        """
        Метод pipelines должен быть вызываемым и возвращать список связанных pipelines.
        Если нет связанных pipelines, можно вернуть пустой список или другую подходящую структуру.
        """
        return []  # Здесь можно указать имена связанных pipelines, если они есть

    async def run(
        self,
        body: Dict[str, Any],
        websocket_conn=None,
        request=None,
    ):
        client = R2RClient("https://api.durka.dndstudio.ru")  # URL вашего R2R сервера
        messages = body.get("messages", [])

        # Формируем историю сообщений для агента R2R
        conversation = [
            {"role": message["role"], "content": message["content"]}
            for message in messages
        ]

        # Настраиваем параметры генерации
        rag_generation_config = {
            "max_tokens": 300,
            "stream": True  # Установите True для потоковой генерации
        }

        try:
            # Проверяем, есть ли асинхронный метод в R2RClient
            if hasattr(client, 'agent_async') and asyncio.iscoroutinefunction(client.agent_async):
                response = await client.agent_async(
                    messages=conversation,
                    vector_search_settings={
                        "search_limit": 5,
                        "filters": {}
                    },
                    kg_search_settings={
                        "use_kg_search": True
                    },
                    rag_generation_config=rag_generation_config
                )
            else:
                # Используем asyncio.to_thread для вызова синхронного метода
                response = await asyncio.to_thread(
                    client.agent,
                    messages=conversation,
                    vector_search_settings={
                        "search_limit": 5,
                        "filters": {}
                    },
                    kg_search_settings={
                        "use_kg_search": True
                    },
                    rag_generation_config=rag_generation_config
                )

            if rag_generation_config.get("stream", False):
                # Обработка потокового ответа
                content = ""
                async for chunk in response:
                    content += chunk
                return {"response": content}
            else:
                # Обработка стандартного ответа
                assistant_response = response.get("results", [])
                # Извлекаем последнее сообщение от ассистента
                assistant_messages = [msg for msg in assistant_response if msg.get('role') == 'assistant']
                if assistant_messages:
                    content = assistant_messages[-1].get('content', '')
                    return {"response": content}
                else:
                    return {"response": "Извините, не удалось получить ответ от агента R2R."}

        except Exception as e:
            logger.exception("Ошибка при обращении к агенту R2R.")
            return {"response": f"Ошибка при обращении к агенту R2R: {str(e)}"}
