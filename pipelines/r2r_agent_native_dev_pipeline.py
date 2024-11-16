"""
title: R2R Native Agent Pipeline
author: Evgeny A.
author_url: https://github.com/open-webui
funding_url: https://github.com/open-webui
version: 0.0.0.001a
requirements: r2r, langchain_community, duckduckgo-search
"""

from typing import Dict, Any, List, Optional, Callable, Union, AsyncGenerator
import asyncio
import logging
import os
import httpx
import json
from pydantic import BaseModel, Field
from r2r import R2RClient

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Pipeline:
    class Valves(BaseModel):
        API_BASE_URL: str = Field(
            default="https://api.durka.dndstudio.ru",
            description="Base API URL",
        )
        TIMEOUT: int = Field(
            default=180,  # Увеличен таймаут
            description="Request timeout in seconds",
        )
        summarizer_model_id: str = Field(
            default="gpt-4o-mini",
            description="Summarization model ID",
        )

    class UserValves(BaseModel):
        use_hybrid_search: bool = Field(
            default=True,
            description="Enable or disable hybrid search",
        )
        temperature: float = Field(
            default=0.5,
            description="Generation temperature",
        )
        top_p: float = Field(
            default=0.9,
            description="Top-p parameter for generation",
        )
        max_tokens: int = Field(
            default=4096,
            description="Maximum number of tokens for generation",
        )
        memory_limit: int = Field(
            default=10,
            description="Number of recent messages to process",
        )

    def __init__(self):
        self.type = "pipe"
        self.name = "R2R Agent Pipeline"
        self.valves = self.Valves(
            API_BASE_URL=os.getenv(
                "API_BASE_URL", "https://api.durka.dndstudio.ru"
            ),
            TIMEOUT=int(os.getenv("TIMEOUT", "180")),
            summarizer_model_id=os.getenv("SUMMARIZER_MODEL_ID", "gpt-4o"),
        )
        self.user_valves = self.UserValves()
        self.client = httpx.AsyncClient(
            timeout=self.valves.TIMEOUT, verify=False
        )

    def pipelines(self) -> List[str]:
        return []

    async def pipe(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> Union[str, AsyncGenerator[str, None]]:
        stream = body.get("kwargs", {}).get("stream", False)
        if stream:
            return self.pipe_stream(body, __event_emitter__)
        else:
            return await self.pipe_non_stream(body, __event_emitter__)

    async def pipe_stream(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> AsyncGenerator[str, None]:
        try:
            await self.emit_status(
                __event_emitter__,
                "Starting RAG Pipeline (Streaming)...",
                done=False,
            )
            messages = body.get("messages", [])
            if not messages:
                raise ValueError("Empty 'messages' list in 'body'.")

            memory_limit = self.user_valves.memory_limit
            past_messages = messages[-memory_limit:]
            # Извлекаем последнее сообщение от пользователя
            prompt = ""
            for message in reversed(past_messages):
                if message.get("role") == "user" and "content" in message:
                    content = message["content"]
                    if isinstance(content, str):
                        prompt = content.strip()
                    elif isinstance(content, list):
                        content_list = []
                        for item in content:
                            if isinstance(item, str):
                                content_list.append(item)
                            elif isinstance(item, dict):
                                item_content = item.get("content", "")
                                if isinstance(item_content, str):
                                    content_list.append(item_content)
                        prompt = " ".join(content_list).strip()
                    elif isinstance(content, dict):
                        prompt = content.get("content", "").strip()
                    if prompt:
                        break  # Выходим из цикла, если нашли prompt
            if not prompt:
                raise ValueError("Prompt cannot be empty!")

            API_BASE_URL = self.valves.API_BASE_URL
            summarizer_model_id = self.valves.summarizer_model_id
            endpoint = f"{API_BASE_URL}/v2/rag"
            headers = {"Content-Type": "application/json"}
            payload = {
                "query": prompt,
                "vector_search_settings": {
                    "use_vector_search": True,
                    "use_hybrid_search": True,  # self.user_valves.use_hybrid_search,
                    "filters": {},
                    "search_limit": 10,
                    "index_measure": "cosine_distance",
                    "include_values": True,
                    "include_metadatas": True,
                    "probes": 10,
                    "ef_search": 40,
                    "hybrid_search_settings": {
                        "full_text_weight": 1.0,
                        "semantic_weight": 5.0,
                        "full_text_limit": 500,
                        "rrf_k": 50,
                    },
                    "search_strategy": "hyde",
                },
                "rag_generation_config": {
                    "model": summarizer_model_id,
                    "temperature": self.user_valves.temperature,
                    "top_p": self.user_valves.top_p,
                    "max_tokens_to_sample": self.user_valves.max_tokens,
                    "stream": True,  # Убедитесь, что потоковая генерация включена
                },
                "task_prompt_override": None,
                "include_title_if_available": False,
            }
            logger.debug(
                f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}"
            )
            await self.emit_status(
                __event_emitter__, "Sending request to RAG API...", done=False
            )
            await self.emit_message(
                __event_emitter__, "Streaming assistant's response..."
            )
            async with self.client.stream(
                "POST", endpoint, json=payload, headers=headers
            ) as streamed_response:
                if streamed_response.status_code != 200:
                    error_text = await streamed_response.aread()
                    raise httpx.HTTPStatusError(
                        f"Unexpected status code {streamed_response.status_code}: {error_text.decode()}",
                        request=streamed_response.request,
                        response=streamed_response,
                    )
                async for line in streamed_response.aiter_lines():
                    if line:
                        logger.debug(f"Streamed line: {line}")
                        # Предполагаем, что строка - это JSON, пытаемся его разобрать
                        try:
                            data = json.loads(line)
                            # Извлекаем контент из данных
                            content = (
                                data.get("choices", [{}])[0]
                                .get("delta", {})
                                .get("content", "")
                            )
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            # Если не JSON, возвращаем саму строку
                            yield line
            await self.emit_status(
                __event_emitter__,
                "Completed streaming data to user.",
                done=True,
            )
        except httpx.ReadTimeout as e:
            error_message = f"ReadTimeout during streaming: {e}"
            logger.error(error_message)
            await self.emit_status(__event_emitter__, error_message, done=True)
            raise
        except Exception as e:
            error_message = f"Error during streaming: {e}"
            logger.exception(error_message)
            await self.emit_status(__event_emitter__, error_message, done=True)
            raise

    async def pipe_non_stream(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        try:
            await self.emit_status(
                __event_emitter__, "Starting RAG Pipeline...", done=False
            )
            messages = body.get("messages", [])
            if not messages:
                raise ValueError("Empty 'messages' list in 'body'.")

            memory_limit = self.user_valves.memory_limit
            past_messages = messages[-memory_limit:]
            # Извлекаем последнее сообщение от пользователя
            prompt = ""
            for message in reversed(past_messages):
                if message.get("role") == "user" and "content" in message:
                    content = message["content"]
                    if isinstance(content, str):
                        prompt = content.strip()
                    elif isinstance(content, list):
                        content_list = []
                        for item in content:
                            if isinstance(item, str):
                                content_list.append(item)
                            elif isinstance(item, dict):
                                item_content = item.get("content", "")
                                if isinstance(item_content, str):
                                    content_list.append(item_content)
                        prompt = " ".join(content_list).strip()
                    elif isinstance(content, dict):
                        prompt = content.get("content", "").strip()
                    if prompt:
                        break  # Выходим из цикла, если нашли prompt
            if not prompt:
                raise ValueError("Prompt cannot be empty!")

            API_BASE_URL = self.valves.API_BASE_URL
            summarizer_model_id = self.valves.summarizer_model_id
            endpoint = f"{API_BASE_URL}/v2/rag"
            headers = {"Content-Type": "application/json"}
            payload = {
                "query": prompt,
                "vector_search_settings": {
                    "use_vector_search": True,
                    "use_hybrid_search": self.user_valves.use_hybrid_search,
                    "filters": {},
                    "search_limit": 10,
                    "index_measure": "cosine_distance",
                    "include_values": True,
                    "include_metadatas": True,
                    "probes": 10,
                    "ef_search": 40,
                    "hybrid_search_settings": {
                        "full_text_weight": 1.0,
                        "semantic_weight": 5.0,
                        "full_text_limit": 200,
                        "rrf_k": 50,
                    },
                    "search_strategy": "hyde",
                },
                "rag_generation_config": {
                    "model": summarizer_model_id,
                    "temperature": self.user_valves.temperature,
                    "top_p": self.user_valves.top_p,
                    "max_tokens_to_sample": self.user_valves.max_tokens,
                    "stream": False,
                },
                "task_prompt_override": None,
                "include_title_if_available": False,
            }
            logger.debug(
                f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}"
            )
            await self.emit_status(
                __event_emitter__, "Sending request to RAG API...", done=False
            )
            response = await self.client.post(
                url=endpoint, json=payload, headers=headers
            )
            if response.status_code != 200:
                error_text = await response.aread()
                raise httpx.HTTPStatusError(
                    f"Unexpected status code {response.status_code}: {error_text.decode()}",
                    request=response.request,
                    response=response,
                )
            await self.emit_status(
                __event_emitter__,
                "Received response from RAG API.",
                done=False,
            )
            result = response.json()
            logger.debug(
                f"API Response: {json.dumps(result, indent=2, ensure_ascii=False)}"
            )
            completion = result.get("results", {}).get("completion", {})
            choices = completion.get("choices", [])
            if choices and "message" in choices[0]:
                content = choices[0].get("message", {}).get("content", "")
                if isinstance(content, list):
                    # Обрабатываем каждый элемент в списке
                    content_list = []
                    for item in content:
                        if isinstance(item, str):
                            content_list.append(item)
                        elif isinstance(item, dict):
                            item_content = item.get("content", "")
                            if isinstance(item_content, str):
                                content_list.append(item_content)
                    content = " ".join(content_list)
                elif isinstance(content, dict):
                    content = content.get("content", "")
                elif not isinstance(content, str):
                    raise ValueError(
                        "'content' in response must be a string, list, or dict with 'content' key."
                    )
                await self.emit_message(
                    __event_emitter__, "Received assistant response."
                )
                await self.emit_status(
                    __event_emitter__,
                    "RAG Pipeline successfully completed.",
                    done=True,
                )
                return content
            else:
                await self.emit_status(
                    __event_emitter__,
                    "No response from the assistant.",
                    done=True,
                )
                return "No response from the assistant."
        except httpx.ReadTimeout as e:
            error_message = f"ReadTimeout during non-streaming: {e}"
            logger.error(error_message)
            await self.emit_status(__event_emitter__, error_message, done=True)
            raise
        except Exception as e:
            error_message = f"Error during non-streaming: {e}"
            logger.exception(error_message)
            await self.emit_status(__event_emitter__, error_message, done=True)
            raise

    async def emit_status(
        self,
        __event_emitter__: Optional[Callable[[dict], Any]],
        description: str,
        done: bool,
    ):
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {"description": description, "done": done},
                }
            )

    async def emit_message(
        self, __event_emitter__: Optional[Callable[[dict], Any]], content: str
    ):
        if __event_emitter__:
            await __event_emitter__(
                {"type": "message", "data": {"content": content}}
            )


# # Пример использования Pipeline с обработкой событий
# async def example_event_emitter(event: dict):
#     event_type = event.get("type")
#     data = event.get("data", {})
#     if event_type == "status":
#         description = data.get("description", "")
#         done = data.get("done", False)
#         print(f"[STATUS] {description} (Done: {done})")
#     elif event_type == "message":
#         content = data.get("content", "")
#         print(f"[MESSAGE] {content}")


# async def main():
#     pipeline = Pipeline()
#     request_body_stream = {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "Hello, can you tell me about Aristotle?",
#             }
#         ],
#         "kwargs": {"stream": True},
#     }
#     request_body_non_stream = {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "Hello, can you tell me about Aristotle?",
#             }
#         ],
#         "kwargs": {"stream": False},
#     }

#     # Обработка потокового ответа
#     print("=== Streaming Response ===")
#     try:
#         response_stream = await pipeline.pipe(
#             request_body_stream, example_event_emitter
#         )
#         if hasattr(response_stream, "__aiter__"):
#             async for response in response_stream:
#                 print("Streamed Response:", response)
#         else:
#             print("Unexpected response type for streaming.")
#     except Exception as e:
#         print(f"Error during streaming: {e}")

#     # Обработка непотокового ответа
#     print("\n=== Non-Streaming Response ===")
#     try:
#         response = await pipeline.pipe(
#             request_body_non_stream, example_event_emitter
#         )
#         print("Final Response:", response)
#     except Exception as e:
#         print(f"Error during non-streaming: {e}")


# if __name__ == "__main__":
#     asyncio.run(main())
