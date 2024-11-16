"""
title: R2R Native Agent Pipeline
author: Evgeny A.
author_url: https://github.com/open-webui
funding_url: https://github.com/open-webui
version: 0.0.0.001a
requirements: r2r, langchain_community, duckduckgo-search, httpx, pydantic
"""

from typing import Dict, Any, List, Optional, Callable, Union, AsyncGenerator
import asyncio
import logging
import os
import httpx
import json
from pydantic import BaseModel, Field

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetryTransport(httpx.AsyncBaseTransport):
    def __init__(self, retries: int = 3, delay: float = 1.0, **kwargs):
        self.retries = retries
        self.delay = delay
        self.transport = httpx.AsyncHTTPTransport(**kwargs)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        last_exception = None
        for attempt in range(1, self.retries + 1):
            try:
                logger.debug(f"Attempt {attempt} for {request.method} {request.url}")
                response = await self.transport.handle_async_request(request)
                return response
            except (httpx.ReadTimeout, httpx.ConnectError) as e:
                last_exception = e
                if attempt < self.retries:
                    logger.warning(
                        f"Attempt {attempt} failed with error: {e}. Retrying in {self.delay} seconds..."
                    )
                    await asyncio.sleep(self.delay)
                else:
                    logger.error(f"All {self.retries} attempts failed.")
                    raise last_exception

class Pipeline:
    class Valves(BaseModel):
        API_BASE_URL: str = Field(
            default="https://api.durka.dndstudio.ru",
            description="Base API URL",
        )
        TIMEOUT: int = Field(
            default=300,  # Увеличен таймаут до 300 секунд
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
            API_BASE_URL=os.getenv("API_BASE_URL", "https://api.durka.dndstudio.ru"),
            TIMEOUT=int(os.getenv("TIMEOUT", "300")),
            summarizer_model_id=os.getenv("SUMMARIZER_MODEL_ID", "gpt-4o"),
        )
        self.user_valves = self.UserValves()
        self.client = httpx.AsyncClient(
            timeout=self.valves.TIMEOUT,
            verify=False,  # Отключение проверки SSL сертификатов
            transport=RetryTransport(retries=3, delay=2.0)
        )

    def pipelines(self) -> List[str]:
        return []

    async def pipe(self, body: dict, __event_emitter__: Optional[Callable[[dict], Any]] = None) -> Union[str, AsyncGenerator[str, None]]:
        stream = body.get("kwargs", {}).get("stream", False)
        if stream:
            return self.pipe_stream(body, __event_emitter__)
        else:
            return await self.pipe_non_stream(body, __event_emitter__)

    async def pipe_stream(self, body: dict, __event_emitter__: Optional[Callable[[dict], Any]] = None) -> AsyncGenerator[str, None]:
        try:
            logger.info("Starting RAG Pipeline (Streaming)...")
            await self.emit_status(__event_emitter__, "Starting RAG Pipeline (Streaming)...", done=False)

            messages = body.get("messages", [])
            if not messages:
                raise ValueError("Empty 'messages' list in 'body'.")

            past_messages = messages[-self.user_valves.memory_limit:]
            prompt = self.extract_latest_prompt(past_messages)
            if not prompt:
                raise ValueError("Prompt cannot be empty!")

            payload = self.construct_payload(prompt, streaming=True)
            logger.debug(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
            await self.emit_status(__event_emitter__, "Sending request to RAG API...", done=False)
            await self.emit_message(__event_emitter__, "Streaming assistant's response...")

            async with self.client.stream("POST", f"{self.valves.API_BASE_URL}/v2/rag", json=payload, headers={"Content-Type": "application/json"}) as streamed_response:
                logger.info(f"Received response with status code {streamed_response.status_code}")

                if streamed_response.status_code != 200:
                    error_text = await streamed_response.aread()
                    logger.error(f"Error response: {error_text.decode()}")
                    raise httpx.HTTPStatusError(f"Unexpected status code {streamed_response.status_code}: {error_text.decode()}", request=streamed_response.request, response=streamed_response)

                async for line in streamed_response.aiter_lines():
                    if line:
                        logger.debug(f"Streamed line: {line}")
                        try:
                            data = json.loads(line)
                            content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            logger.warning(f"Non-JSON line received: {line}")
                            yield line

            logger.info("Completed streaming data to user.")
            await self.emit_status(__event_emitter__, "Completed streaming data to user.", done=True)

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

    async def pipe_non_stream(self, body: dict, __event_emitter__: Optional[Callable[[dict], Any]] = None) -> str:
        try:
            logger.info("Starting RAG Pipeline (Non-Streaming)...")
            await self.emit_status(__event_emitter__, "Starting RAG Pipeline...", done=False)

            messages = body.get("messages", [])
            if not messages:
                raise ValueError("Empty 'messages' list in 'body'.")

            past_messages = messages[-self.user_valves.memory_limit:]
            prompt = self.extract_latest_prompt(past_messages)
            if not prompt:
                raise ValueError("Prompt cannot be empty!")

            payload = self.construct_payload(prompt, streaming=False)
            logger.debug(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
            await self.emit_status(__event_emitter__, "Sending request to RAG API...", done=False)

            response = await self.client.post(url=f"{self.valves.API_BASE_URL}/v2/rag", json=payload, headers={"Content-Type": "application/json"})
            if response.status_code != 200:
                error_text = await response.aread()
                logger.error(f"Error response: {error_text.decode()}")
                raise httpx.HTTPStatusError(f"Unexpected status code {response.status_code}: {error_text.decode()}", request=response.request, response=response)

            result = response.json()
            logger.debug(f"API Response: {json.dumps(result, indent=2, ensure_ascii=False)}")

            completion = result.get("results", {}).get("completion", {})
            choices = completion.get("choices", [])
            if choices and "message" in choices[0]:
                content = choices[0].get("message", {}).get("content", "")
                content = self.process_content(content)
                await self.emit_message(__event_emitter__, "Received assistant response.")
                await self.emit_status(__event_emitter__, "RAG Pipeline successfully completed.", done=True)
                return content
            else:
                await self.emit_status(__event_emitter__, "No response from the assistant.", done=True)
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

    def extract_latest_prompt(self, past_messages: List[dict]) -> str:
        for message in reversed(past_messages):
            if message.get("role") == "user" and "content" in message:
                content = message["content"]
                if isinstance(content, str):
                    return content.strip()
                elif isinstance(content, list):
                    content_list = [item["content"].strip() for item in content if isinstance(item, dict) and "content" in item]
                    return " ".join(content_list)
                elif isinstance(content, dict):
                    return content.get("content", "").strip()
        return ""

    def construct_payload(self, prompt: str, streaming: bool) -> Dict[str, Any]:
        return {
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
                    "full_text_limit": 500 if streaming else 200,
                    "rrf_k": 50,
                },
                "search_strategy": "hyde",
            },
            "rag_generation_config": {
                "model": self.valves.summarizer_model_id,
                "temperature": self.user_valves.temperature,
                "top_p": self.user_valves.top_p,
                "max_tokens_to_sample": self.user_valves.max_tokens,
                "stream": streaming,
            },
            "task_prompt_override": None,
            "include_title_if_available": False,
        }

    def process_content(self, content: Union[str, list, dict]) -> str:
        if isinstance(content, list):
            return " ".join([item["content"].strip() for item in content if isinstance(item, dict) and "content" in item])
        elif isinstance(content, dict):
            return content.get("content", "").strip()
        elif isinstance(content, str):
            return content.strip()
        else:
            raise ValueError("'content' in response must be a string, list, or dict with 'content' key.")

    async def emit_status(self, __event_emitter__: Optional[Callable[[dict], Any]], description: str, done: bool):
        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {"description": description, "done": done},
            })

    async def emit_message(self, __event_emitter__: Optional[Callable[[dict], Any]], content: str):
        if __event_emitter__:
            await __event_emitter__({
                "type": "message",
                "data": {"content": content}
            })

# Пример использования Pipeline с обработкой событий
async def example_event_emitter(event: dict):
    event_type = event.get("type")
    data = event.get("data", {})
    if event_type == "status":
        description = data.get("description", "")
        done = data.get("done", False)
        print(f"[STATUS] {description} (Done: {done})")
    elif event_type == "message":
        content = data.get("content", "")
        print(f"[MESSAGE] {content}")

async def main():
    pipeline = Pipeline()

    request_body_stream = {
        "messages": [
            {
                "role": "user",
                "content": "Hello, can you tell me about Aristotle?",
            }
        ],
        "kwargs": {"stream": True},
    }

    request_body_non_stream = {
        "messages": [
            {
                "role": "user",
                "content": "Hello, can you tell me about Aristotle?",
            }
        ],
        "kwargs": {"stream": False},
    }

    # Обработка потокового запроса
    async def handle_stream_request():
        try:
            async for chunk in await pipeline.pipe(request_body_stream, example_event_emitter):
                print("Streamed Response:", chunk)
        except Exception as e:
            print(f"Stream Request Error: {e}")

    # Обработка непотокового запроса
    async def handle_non_stream_request():
        try:
            response = await pipeline.pipe(request_body_non_stream, example_event_emitter)
            print("Final Response:", response)
        except Exception as e:
            print(f"Non-Stream Request Error: {e}")

    print("=== Начало обработки потокового запроса ===")
    await handle_stream_request()
    print("=== Завершение обработки потокового запроса ===\n")

    print("=== Начало обработки непотокового запроса ===")
    await handle_non_stream_request()
    print("=== Завершение обработки непотокового запроса ===")

if __name__ == "__main__":
    asyncio.run(main())
