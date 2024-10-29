from typing import List, Union, AsyncGenerator, Optional, Callable, Any
from pydantic import BaseModel, Field
import os
import httpx
import json
import asyncio
import requests

# Предполагается, что необходимые модули доступны
# from schemas import OpenAIChatMessage
# from utils.pipelines.main import (
#     get_last_user_message,
#     add_or_update_system_message,
#     get_tools_specs,
# )
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.tools.openapi import OpenAPISpec
from r2r import R2RClient  # Убедитесь, что модуль r2r доступен

# Константы для URL API
API_SPEC_URL = "https://api.mikoshi.company/v2/openapi_spec"
R2R_API_BASE_URL = "http://localhost:7272"

# Функция загрузки спецификации OpenAPI
def load_openapi_spec(api_spec_url: str):
    """Загрузка OpenAPI спецификации"""
    try:
        response = requests.get(api_spec_url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Ошибка при загрузке OpenAPI: {e}")
        return None

# Системный промпт для вызова функций
DEFAULT_SYSTEM_PROMPT = (
    """Tools: {}
If a function tool doesn't match the query, return an empty string. Else, pick a
function tool, fill in the parameters from the function tool's schema, and
return it in the format {{ "name": "functionName", "parameters": {{ "key": "value" }} }}. Only pick a function if the user asks. Only return the object. Do not return any other text."
"""
)

class Pipeline:
    class Valves(BaseModel):
        # Список идентификаторов целевых конвейеров (моделей), к которым будет подключен этот фильтр.
        # Если хотите подключить фильтр ко всем конвейерам, установите pipelines = ["*"]
        pipelines: List[str] = []
        
        # Приоритетный уровень для фильтра конвейера.
        # Уровень приоритета определяет порядок выполнения фильтровых конвейеров.
        # Чем меньше число, тем выше приоритет.
        priority: int = 0
        
        # Настройки для вызова функций
        OPENAI_API_BASE_URL: str = Field(
            default="https://api.openai.com/v1",
            description="OpenAI API Base URL",
        )
        OPENAI_API_KEY: str = Field(
            default=os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
            description="OpenAI API Key",
        )
        TASK_MODEL: str = Field(
            default="gpt-3.5-turbo",
            description="Model to use for tasks",
        )
        TEMPLATE: str = Field(
            default="""Use the following context as your learned knowledge, inside <context></context> XML tags.
<context>
    {{CONTEXT}}
</context>

When answering the user:
- If you don't know, just say that you don't know.
- If you are not sure, ask for clarification.
Avoid mentioning that you obtained the information from the context.
Answer according to the language of the user's question.""",
            description="Template for responses",
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
    
    def __init__(self, prompt: Optional[str] = None) -> None:
        # Конвейеры фильтров совместимы только с Open WebUI
        # Фильтр конвейера можно рассматривать как middleware, который может редактировать данные формы перед отправкой их в OpenAI API.
        self.type = "filter"
        
        # Опционально, можно задать id и имя конвейера.
        # Рекомендуется не указывать id, чтобы он мог быть автоматически определен из имени файла, позволяя пользователям устанавливать несколько версий одного и того же конвейера.
        self.name = "Function Calling Blueprint"
        self.prompt = prompt or DEFAULT_SYSTEM_PROMPT
        self.tools: object = None
        
        # Инициализация клапанов
        self.valves = self.Valves(
            pipelines=["*"],  # Подключение ко всем конвейерам
            OPENAI_API_BASE_URL=os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1"),
            OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
            TASK_MODEL=os.getenv("TASK_MODEL", "gpt-3.5-turbo"),
            TEMPLATE=self.Valves.TEMPLATE
        )
        
        self.user_valves = self.UserValves()
        self.client = httpx.AsyncClient(timeout=self.valves.TIMEOUT)
    
    async def on_startup(self):
        # Эта функция вызывается при запуске сервера.
        print(f"on_startup: {__name__}")
        pass
    
    async def on_shutdown(self):
        # Эта функция вызывается при остановке сервера.
        print(f"on_shutdown: {__name__}")
        pass
    
    async def inlet(
        self,
        body: dict,
        user: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> Union[dict, AsyncGenerator[str, None]]:
        # Если запрашивается генерация заголовка, пропускаем фильтр вызова функций
        if body.get("title", False):
            return body
    
        print(f"pipe: {__name__}")
        print(f"user: {user}")
    
        # Получаем последние сообщения пользователя
        messages = body.get("messages", [])
        if not messages:
            raise ValueError("Empty 'messages' list in 'body'.")
    
        memory_limit = self.user_valves.memory_limit
        past_messages = messages[-memory_limit:]
    
        # Ищем последнее сообщение от пользователя
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
                else:
                    continue
                if prompt:
                    break  # Если нашли непустой prompt, выходим из цикла
    
        if not prompt:
            raise ValueError("Prompt cannot be empty!")
    
        # Определяем, требуется ли потоковая обработка
        stream = body.get("kwargs", {}).get("stream", False)
        if stream:
            return self.pipe_stream(prompt, __event_emitter__)
        else:
            return await self.pipe_non_stream(prompt, body, messages, __event_emitter__)
    
    async def pipe_stream(
        self,
        prompt: str,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> AsyncGenerator[str, None]:
        try:
            await self.emit_status(
                __event_emitter__, "Starting RAG Pipeline (Streaming)...", done=False
            )
    
            # Подготовка полезной нагрузки для RAG API
            endpoint = f"{self.valves.API_BASE_URL}/v2/rag"
            headers = {"Content-Type": "application/json"}
            payload = {
                "query": prompt,
                "vector_search_settings": {
                    "use_vector_search": True,
                    "use_hybrid_search": self.user_valves.use_hybrid_search,
                    "filters": {},
                    "search_limit": 30,
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
                    "model": self.valves.summarizer_model_id,
                    "temperature": self.user_valves.temperature,
                    "top_p": self.user_valves.top_p,
                    "max_tokens_to_sample": self.user_valves.max_tokens,
                    "stream": True,  # Обеспечиваем потоковую передачу
                },
                "task_prompt_override": None,
                "include_title_if_available": False,
            }
            print(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
            await self.emit_status(
                __event_emitter__, "Sending request to RAG API...", done=False
            )
            await self.emit_message(__event_emitter__, "Streaming assistant's response...")
    
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
                        print(f"Streamed line: {line}")
                        # Предполагаем, что линия - это JSON; пытаемся распарсить
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
                            # Если это не JSON, просто возвращаем линию
                            yield line
            await self.emit_status(
                __event_emitter__, "Completed streaming data to user.", done=True
            )
        except Exception as e:
            error_message = f"Error during streaming: {e}"
            print(error_message)
            await self.emit_status(__event_emitter__, error_message, done=True)
            raise
    
    async def pipe_non_stream(
        self,
        prompt: str,
        body: dict,
        messages: List[dict],
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> dict:
        try:
            await self.emit_status(
                __event_emitter__, "Starting RAG Pipeline...", done=False
            )
    
            # Подготовка полезной нагрузки для RAG API
            endpoint = f"{self.valves.API_BASE_URL}/v2/rag"
            headers = {"Content-Type": "application/json"}
            payload = {
                "query": prompt,
                "vector_search_settings": {
                    "use_vector_search": True,
                    "use_hybrid_search": self.user_valves.use_hybrid_search,
                    "filters": {},
                    "search_limit": 30,
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
                    "model": self.valves.summarizer_model_id,
                    "temperature": self.user_valves.temperature,
                    "top_p": self.user_valves.top_p,
                    "max_tokens_to_sample": self.user_valves.max_tokens,
                    "stream": False,
                },
                "task_prompt_override": None,
                "include_title_if_available": False,
            }
            print(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
            await self.emit_status(
                __event_emitter__, "Sending request to RAG API...", done=False
            )
    
            response = await self.client.post(
                url=endpoint,
                json=payload,
                headers=headers,
            )
            if response.status_code != 200:
                error_text = response.text
                raise httpx.HTTPStatusError(
                    f"Unexpected status code {response.status_code}: {error_text}",
                    request=response.request,
                    response=response,
                )
            await self.emit_status(
                __event_emitter__, "Received response from RAG API.", done=False
            )
            result = response.json()
            print(f"API Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
            completion = result.get("results", {}).get("completion", {})
            choices = completion.get("choices", [])
            if choices and "message" in choices[0]:
                content = choices[0].get("message", {}).get("content", "")
                if isinstance(content, list):
                    # Обрабатываем каждый элемент списка
                    content_list = []
                    for item in content:
                        if isinstance(item, str):
                            content_list.append(item)
                        elif isinstance(item, dict):
                            # Извлекаем строку из словаря
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
                # Добавляем ответ ассистента в сообщения
                messages.append({"role": "assistant", "content": content})
                body["messages"] = messages
                return body
            else:
                await self.emit_status(
                    __event_emitter__,
                    "No response from the assistant.",
                    done=True,
                )
                messages.append(
                    {"role": "assistant", "content": "No response from the assistant."}
                )
                body["messages"] = messages
                return body
        except Exception as e:
            error_message = f"Error during non-streaming: {e}"
            print(error_message)
            await self.emit_status(__event_emitter__, error_message, done=True)
            raise
    
    async def pipe(
        self,
        body: dict,
        user: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> Union[dict, AsyncGenerator[str, None]]:
        return await self.inlet(body, user, __event_emitter__)
    
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
            await __event_emitter__({"type": "message", "data": {"content": content}})
    
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
    
    # Пример теста для выполнения взаимодействия с pipeline и Open-WebUI
    async def main():
        pipeline = Pipeline()
        request_body_stream = {
            "messages": [
                {"role": "user", "content": "Hello, can you tell me about Aristotle?"}
            ],
            "kwargs": {"stream": True},
        }
        request_body_non_stream = {
            "messages": [
                {"role": "user", "content": "Hello, can you tell me about Aristotle?"}
            ],
            "kwargs": {"stream": False},
        }
        print("=== Streaming Response ===")
        response_stream = await pipeline.pipe(request_body_stream, example_event_emitter)
        if hasattr(response_stream, "__aiter__"):
            async for response in response_stream:
                print("Streamed Response:", response)
        else:
            print("Unexpected response type for streaming.")
        print("\n=== Non-Streaming Response ===")
        response = await pipeline.pipe(request_body_non_stream, example_event_emitter)
        print("Final Response:", response)
    
    # Запуск теста с использованием asyncio в основной функции
    if __name__ == "__main__":
        asyncio.run(main())
