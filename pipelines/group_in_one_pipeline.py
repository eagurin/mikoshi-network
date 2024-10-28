from typing import (
    TYPE_CHECKING,
    Any,
    Union,
    AsyncGenerator,
    Optional,
    Callable,
    List,
    Generator,
    Iterator,
    Literal,
)
from pydantic import BaseModel, Field
import os
import httpx
import json
import asyncio
import threading
import subprocess
from datetime import datetime

if TYPE_CHECKING:
    from langchain_community.utilities.openapi import OpenAPISpec
    from langchain_community.utilities.openapi import HTTPVerb

from langchain._api import create_importer
from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent

# 1. Динамический импорт устаревших компонентов
DEPRECATED_LOOKUP = {
    "HTTPVerb": "langchain_community.utilities.openapi",
    "OpenAPISpec": "langchain_community.utilities.openapi",
}
_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)

def __getattr__(name: str) -> Any:
    """Динамический поиск атрибутов."""
    return _import_attribute(name)

__all__ = [
    "HTTPVerb",
    "OpenAPISpec",
]

# 2. Определение Pydantic моделей для валидации данных
class OpenAIChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    session_id: str
    messages: List[OpenAIChatMessage]
    kwargs: Optional[dict] = None

class ChatResponse(BaseModel):
    status: str
    response: Optional[str] = None
    error: Optional[str] = None

# 3. Основной класс Pipeline, объединяющий все конвейеры
class Pipeline:
    # 3.1. Внутренние классы для различных конвейеров
    class RAGPipeline:
        # Реализация RAGPipeline
        # (См. предыдущие реализации или заполните по необходимости)
        pass  # Предполагается, что RAGPipeline реализован корректно

    class OpenAPIFunctionsPipeline:
        class Valves(BaseModel):
            OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
            OPENAI_API_BASE_URL: str = Field(default="https://api.openai.com/v1")

        def __init__(self):
            self.type = "filter"
            self.name = "OpenAPI Functions Pipeline"
            self.valves = self.Valves()
            # Загрузка спецификации OpenAPI
            self.openapi_spec_url = 'https://api.mikoshi.company/v2/openapi_spec'
            self.openapi_spec = self.load_openapi_spec()

            # Создание объекта OpenAPISpec
            self.openapi_tool = OpenAPISpec.from_text(json.dumps(self.openapi_spec))

            # Инициализация агента LangChain
            self.llm = OpenAI(
                openai_api_key=self.valves.OPENAI_API_KEY,
                model_name="gpt-3.5-turbo-0613",
                temperature=0,
            )
            self.agent = initialize_agent(
                tools=[self.openapi_tool],
                llm=self.llm,
                agent=AgentType.OPENAI_FUNCTIONS,
                verbose=True,
            )

        def load_openapi_spec(self):
            response = httpx.get(self.openapi_spec_url)
            response.raise_for_status()
            return response.json()

        async def on_startup(self):
            print(f"on_startup:{self.name}")
            pass

        async def on_shutdown(self):
            print(f"on_shutdown:{self.name}")
            pass

        async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
            # Обработка сообщения пользователя с помощью агента
            last_user_message = get_last_user_message(body.get("messages", []))
            if not last_user_message:
                return body  # Нет сообщения пользователя для обработки
            # Использование агента для обработки запроса пользователя
            try:
                result = self.agent.run(last_user_message)
                # Добавление ответа агента в список сообщений
                body["messages"].append({"role": "assistant", "content": result})
                return body
            except Exception as e:
                print(f"Error in OpenAPIFunctionsPipeline: {e}")
                return body

    # Другие конвейеры (FunctionCallingPipeline, MemoryFilterPipeline, и т.д.)
    # Реализуйте их по необходимости

    # 4. Утилиты
    def get_last_user_message(messages: List[dict]) -> str:
        """
        Получить последнее сообщение пользователя из списка сообщений.
        """
        for message in reversed(messages):
            if message.get("role") == "user" and "content" in message:
                return message["content"]
        return ""

    def add_or_update_system_message(
        system_prompt: str, messages: List[dict]
    ) -> List[dict]:
        """
        Добавить или обновить системное сообщение в списке сообщений.
        """
        for message in messages:
            if message.get("role") == "system":
                message["content"] = system_prompt
                return messages
        # Если системное сообщение отсутствует, добавить его в начало
        messages.insert(0, {"role": "system", "content": system_prompt})
        return messages

    def get_tools_specs(tools) -> List[dict]:
        """
        Получить спецификации инструментов для конвейера вызова функций.
        """
        tool_specs = []
        for tool_name in dir(tools):
            tool = getattr(tools, tool_name)
            if callable(tool) and not tool_name.startswith("__"):
                tool_specs.append(
                    {
                        "name": tool_name,
                        "description": tool.__doc__,
                        # Можно добавить параметры, если необходимо
                    }
                )
        return tool_specs

    # 5. Интеграция всех конвейеров в общий Pipeline
    class OpenWebUIPipeline:
        def __init__(self):
            # Инициализация всех конвейеров
            self.rag_pipeline = Pipeline.RAGPipeline()
            self.function_calling_pipeline = Pipeline.OpenAPIFunctionsPipeline()
            # Инициализируйте другие конвейеры по необходимости

        async def on_startup(self):
            # Запуск всех конвейеров
            await self.rag_pipeline.on_startup()
            await self.function_calling_pipeline.on_startup()
            # Запустите другие конвейеры по необходимости

        async def on_shutdown(self):
            # Остановка всех конвейеров
            await self.rag_pipeline.on_shutdown()
            await self.function_calling_pipeline.on_shutdown()
            # Остановите другие конвейеры по необходимости

        async def handle_request(self, chat_request: ChatRequest) -> ChatResponse:
            try:
                # Преобразование модели в словарь
                data = chat_request.dict()
                # Обработка через Function Calling Pipeline
                data = await self.function_calling_pipeline.inlet(body=data)
                # Обработка через RAG Pipeline (стриминг или нет)
                rag_response = await self.rag_pipeline.pipe(
                    data, self.example_event_emitter
                )
                if isinstance(rag_response, AsyncGenerator):
                    # Стриминговый ответ
                    streamed_response = []
                    async for chunk in rag_response:
                        streamed_response.append(chunk)
                    final_response = "".join(streamed_response)
                else:
                    # Нестриминговый ответ
                    final_response = rag_response

                return ChatResponse(status="success", response=final_response)
            except KeyError as e:
                error_message = str(e)
                return ChatResponse(status="error", error=error_message)
            except Exception as e:
                error_message = f"An error occurred during request processing: {str(e)}"
                return ChatResponse(status="error", error=error_message)

        # Пример обработчика событий
        async def example_event_emitter(self, event: dict):
            event_type = event.get("type")
            data = event.get("data", {})
            if event_type == "status":
                description = data.get("description", "")
                done = data.get("done", False)
                print(f"[STATUS] {description} (Done: {done})")
            elif event_type == "message":
                content = data.get("content", "")
                print(f"[MESSAGE] {content}")

# 6. Интеграция с FastAPI
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()
pipeline = Pipeline.OpenWebUIPipeline()

@app.on_event("startup")
async def startup_event():
    await pipeline.on_startup()

@app.on_event("shutdown")
async def shutdown_event():
    await pipeline.on_shutdown()

@app.post("/chat/completed", response_model=ChatResponse)
async def chat_completed(chat_request: ChatRequest):
    response = await pipeline.handle_request(chat_request)
    return response
