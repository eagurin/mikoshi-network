import asyncio
from typing import List, Union, AsyncGenerator, Optional, Callable, Any
from pydantic import BaseModel, Field
import os
import httpx
import json
import requests
from langchain.agents import initialize_agent, AgentType
from langchain_community.llms import OpenAI
from langchain_community.tools import OpenAPISpec
from r2r import R2RClient

# Вспомогательные функции
def get_last_user_message(messages: List[dict]) -> str:
    """Получает последнее сообщение от пользователя из списка сообщений."""
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""

def add_or_update_system_message(system_prompt: str, messages: List[dict]) -> List[dict]:
    """Добавляет или обновляет системное сообщение в списке сообщений."""
    for message in messages:
        if message.get("role") == "system":
            message["content"] = system_prompt
            return messages
    # Если системное сообщение отсутствует, добавляем его в начало списка
    messages.insert(0, {"role": "system", "content": system_prompt})
    return messages

def get_tools_specs(tools: Optional[List[Any]]) -> List[dict]:
    """Генерирует спецификации инструментов для передачи в системный промпт."""
    if tools is None:
        return []
    specs = []
    for tool in tools:
        spec = {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        specs.append(spec)
    return specs

# Системный промпт для вызова функций
DEFAULT_SYSTEM_PROMPT = (
    """Tools: {}
If a function tool doesn't match the query, return an empty string. Else, pick a
function tool, fill in the parameters from the function tool's schema, and
return it in the format { "name": "functionName", "parameters": { "key": "value" } }. Only pick a function if the user asks. Only return the object. Do not return any other text."
"""
)

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
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
And answer according to the language of the user's question.""",
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
        self.type = "filter"
        self.name = "Function Calling Blueprint"
        self.prompt = prompt or DEFAULT_SYSTEM_PROMPT
        self.tools: object = None
        self.valves = self.Valves(
            pipelines=["*"],
            OPENAI_API_BASE_URL=os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com/v1"),
            OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
            TASK_MODEL=os.getenv("TASK_MODEL", "gpt-3.5-turbo"),
            TEMPLATE=self.Valves.TEMPLATE
        )
        self.user_valves = self.UserValves()
        self.client = httpx.AsyncClient(timeout=10)  # Убедитесь, что TIMEOUT определен
        self.r2r_client = R2RClient(self.valves.OPENAI_API_BASE_URL)
        openapi_spec = self.load_openapi_spec("https://example.com/api_spec.json")  # Убедитесь в правильности URL
        if not openapi_spec:
            raise Exception("Не удалось загрузить OpenAPI спецификацию")
        self.openapi_tool = OpenAPISpec.from_dict(openapi_spec)
        self.llm = OpenAI(model_name=self.valves.TASK_MODEL, temperature=0.7)
        self.agent = initialize_agent(
            tools=[self.openapi_tool],
            agent=AgentType.OPENAI_FUNCTIONS,
            llm=self.llm,
            verbose=True
        )
            
    def load_openapi_spec(self, api_spec_url: str):
        """Загрузка OpenAPI спецификации"""
        try:
            response = requests.get(api_spec_url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Ошибка при загрузке OpenAPI: {e}")
            return None
            
    # Остальные методы класса Pipeline здесь ...

    async def pipe(
        self,
        body: dict,
        user: Optional[dict] = None,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> Union[dict, AsyncGenerator[str, None]]:
        """
        Метод для определения типа обработки (стриминг или нет)
        """
        return await self.inlet(body, user, __event_emitter__)

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

# Функция main должна быть вне класса Pipeline
async def main():
    pipeline = Pipeline()
    
    # Пример запроса с потоковой передачей
    request_body_stream = {
        "messages": [
            {"role": "user", "content": "Hello, can you tell me about Aristotle?"}
        ],
        "kwargs": {"stream": True},
    }
    
    # Пример запроса без потоковой передачи
    request_body_non_stream = {
        "messages": [
            {"role": "user", "content": "Hello, can you tell me about Aristotle?"}
        ],
        "kwargs": {"stream": False},
    }
    
    print("=== Streaming Response ===")
    response_stream = await pipeline.pipe(
        request_body_stream, 
        user=None, 
        __event_emitter__=pipeline.example_event_emitter
    )
    if isinstance(response_stream, AsyncGenerator):
        async for response in response_stream:
            print("Streamed Response:", response)
    else:
        print("Unexpected response type for streaming.")
    
    print("\n=== Non-Streaming Response ===")
    response = await pipeline.pipe(
        request_body_non_stream, 
        user=None, 
        __event_emitter__=pipeline.example_event_emitter
    )
    print("Final Response:", response)

if __name__ == "__main__":
    asyncio.run(main())
