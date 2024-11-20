"""
title: R2R Unified Agent Pipeline o1o
author: Evgeny A.
author_url: https://github.com/open-webui
funding_url: https://github.com/open-webui
version: 0.0.0.003d
requirements: r2r, langchain, langchain-openai, langchain-community, duckduckgo-search, httpx, pydantic
"""

import asyncio
import os
import logging
from typing import List, Optional, Callable, Any
import httpx
from pydantic import BaseModel, Field
from r2r import R2RClient
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain_community.tools import DuckDuckGoSearchRun

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
            default=300,
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
            default=1024,
            description="Maximum number of tokens for generation",
        )
        memory_limit: int = Field(
            default=10,
            description="Number of recent messages to process",
        )

    def __init__(self):
        self.type = "pipe"
        self.name = "R2R Unified Agent Pipeline"
        self.valves = self.Valves(
            API_BASE_URL=os.getenv(
                "API_BASE_URL", "https://api.durka.dndstudio.ru"
            ),
            TIMEOUT=int(os.getenv("TIMEOUT", "300")),
            summarizer_model_id=os.getenv(
                "SUMMARIZER_MODEL_ID", "gpt-4o-mini"
            ),
        )
        self.user_valves = self.UserValves()
        self.client = httpx.AsyncClient(
            timeout=self.valves.TIMEOUT, verify=False
        )
        self.r2r_client = R2RClient(self.valves.API_BASE_URL)

        # Получение API ключа OpenAI
        self.openai_api_key = self.get_api_key()

        # Инициализация ChatOpenAI
        self.llm = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            temperature=self.user_valves.temperature,
        )

        # Настройка инструментов
        self.tools = [
            Tool(
                name="Search",
                func=DuckDuckGoSearchRun().run,
                description="Полезно для поиска информации в интернете.",
            )
        ]

        # Инициализация памяти
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
        )

        # Создание агента
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
        )

    def get_api_key(self) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY is not set in environment variables.")
            raise ValueError("OPENAI_API_KEY is not set.")
        return api_key

    def pipelines(self) -> List[str]:
        return []

    async def pipe(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        try:
            messages = body.get("messages", [])
            if not messages:
                raise ValueError("Empty 'messages' list in 'body'.")

            # Извлечение последнего запроса пользователя
            user_message = messages[-1]["content"]

            # Запуск двух параллельных задач
            quick_response_task = asyncio.create_task(
                self.quick_response(user_message)
            )
            agent_response_task = asyncio.create_task(
                self.agent_response(user_message)
            )

            # Ожидаем быструю задачу
            quick_response = await quick_response_task
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "message", "data": {"content": quick_response}}
                )
            logger.info(f"Быстрый ответ: {quick_response}")

            # Ожидаем основной ответ от агента
            agent_response = await agent_response_task
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "message", "data": {"content": agent_response}}
                )
            logger.info(f"Ответ агента: {agent_response}")
            return agent_response
        except Exception as e:
            error_message = f"Error during processing: {e}"
            logger.exception(error_message)
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "error", "data": {"description": error_message}}
                )
            raise

    async def quick_response(self, query: str) -> str:
        """Асинхронная быстрая задача."""
        logger.info(f"Старт быстрой задачи для запроса: {query}")
        await asyncio.sleep(0.5)  # эмуляция задержки
        return f"Мы получили ваш запрос: '{query}'. Пожалуйста, подождите..."

    async def agent_response(self, query: str) -> str:
        """Основная задача агента."""
        logger.info(f"Запуск агента для запроса: {query}")
        response = await asyncio.get_event_loop().run_in_executor(
            None, self.agent.run, query
        )
        return response


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
    elif event_type == "error":
        description = data.get("description", "")
        print(f"[ERROR] {description}")


async def main():
    pipeline = Pipeline()
    request_body = {
        "messages": [
            {
                "role": "user",
                "content": """
                Нужно выполнять задачи асинхронно, максимально сократив время молчания в общении с клиентом. В самом начале нужно разделять задачи, то есть их выполнять синхронно. Первая задача – это мы делаем просто query-запрос, с темой, чтобы что-то, ну, забить время разговора, пока ответ агента подготавливается. Это должно бы быть что-то вроде быстрого ответа. 
                """,
            }
        ],
        "kwargs": {"stream": False},
    }
    try:
        response = await pipeline.pipe(request_body, example_event_emitter)
        print("Final Response:", response)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY is not set in environment variables.")
        exit(1)
    asyncio.run(main())
