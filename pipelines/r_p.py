"""
title: R2R Green Agent Pipeline with OpenWebUI
author: Evgeny A.
author_url: https://github.com/open-webui
version: 1.0.0
requirements: r2r, langchain, langchain-openai, langchain-community, duckduckgo-search, httpx, pydantic
"""

import asyncio
import os
import logging
from typing import List, Optional, Callable, Any
import httpx
from pydantic import BaseModel, Field
from r2r import R2RClient
from langchain_openai.llms import OpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferWindowMemory
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
            default=4024,
            description="Maximum number of tokens for generation",
        )
        memory_limit: int = Field(
            default=10,
            description="Number of recent messages to process",
        )

    def __init__(self):
        self.type = "pipe"
        self.name = "R2R Green Agent Pipeline"
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
        self.r2r_client = R2RClient(self.valves.API_BASE_URL)
        self.client = httpx.AsyncClient(timeout=self.valves.TIMEOUT)

        # Инициализация LLM
        self.llm = OpenAI(
            openai_api_key=self.get_api_key(),
            temperature=self.user_valves.temperature,
        )

        # Настройка инструментов
        self.tools = [
            Tool(
                name="Search",
                func=DuckDuckGoSearchRun().run,
                description="Полезно для поиска информации в интернете.",
            ),
            Tool(
                name="R2R Query",
                func=self.r2r_query_tool,
                description="Запрос к R2R для получения ответа.",
            ),
        ]

        # Инициализация памяти
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=self.user_valves.memory_limit,
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
        return os.getenv("OPENAI_API_KEY", "your-openai-api-key")

    async def pipe(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        try:
            messages = body.get("messages", [])
            if not messages:
                raise ValueError("Empty 'messages' list in 'body'.")

            # Этап 1: Быстрый ответ (например, подтверждение принятия запроса)
            quick_response = "Спасибо, ваш запрос обрабатывается."
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "message", "data": {"content": quick_response}}
                )

            # Этап 2: Параллельный поиск
            user_message = messages[-1]["content"]
            search_task = asyncio.create_task(
                self.perform_search(user_message)
            )
            agent_response_task = asyncio.create_task(
                self.agent.run(user_message)
            )

            # Ожидаем результата поиска и ответа от агента
            search_result = await search_task
            agent_response = await agent_response_task

            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "message",
                        "data": {
                            "content": f"Результат поиска: {search_result}"
                        },
                    }
                )
                await __event_emitter__(
                    {"type": "message", "data": {"content": agent_response}}
                )

            return agent_response
        except Exception as e:
            error_message = f"Ошибка при обработке: {str(e)}"
            logger.exception(error_message)
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "error", "data": {"description": error_message}}
                )
            raise

    async def perform_search(self, query: str) -> str:
        """Поиск информации используется DuckDuckGo или запрос к R2R."""
        search_tool = DuckDuckGoSearchRun()
        return await search_tool.run(query)

    async def r2r_query_tool(self, query: str) -> str:
        """Инструмент для выполнения запроса к R2R."""
        response = await self.r2r_client.agent(
            messages=[{"role": "user", "content": query}],
            vector_search_settings={"search_limit": 5, "filters": {}},
        )
        return response.get("results", [{}])[0].get(
            "content", "Нет ответа от R2R."
        )


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
            {"role": "user", "content": "Расскажи мне о нейронных сетях."}
        ],
        "kwargs": {"stream": False},
    }
    try:
        response = await pipeline.pipe(request_body, example_event_emitter)
        print("Final Response:", response)
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
