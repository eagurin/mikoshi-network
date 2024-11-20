"""
title: R2R Agent Pipeline for OpenWebUI
author: Ваше Имя
version: 1.0.0
requirements: r2r, langchain, langchain-openai, langchain-community, duckduckgo-search, httpx, pydantic
description: Pipeline для интеграции R2R агента в OpenWebUI с динамической конфигурацией параметров с помощью промптов и LangChain.
"""

import os
import logging
from typing import Optional, Callable, Any
from pydantic import BaseModel, Field
from r2r import R2RClient
from langchain_openai.llms import OpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.tools import DuckDuckGoSearchRun


# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Pipe:
    class Valves(BaseModel):
        R2R_API_URL: str = Field(
            default=os.getenv("R2R_API_URL", "http://90.156.254.145:7272"),
            description="API URL R2R агента",
        )
        R2R_API_TOKEN: str = Field(
            default=os.getenv("R2R_API_TOKEN", ""),
            description="API токен для доступа к R2R",
        )
        TIMEOUT: int = Field(
            default=300,
            description="Таймаут запроса в секундах",
        )

    class UserValves(BaseModel):
        temperature: float = Field(
            default=0.7,
            description="Температура генерации",
        )
        max_tokens: int = Field(
            default=1024,
            description="Максимальное количество токенов для генерации",
        )
        memory_limit: int = Field(
            default=5,
            description="Количество последних сообщений для обработки",
        )
        use_hybrid_search: bool = Field(
            default=True,
            description="Использовать ли гибридный поиск",
        )

    def __init__(self):
        self.type = "filter"
        self.id = "r2r_agent_pipeline"
        self.name = "R2R Agent Pipeline"
        self.valves = self.Valves()
        self.user_valves = self.UserValves()

        # Проверка наличия API токена
        if not self.valves.R2R_API_TOKEN:
            raise ValueError(
                "R2R_API_TOKEN не указан. Пожалуйста, установите его в Valves или переменных окружения."
            )

        # Инициализация клиента R2R
        self.r2r_client = R2RClient(
            base_url=self.valves.R2R_API_URL,
            api_token=self.valves.R2R_API_TOKEN,
            timeout=self.valves.TIMEOUT,
        )

        # Инициализация LLM с использованием LangChain
        self.llm = OpenAI(
            openai_api_key=self.get_openai_api_key(),
            temperature=self.user_valves.temperature,
            max_tokens=self.user_valves.max_tokens,
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

    def get_openai_api_key(self) -> str:
        return os.getenv("OPENAI_API_KEY", "your-openai-api-key")

    def get_provider_models(self):
        return ["agent", "rag", "search"]

    async def pipe(
        self,
        body: dict,
        __event_emitter__: Optional[Callable[[dict], Any]] = None,
    ) -> str:
        try:
            model_id = body.get("model", "agent")
            messages = body.get("messages", [])
            kwargs = body.get("kwargs", {})

            # Динамическая конфигурация параметров с помощью промптов и LangChain
            # Можно использовать промпты для настройки параметров на лету

            if model_id == "agent":
                response = await self.r2r_client.agent(
                    messages=messages,
                    vector_search_settings={
                        "search_limit": 5,
                        "use_hybrid_search": self.user_valves.use_hybrid_search,
                        "filters": {},
                    },
                    rag_generation_config={
                        "max_tokens": self.user_valves.max_tokens,
                        "temperature": self.user_valves.temperature,
                    },
                    **kwargs,
                )
                assistant_message = self.extract_assistant_message(response)
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "message",
                            "data": {"content": assistant_message},
                        }
                    )
                return assistant_message

            elif model_id == "rag":
                user_message = self.get_last_user_message(messages)
                response = await self.r2r_client.rag(
                    query=user_message["content"],
                    vector_search_settings={
                        "search_limit": 5,
                        "use_hybrid_search": self.user_valves.use_hybrid_search,
                        "filters": {},
                    },
                    generation_config={
                        "max_tokens": self.user_valves.max_tokens,
                        "temperature": self.user_valves.temperature,
                    },
                    **kwargs,
                )
                generated_text = response.get("generated_text", "")
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "message",
                            "data": {"content": generated_text},
                        }
                    )
                return generated_text

            elif model_id == "search":
                user_message = self.get_last_user_message(messages)
                response = await self.r2r_client.search(
                    query=user_message["content"],
                    search_limit=5,
                    use_hybrid_search=self.user_valves.use_hybrid_search,
                    filters={},
                    **kwargs,
                )
                search_results = response.get("results", [])
                formatted_results = self.format_search_results(search_results)
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "message",
                            "data": {"content": formatted_results},
                        }
                    )
                return formatted_results

            else:
                raise ValueError(f"Unsupported model_id: {model_id}")

        except Exception as e:
            error_message = f"Ошибка при обработке: {str(e)}"
            logger.exception(error_message)
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "error", "data": {"description": error_message}}
                )
            return error_message

    def get_last_user_message(self, messages):
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg
        return {"content": ""}

    def extract_assistant_message(self, response):
        for msg in reversed(response.get("results", [])):
            if msg.get("role") == "assistant":
                return msg.get("content", "")
        return ""

    def format_search_results(self, results):
        formatted = ""
        for idx, result in enumerate(results, start=1):
            formatted += f"{idx}. {result.get('title', 'No Title')}\n{result.get('snippet', '')}\n\n"
        return formatted

    async def r2r_query_tool(self, query: str) -> str:
        response = await self.r2r_client.agent(
            messages=[{"role": "user", "content": query}],
            vector_search_settings={
                "search_limit": 5,
                "use_hybrid_search": self.user_valves.use_hybrid_search,
                "filters": {},
            },
            rag_generation_config={
                "max_tokens": self.user_valves.max_tokens,
                "temperature": self.user_valves.temperature,
            },
        )
        return self.extract_assistant_message(response)
