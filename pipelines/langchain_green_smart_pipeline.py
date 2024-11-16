"""
title: R2R Native Agent Pipeline
author: Evgeny A.
author_url: https://github.com/open-webui
funding_url: https://github.com/open-webui
version: 0.0.0.001a
requirements: r2r, langchain, openai, duckduckgo-search, httpx, pydantic
"""

import asyncio
import os
import logging
from typing import List, Optional, Callable, Any

import httpx
from pydantic import BaseModel, Field
from r2r import R2RClient

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Импорты из библиотеки LangChain
from langchain_openai.llms import OpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.tools import DuckDuckGoSearchRun

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
        self.name = "R2R Agent Pipeline"
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
            )
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

            # Обработка запроса агента - синхронный вызов в отдельном потоке
            response = await asyncio.to_thread(self.agent.run, user_message)

            # Отправляем ответ через эмиттер событий, если он есть
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "message", "data": {"content": response}}
                )

            return response

        except Exception as e:
            error_message = f"Error during processing: {e}"
            logger.exception(error_message)
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "error", "data": {"description": error_message}}
                )
            raise


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
                нужно выполнять задачи асинхронно, максимально сократив время молчания в общении с клиентом. в  самом начале надоВ самом начале нужно разделять задачи, то есть их выполнять синхронно. Первая задача – это мы делаем просто query-запрос быстрый какой-то, с темой, чтобы что-то, ну, забить время разговора, пока ответ агента подготавливается. То есть это что-то, ну, там, вводная какая-то информация. Вот, например, у OpenAI сейчас там, вот, была эта хрень, то, что там как разные сообщения, типа предподготовка. Вот, надо что-то подобное сделать. А в идеале вообще сделать так же. Так же это все делается синхронно. То есть в самом начале у нас стартуют, так, две задачи. Одна – это query, вот этот быстрый ответ. Другая – агент готовит ответ. А так же, ну, пока давай с этим разберемся. Да, агент мы используем максимально, то есть пытаемся его прикрутить туда, чтобы в самом pipeline, и вообще мы почти не взаимодействовали с кодом, чтобы все взаимодействие с чатом максимально сразу передавалось и замыкалось, то есть закупоривалось вот на R2R-агенте. В него, ну, то есть это его полностью задача. Мы не должны с этим данным никак будем перестираться.


В общем, такое дело. Работает хорошо. Другой момент. Он должен работать как бы всегда, то есть всегда мог услышать то, что ты ему напишешь. То есть вот он сейчас долго висит, долго грузит, но ничего не пишет. Но все равно он что-то же думает, изучает. В итоге, когда ведется диалог с агентом, то есть он оставался в автономном режиме работать и прогружал, пока не ответит. То есть чтобы он думал не только отвечая на вопрос, а думал все время. Вот ты понимаешь о чем. Я думаю использовать либо лэнгчейн, либо аутджен для этих целей. Подумай, как лучше всего сделать.

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
    asyncio.run(main())
