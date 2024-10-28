# Понял вас. Вы хотите объединить все ваши конвейеры (pipelines) в один файл для `open-webui`, включая:

# 1. **Динамический импорт устаревших компонентов**.
# 2. **RAG Pipeline** для обработки запросов через API с использованием OpenAPI и LangChain агента.
# 3. **Конвейер вызова функций**.
# 4. **Фильтр долгосрочной памяти**.
# 5. **Конвейер выполнения Python-кода**.
# 6. **Интеграция с Llama Index и Ollama**.

# Ниже приведен полный пример объединенного конвейера, который включает все перечисленные компоненты. Этот файл можно разместить, например, как `pipeline.py` внутри вашего проекта `open-webui`.

# ```python
# # backend/open_webui/pipelines/pipeline.py

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
)
from pydantic import BaseModel, Field
import os
import httpx
import json
import asyncio
import threading
import subprocess

if TYPE_CHECKING:
    from langchain_community.tools import OpenAPISpec
    from langchain_community.utilities.openapi import HTTPVerb

from langchain._api import create_importer

# 1. Динамический импорт устаревших компонентов
DEPRECATED_LOOKUP = {
    "HTTPVerb": "langchain_community.utilities.openapi",
    "OpenAPISpec": "langchain_community.tools",
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
        class Valves(BaseModel):
            API_BASE_URL: str = Field(
                default="https://api.mikoshi.company",
                description="Base API URL",
            )
            TIMEOUT: int = Field(
                default=60,
                description="Request timeout in seconds",
            )
            summarizer_model_id: str = Field(
                default="gpt-4o",
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
            self.name = "RAG Pipeline"
            self.valves = self.Valves(
                API_BASE_URL=os.getenv("API_BASE_URL", "https://api.mikoshi.company"),
                TIMEOUT=int(os.getenv("TIMEOUT", 60)),
                summarizer_model_id=os.getenv("SUMMARIZER_MODEL_ID", "gpt-4o"),
            )
            self.user_valves = self.UserValves()
            # Disable SSL verification by setting verify=False here
            self.client = httpx.AsyncClient(timeout=self.valves.TIMEOUT, verify=False)

        async def pipe_stream(
            self, body: dict, event_emitter: Optional[Callable[[dict], Any]] = None
        ) -> AsyncGenerator[str, None]:
            try:
                await self.emit_status(
                    event_emitter, "Starting RAG Pipeline (Streaming)...", done=False
                )
                messages = body.get("messages", [])
                if not messages:
                    raise ValueError("Empty 'messages' list in 'body'.")
                memory_limit = self.user_valves.memory_limit
                past_messages = messages[-memory_limit:]
                # Extract the latest user message
                prompt = self.extract_prompt(past_messages)
                if not prompt:
                    raise ValueError("Prompt cannot be empty!")
                payload = self.build_payload(prompt)
                print(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
                await self.emit_status(
                    event_emitter, "Sending request to RAG API...", done=False
                )
                await self.emit_message(
                    event_emitter, "Streaming assistant's response..."
                )
                async with self.client.stream(
                    "POST",
                    payload["endpoint"],
                    json=payload["payload"],
                    headers=payload["headers"],
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
                            # Assume line is JSON, try to parse
                            try:
                                data = json.loads(line)
                                # Extract content from data
                                content = (
                                    data.get("choices", [{}])[0]
                                    .get("delta", {})
                                    .get("content", "")
                                )
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                # If not JSON, yield the raw line
                                yield line
                await self.emit_status(
                    event_emitter, "Completed streaming data to user.", done=True
                )
            except Exception as e:
                error_message = f"Error during streaming: {e}"
                print(error_message)
                await self.emit_status(event_emitter, error_message, done=True)
                raise

        async def pipe_non_stream(
            self, body: dict, event_emitter: Optional[Callable[[dict], Any]] = None
        ) -> str:
            try:
                await self.emit_status(
                    event_emitter, "Starting RAG Pipeline...", done=False
                )
                messages = body.get("messages", [])
                if not messages:
                    raise ValueError("Empty 'messages' list in 'body'.")
                memory_limit = self.user_valves.memory_limit
                past_messages = messages[-memory_limit:]
                # Extract the latest user message
                prompt = self.extract_prompt(past_messages)
                if not prompt:
                    raise ValueError("Prompt cannot be empty!")
                payload = self.build_payload(prompt)
                print(f"Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
                await self.emit_status(
                    event_emitter, "Sending request to RAG API...", done=False
                )
                response = await self.client.post(
                    url=payload["endpoint"],
                    json=payload["payload"],
                    headers=payload["headers"],
                )  # Removed verify=False here
                if response.status_code != 200:
                    error_text = response.text
                    raise httpx.HTTPStatusError(
                        f"Unexpected status code {response.status_code}: {error_text}",
                        request=response.request,
                        response=response,
                    )
                await self.emit_status(
                    event_emitter, "Received response from RAG API.", done=False
                )
                result = response.json()
                print(
                    f"API Response: {json.dumps(result, indent=2, ensure_ascii=False)}"
                )
                completion = result.get("results", {}).get("completion", {})
                choices = completion.get("choices", [])
                if choices and "message" in choices[0]:
                    content = choices[0].get("message", {}).get("content", "")
                    if isinstance(content, list):
                        # Process each item in the list
                        content_list = []
                        for item in content:
                            if isinstance(item, str):
                                content_list.append(item)
                            elif isinstance(item, dict):
                                item_content = item.get("content", "")
                                if isinstance(item_content, str):
                                    content_list.append(item_content)
                            else:
                                continue
                        content = " ".join(content_list)
                    elif isinstance(content, dict):
                        content = content.get("content", "")
                    elif not isinstance(content, str):
                        raise ValueError(
                            "'content' in response must be a string, list, or dict with 'content' key."
                        )
                    await self.emit_message(
                        event_emitter, "Received assistant response."
                    )
                    await self.emit_status(
                        event_emitter,
                        "RAG Pipeline successfully completed.",
                        done=True,
                    )
                    return content
                else:
                    await self.emit_status(
                        event_emitter,
                        "No response from the assistant.",
                        done=True,
                    )
                    return "No response from the assistant."
            except Exception as e:
                error_message = f"Error during non-streaming: {e}"
                print(error_message)
                await self.emit_status(event_emitter, error_message, done=True)
                raise

        async def pipe(
            self, body: dict, event_emitter: Optional[Callable[[dict], Any]] = None
        ) -> Union[str, AsyncGenerator[str, None]]:
            stream = body.get("kwargs", {}).get("stream", False)
            if stream:
                return self.pipe_stream(body, event_emitter)
            else:
                return await self.pipe_non_stream(body, event_emitter)

        async def emit_status(
            self,
            event_emitter: Optional[Callable[[dict], Any]],
            description: str,
            done: bool,
        ):
            if event_emitter:
                await event_emitter(
                    {
                        "type": "status",
                        "data": {"description": description, "done": done},
                    }
                )

        async def emit_message(
            self, event_emitter: Optional[Callable[[dict], Any]], content: str
        ):
            if event_emitter:
                await event_emitter({"type": "message", "data": {"content": content}})

        def extract_prompt(self, past_messages: List[dict]) -> str:
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
                            else:
                                continue
                        prompt = " ".join(content_list).strip()
                    elif isinstance(content, dict):
                        prompt = content.get("content", "").strip()
                    else:
                        continue
                    if prompt:
                        break  # Exit loop if prompt is found
            return prompt

        def build_payload(self, prompt: str) -> dict:
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
                    "model": summarizer_model_id,
                    "temperature": self.user_valves.temperature,
                    "top_p": self.user_valves.top_p,
                    "max_tokens_to_sample": self.user_valves.max_tokens,
                    "stream": False,  # Default to non-streaming
                },
                "task_prompt_override": None,
                "include_title_if_available": False,
            }
            return {"endpoint": endpoint, "payload": payload, "headers": headers}

    class FunctionCallingPipeline:
        class Valves(BaseModel):
            pipelines: List[str] = []
            priority: int = 0
            OPENAI_API_BASE_URL: str
            OPENAI_API_KEY: str
            TASK_MODEL: str
            TEMPLATE: str

        class Tools:
            def __init__(self, pipeline) -> None:
                self.pipeline = pipeline

            def get_current_time(self) -> str:
                """
                Get the current time.
                :return: The current time.
                """
                from datetime import datetime

                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                return f"Current Time = {current_time}"

            def get_current_weather(
                self,
                location: str,
                unit: Literal["metric", "fahrenheit"] = "fahrenheit",
            ) -> str:
                """
                Get the current weather for a location. If the location is not found, return an empty string.
                :param location: The location to get the weather for.
                :param unit: The unit to get the weather in. Default is fahrenheit.
                :return: The current weather for the location.
                """
                # https://openweathermap.org/api
                if self.pipeline.valves.OPENWEATHERMAP_API_KEY == "":
                    return "OpenWeatherMap API Key not set, ask the user to set it up."
                else:
                    units = "imperial" if unit == "fahrenheit" else "metric"
                    params = {
                        "q": location,
                        "appid": self.pipeline.valves.OPENWEATHERMAP_API_KEY,
                        "units": units,
                    }
                    response = requests.get(
                        "http://api.openweathermap.org/data/2.5/weather", params=params
                    )
                    response.raise_for_status()  # Raises an HTTPError for bad responses
                    data = response.json()
                    weather_description = data["weather"][0]["description"]
                    temperature = data["main"]["temp"]
                    return f"{location}: {weather_description.capitalize()}, {temperature}°{unit.capitalize()[0]}"

            def calculator(self, equation: str) -> str:
                """
                Calculate the result of an equation.
                :param equation: The equation to calculate.
                """
                # Avoid using eval in production code
                # https://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html
                try:
                    result = eval(equation)
                    return f"{equation} = {result}"
                except Exception as e:
                    print(e)
                    return "Invalid equation"

        def __init__(self):
            self.type = "filter"
            self.name = "Function Calling Blueprint"
            self.valves = self.Valves(
                pipelines=["*"],  # Connect to all pipelines
                OPENAI_API_BASE_URL=os.getenv(
                    "OPENAI_API_BASE_URL", "https://api.openai.com/v1"
                ),
                OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
                TASK_MODEL=os.getenv("TASK_MODEL", "gpt-3.5-turbo"),
                TEMPLATE="""Use the following context as your learned knowledge, inside <context></context> XML tags.
<context>
    {{CONTEXT}}
</context>
When answering the user:
- If you don't know, just say that you don't know.
- If you don't know when you are not sure, ask for clarification.
Avoid mentioning that you obtained the information from the context.
And answer according to the language of the user's question.""",
            )
            self.tools = self.Tools(self)

        async def on_startup(self):
            # This function is called when the server is started.
            print(f"on_startup:{__name__}")
            pass

        async def on_shutdown(self):
            # This function is called when the server is stopped.
            print(f"on_shutdown:{__name__}")
            pass

        async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
            # If title generation is requested, skip the function calling filter
            if body.get("title", False):
                return body
            print(f"pipe:{self.name}")
            print(user)
            # Get the last user message
            user_message = get_last_user_message(body.get("messages", []))
            # Get the tools specs
            tools_specs = get_tools_specs(self.tools)
            # System prompt for function calling
            fc_system_prompt = (
                f"Tools: {json.dumps(tools_specs, indent=2)}"
                + """
If a function tool doesn't match the query, return an empty string. Else, pick a function tool, fill in the parameters from the function tool's schema, and return it in the format { "name": "functionName", "parameters": { "key": "value" } }. Only pick a function if the user asks. Only return the object. Do not return any other text."
"""
            )
            r = None
            try:
                # Call the OpenAI API to get the function response
                r = requests.post(
                    url=f"{self.valves.OPENAI_API_BASE_URL}/chat/completions",
                    json={
                        "model": self.valves.TASK_MODEL,
                        "messages": [
                            {
                                "role": "system",
                                "content": fc_system_prompt,
                            },
                            {
                                "role": "user",
                                "content": "History:\n"
                                + "\n".join(
                                    [
                                        f"{message.role}: {message.content}"
                                        for message in reversed(
                                            body.get("messages", [])
                                        )[:4]
                                    ]
                                )
                                + f"\nQuery: {user_message}",
                            },
                        ],
                        # TODO: dynamically add response_format?
                        # "response_format": {"type": "json_object"},
                    },
                    headers={
                        "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    stream=False,
                )
                r.raise_for_status()
                response = r.json()
                content = response["choices"][0]["message"]["content"]
                # Parse the function response
                if content != "":
                    result = json.loads(content)
                    print(result)
                    # Call the function
                    if "name" in result:
                        function = getattr(self.tools, result["name"])
                        function_result = None
                        try:
                            function_result = function(**result["parameters"])
                        except Exception as e:
                            print(e)
                        # Add the function result to the system prompt
                        if function_result:
                            system_prompt = self.valves.TEMPLATE.replace(
                                "{{CONTEXT}}", function_result
                            )
                            print(system_prompt)
                            messages = add_or_update_system_message(
                                system_prompt, body.get("messages", [])
                            )
                            # Return the updated messages
                            return {**body, "messages": messages}
            except Exception as e:
                print(f"Error: {e}")
                if r:
                    try:
                        print(r.json())
                    except:
                        pass
            return body

    class MemoryFilterPipeline:
        class Valves(BaseModel):
            pipelines: List[str] = []
            priority: int = 0
            store_cycles: int = 5  # Number of user messages before storing
            mem_zero_user: str = "user"
            # mem0 vector store defaults
            vector_store_qdrant_name: str = "memories"
            vector_store_qdrant_url: str = "host.docker.internal"
            vector_store_qdrant_port: int = 6333
            vector_store_qdrant_dims: int = 768
            # mem0 language model defaults
            ollama_llm_model: str = "llama3.1:latest"
            ollama_llm_temperature: float = 0
            ollama_llm_tokens: int = 8000
            ollama_llm_url: str = "http://host.docker.internal:11434"
            # mem0 embedder defaults
            ollama_embedder_model: str = "nomic-embed-text:latest"
            ollama_embedder_url: str = "http://host.docker.internal:11434"

        def __init__(self):
            self.type = "filter"
            self.name = "Memory Filter"
            self.user_messages = []
            self.thread = None
            self.valves = self.Valves(
                pipelines=["*"],  # Connect to all pipelines
            )
            self.m = self.init_mem_zero()

        async def on_startup(self):
            print(f"on_startup:{__name__}")
            pass

        async def on_shutdown(self):
            print(f"on_shutdown:{__name__}")
            pass

        async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
            print(f"pipe:{self.name}")
            user = self.valves.mem_zero_user
            store_cycles = self.valves.store_cycles
            if isinstance(body, str):
                body = json.loads(body)
            all_messages = body.get("messages", [])
            if not all_messages:
                return body
            last_message = all_messages[-1].get("content", "")
            self.user_messages.append(last_message)
            if len(self.user_messages) >= store_cycles:
                message_text = " ".join(self.user_messages)
                if self.thread and self.thread.is_alive():
                    print("Waiting for previous memory to be done")
                    self.thread.join()
                self.thread = threading.Thread(
                    target=self.m.add, kwargs={"data": message_text, "user_id": user}
                )
                print("Text to be processed into memory:")
                print(message_text)
                self.thread.start()
                self.user_messages.clear()
            memories = self.m.search(last_message, user_id=user)
            fetched_memory = memories[0].get("memory", "") if memories else ""
            print("Memory added to the context:")
            print(fetched_memory)
            if fetched_memory:
                all_messages.insert(
                    0,
                    {
                        "role": "system",
                        "content": f"This is your inner voice talking, you remember this about the person you are chatting with: {fetched_memory}",
                    },
                )
            print("Final body to send to the LLM:")
            print(body)
            return body

        def init_mem_zero(self):
            config = {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": self.valves.vector_store_qdrant_name,
                        "host": self.valves.vector_store_qdrant_url,
                        "port": self.valves.vector_store_qdrant_port,
                        "embedding_model_dims": self.valves.vector_store_qdrant_dims,
                    },
                },
                "llm": {
                    "provider": "ollama",
                    "config": {
                        "model": self.valves.ollama_llm_model,
                        "temperature": self.valves.ollama_llm_temperature,
                        "max_tokens": self.valves.ollama_llm_tokens,
                        "ollama_base_url": self.valves.ollama_llm_url,
                    },
                },
                "embedder": {
                    "provider": "ollama",
                    "config": {
                        "model": self.valves.ollama_embedder_model,
                        "ollama_base_url": self.valves.ollama_embedder_url,
                    },
                },
            }
            return Memory.from_config(config)

    class PythonCodePipeline:
        def __init__(self):
            self.name = "Python Code Pipeline"

        async def on_startup(self):
            # This function is called when the server is started.
            print(f"on_startup:{__name__}")
            pass

        async def on_shutdown(self):
            # This function is called when the server is stopped.
            print(f"on_shutdown:{__name__}")
            pass

        def execute_python_code(self, code: str) -> (str, int):
            try:
                result = subprocess.run(
                    ["python", "-c", code], capture_output=True, text=True, check=True
                )
                stdout = result.stdout.strip()
                return stdout, result.returncode
            except subprocess.CalledProcessError as e:
                return e.output.strip(), e.returncode

        def pipe(
            self, user_message: str, model_id: str, messages: List[dict], body: dict
        ) -> Union[str, Generator, Iterator]:
            # Here you can add your custom pipelines like RAG.
            print(f"pipe:{self.name}")
            print(messages)
            print(user_message)
            if body.get("title", False):
                print("Title Generation")
                return "Python Code Pipeline"
            else:
                stdout, return_code = self.execute_python_code(user_message)
                return stdout

    class LlamaIndexPipeline:
        class Valves(BaseModel):
            LLAMAINDEX_OLLAMA_BASE_URL: str
            LLAMAINDEX_MODEL_NAME: str
            LLAMAINDEX_EMBEDDING_MODEL_NAME: str

        def __init__(self):
            self.documents = None
            self.index = None
            self.valves = self.Valves(
                LLAMAINDEX_OLLAMA_BASE_URL=os.getenv(
                    "LLAMAINDEX_OLLAMA_BASE_URL", "http://localhost:11434"
                ),
                LLAMAINDEX_MODEL_NAME=os.getenv("LLAMAINDEX_MODEL_NAME", "llama3"),
                LLAMAINDEX_EMBEDDING_MODEL_NAME=os.getenv(
                    "LLAMAINDEX_EMBEDDING_MODEL_NAME", "nomic-embed-text"
                ),
            )

        async def on_startup(self):
            from llama_index.embeddings.ollama import OllamaEmbedding
            from llama_index.llms.ollama import Ollama
            from llama_index.core import (
                Settings,
                VectorStoreIndex,
                SimpleDirectoryReader,
            )

            # Setup models in Llama Index
            Settings.embed_model = OllamaEmbedding(
                model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,
                base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
            )
            Settings.llm = Ollama(
                model=self.valves.LLAMAINDEX_MODEL_NAME,
                base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
            )

            # Load documents and create index
            self.documents = SimpleDirectoryReader("/app/backend/data").load_data()
            self.index = VectorStoreIndex.from_documents(self.documents)
            print(f"Llama Index initialized with {len(self.documents)} documents.")

        async def on_shutdown(self):
            # This function is called when the server is stopped.
            print(f"on_shutdown:{__name__}")
            pass

        def pipe(
            self, user_message: str, model_id: str, messages: List[dict], body: dict
        ) -> Union[str, Generator, Iterator]:
            # Here you can add your custom RAG pipeline.
            # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.
            print(messages)
            print(user_message)
            query_engine = self.index.as_query_engine(streaming=True)
            response = query_engine.query(user_message)
            return response.response_gen

    # 4. Utility Functions
    def get_last_user_message(messages: List[dict]) -> str:
        """
        Get the last user message from the list of messages.
        """
        for message in reversed(messages):
            if message.get("role") == "user" and "content" in message:
                return message["content"]
        return ""

    def add_or_update_system_message(
        system_prompt: str, messages: List[dict]
    ) -> List[dict]:
        """
        Add or update a system message in the list of messages.
        """
        for message in messages:
            if message.get("role") == "system":
                message["content"] = system_prompt
                return messages
        # If no system message exists, add a new one at the beginning
        messages.insert(0, {"role": "system", "content": system_prompt})
        return messages

    def get_tools_specs(tools) -> List[dict]:
        """
        Get the tool specifications for the function calling pipeline.
        """
        tool_specs = []
        for tool_name in dir(tools):
            tool = getattr(tools, tool_name)
            if callable(tool):
                tool_specs.append(
                    {
                        "name": tool_name,
                        "description": tool.__doc__,
                        # You can add parameters if necessary
                    }
                )
        return tool_specs

    # 5. Интеграция всех конвейеров в общий Pipeline
    class OpenWebUIPipeline:
        def __init__(self):
            # Initialize all pipelines
            self.rag_pipeline = self.RAGPipeline()
            self.function_calling_pipeline = self.FunctionCallingPipeline()
            self.memory_filter_pipeline = self.MemoryFilterPipeline()
            self.python_code_pipeline = self.PythonCodePipeline()
            self.llama_index_pipeline = self.LlamaIndexPipeline()

        async def on_startup(self):
            # Startup all pipelines
            await self.rag_pipeline.on_startup()
            await self.function_calling_pipeline.on_startup()
            await self.memory_filter_pipeline.on_startup()
            await self.python_code_pipeline.on_startup()
            await self.llama_index_pipeline.on_startup()

        async def on_shutdown(self):
            # Shutdown all pipelines
            await self.rag_pipeline.on_shutdown()
            await self.function_calling_pipeline.on_shutdown()
            await self.memory_filter_pipeline.on_shutdown()
            await self.python_code_pipeline.on_shutdown()
            await self.llama_index_pipeline.on_shutdown()

        async def handle_request(self, chat_request: ChatRequest) -> ChatResponse:
            try:
                # Convert the Pydantic model to a dictionary
                data = chat_request.dict()

                # Process through Memory Filter
                data = await self.memory_filter_pipeline.inlet(body=data)

                # Process through Function Calling
                data = await self.function_calling_pipeline.inlet(body=data)

                # Process through RAG Pipeline (streaming or not)
                rag_response = await self.rag_pipeline.pipe(
                    data, self.example_event_emitter
                )

                if isinstance(rag_response, AsyncGenerator):
                    # Streaming response
                    streamed_response = []
                    async for chunk in rag_response:
                        streamed_response.append(chunk)
                    final_response = " ".join(streamed_response)
                else:
                    # Non-streaming response
                    final_response = rag_response

                # Optionally, process through Python Code Pipeline
                python_response = self.python_code_pipeline.pipe(
                    user_message=get_last_user_message(data.get("messages", [])),
                    model_id="model_id",
                    messages=data.get("messages", []),
                    body=data,
                )

                # Optionally, process through Llama Index Pipeline
                llama_response = self.llama_index_pipeline.pipe(
                    user_message=get_last_user_message(data.get("messages", [])),
                    model_id="model_id",
                    messages=data.get("messages", []),
                    body=data,
                )

                return ChatResponse(status="success", response=final_response)

            except KeyError as e:
                error_message = str(e)
                return ChatResponse(status="error", error=error_message)
            except Exception as e:
                error_message = f"An error occurred during request processing: {str(e)}"
                return ChatResponse(status="error", error=error_message)

        # Example event emitter
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

    # 6. Интеграция с FastAPI
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI()
    pipeline = OpenWebUIPipeline()

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

    # 7. Дополнительные Конвейеры
    # Если необходимо, добавьте другие конвейеры сюда или расширьте существующие классы.

    # Пример использования конвейера в отдельном скрипте (опционально)
    if __name__ == "__main__":

        async def test_pipeline():
            from schemas import ChatRequest

            test_request = ChatRequest(
                session_id="12345",
                messages=[
                    OpenAIChatMessage(
                        role="user", content="Hello, can you tell me about Aristotle?"
                    )
                ],
                kwargs={"stream": False},
            )
            response = await pipeline.handle_request(test_request)
            print(response)

        asyncio.run(test_pipeline())


# ### Пояснения к Объединенному Конвейеру

# 1. **Динамический импорт устаревших компонентов:**
#     - Используется `create_importer` из `langchain._api` для динамического импорта устаревших атрибутов.
#     - `DEPRECATED_LOOKUP` определяет соответствие между устаревшими атрибутами и их новыми модулями.
#     - Определена функция `__getattr__` для динамического поиска атрибутов.

# 2. **Pydantic модели:**
#     - `OpenAIChatMessage`: структура сообщения от OpenAI.
#     - `ChatRequest`: структура входящего запроса от пользователя.
#     - `ChatResponse`: структура ответа, возвращаемого пользователю.

# 3. **Основной класс `Pipeline`:**
#     - Объединяет все конвейеры в один.
#     - Внутренние классы `RAGPipeline`, `FunctionCallingPipeline`, `MemoryFilterPipeline`, `PythonCodePipeline`, и `LlamaIndexPipeline` реализуют соответствующую функциональность.

# 4. **RAG Pipeline (`RAGPipeline`):**
#     - Обрабатывает запросы с использованием RAG через API.
#     - Поддерживает как стриминговую, так и нестиминговую обработку.
#     - Включает методы `pipe_stream` и `pipe_non_stream` для соответствующих режимов.

# 5. **Function Calling Pipeline (`FunctionCallingPipeline`):**
#     - Обрабатывает вызовы функций на основе ответов от OpenAI.
#     - Включает методы для получения текущего времени, погоды и выполнения простых вычислений.

# 6. **Memory Filter Pipeline (`MemoryFilterPipeline`):**
#     - Управляет долгосрочной памятью пользователя с использованием `mem0`.
#     - Сохраняет сообщения пользователя периодически для последующего использования в контексте.

# 7. **Python Code Pipeline (`PythonCodePipeline`):**
#     - Позволяет выполнять Python-код, предоставленный пользователем.
#     - Время выполнения кода обрабатывается через `subprocess`.

# 8. **Llama Index Pipeline (`LlamaIndexPipeline`):**
#     - Интегрируется с Llama Index и Ollama для обработки запросов.
#     - Создает индекс документов и выполняет запросы на основе сообщений пользователя.

# 9. **Утилиты:**
#     - `get_last_user_message`: извлекает последнее сообщение пользователя.
#     - `add_or_update_system_message`: добавляет или обновляет системное сообщение в цепочке сообщений.
#     - `get_tools_specs`: собирает спецификации доступных инструментов для конвейера вызова функций.

# 10. **Интеграция с FastAPI:**
#     - Создается экземпляр `OpenWebUIPipeline`, который управляет всеми конвейерами.
#     - Определены обработчики событий `startup` и `shutdown` для инициализации и завершения работы всех конвейеров.
#     - Эндпоинт `/chat/completed` принимает запросы, обрабатывает их через все конвейеры и возвращает ответ.

# ### Рекомендации и Лучшие Практики

# 1. **Использование `logging` вместо `print`:**
#     - Для более гибкого и управляемого логирования рекомендуется использовать модуль `logging`.

#     ```python
#     import logging

#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)

#     # Пример замены print на logging
#     logger.info("Starting RAG Pipeline (Streaming)...")
#     ```

# 2. **Безопасность при выполнении Python-кода:**
#     - Использование `eval` может быть опасным, если ввод пользователя не контролируется.
#     - Рассмотрите возможность ограничения выполняемого кода или использования безопасных альтернатив.

# 3. **Валидация входящих данных:**
#     - Pydantic модели уже обеспечивают базовую валидацию.
#     - Можно расширить валидацию, добавив дополнительные проверки внутри конвейеров.

# 4. **Асинхронность и Производительность:**
#     - Убедитесь, что все I/O операции выполняются асинхронно, чтобы не блокировать основной поток.

# 5. **Тестирование:**
#     - Напишите тесты для каждого конвейера, чтобы гарантировать их корректную работу в различных сценариях.

# 6. **Разделение Ответственности:**
#     - Хотя объединение всех конвейеров в один файл удобно, для больших проектов лучше разделять функциональность на несколько файлов для улучшения читаемости и поддержки.

# ### Заключение

# Представленный пример демонстрирует, как объединить несколько конвейеров в один файл для `open-webui`, обеспечивая модульность и масштабируемость системы. Следуя приведенным рекомендациям, вы сможете адаптировать и расширить этот конвейер под свои конкретные нужды.

# Если у вас возникнут дополнительные вопросы или потребуется помощь с конкретными частями кода, пожалуйста, дайте знать!
