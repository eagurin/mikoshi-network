from typing import Union, AsyncGenerator, Optional, Callable, Any
from pydantic import BaseModel, Field
import os
import httpx
import json
import asyncio
import logging

logger = logging.getLogger(__name__)

class Pipeline:
    class Valves(BaseModel):
        API_BASE_URL: str = Field(
            default="https://api.durka.dndstudio.ru",
            description="Base API URL",
        )
        TIMEOUT: int = Field(
            default=180,
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
        self.name = "RAG Pipeline"
        self.type = "pipe"  # Or "manifold" if needed
        # Load configuration from environment variables or use default values
        self.valves = self.Valves(
            API_BASE_URL=os.getenv("API_BASE_URL", "https://api.durka.dndstudio.ru"),
            TIMEOUT=int(os.getenv("TIMEOUT", "180")),
            summarizer_model_id=os.getenv("SUMMARIZER_MODEL_ID", "gpt-4o-mini"),
        )
        self.user_valves = self.UserValves()
        # Disable SSL verification by setting verify=False here
        self.client = httpx.AsyncClient(timeout=self.valves.TIMEOUT, verify=False)

    def pipelines(self) -> list:
        return []

    async def run(
        self,
        body: dict,
        websocket_conn=None,
        request=None,
    ) -> Union[str, AsyncGenerator[str, None]]:
        stream = body.get("kwargs", {}).get("stream", False)
        if stream:
            return await self.pipe_stream(body)
        else:
            return await self.pipe_non_stream(body)

    async def pipe_stream(self, body: dict) -> AsyncGenerator[str, None]:
        try:
            messages = body.get("messages", [])
            if not messages:
                raise ValueError("Empty 'messages' list in 'body'.")

            memory_limit = self.user_valves.memory_limit
            past_messages = messages[-memory_limit:]

            # Extract the latest user message
            prompt = self.extract_prompt(past_messages)
            if not prompt:
                raise ValueError("Prompt cannot be empty!")

            payload = self.build_payload(prompt, stream=True)
            endpoint = f"{self.valves.API_BASE_URL}/v2/rag"
            headers = {"Content-Type": "application/json"}

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
        except Exception as e:
            error_message = f"Error during streaming: {e}"
            logger.exception(error_message)
            yield f"Error: {error_message}"

    async def pipe_non_stream(self, body: dict) -> str:
        try:
            messages = body.get("messages", [])
            if not messages:
                raise ValueError("Empty 'messages' list in 'body'.")

            memory_limit = self.user_valves.memory_limit
            past_messages = messages[-memory_limit:]

            # Extract the latest user message
            prompt = self.extract_prompt(past_messages)
            if not prompt:
                raise ValueError("Prompt cannot be empty!")

            payload = self.build_payload(prompt, stream=False)
            endpoint = f"{self.valves.API_BASE_URL}/v2/rag"
            headers = {"Content-Type": "application/json"}

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

            result = response.json()
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
                    content = " ".join(content_list)
                elif isinstance(content, dict):
                    content = content.get("content", "")
                elif not isinstance(content, str):
                    raise ValueError(
                        "'content' in response must be a string, list, or dict with 'content' key."
                    )
                return content
            else:
                return "No response from the assistant."
        except Exception as e:
            error_message = f"Error during non-streaming: {e}"
            logger.exception(error_message)
            return f"Error: {error_message}"

    def extract_prompt(self, messages: list) -> str:
        # Extract the latest user message
        prompt = ""
        for message in reversed(messages):
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
                    break  # Exit loop if prompt is found
        return prompt

    def build_payload(self, prompt: str, stream: bool) -> dict:
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
                    "full_text_limit": 500,
                    "rrf_k": 50,
                },
                "search_strategy": "hyde",
            },
            "rag_generation_config": {
                "model": self.valves.summarizer_model_id,
                "temperature": self.user_valves.temperature,
                "top_p": self.user_valves.top_p,
                "max_tokens_to_sample": self.user_valves.max_tokens,
                "stream": stream,
            },
            "task_prompt_override": None,
            "include_title_if_available": False,
        }
        return payload
