"""
title: R2R Integration Pipeline (Test Version)
author: Your Name
date: 2023-10-05
version: 1.0-test
license: MIT
description: A pipeline integrating R2R for search, RAG, and agent functionalities with dynamic parameter configuration. This version includes stubs for testing imports and functionality.
requirements: r2r, litellm, langchain, unstructured
"""

from typing import List, Union, Generator, Iterator, Optional
from pydantic import BaseModel
import os
import requests
import json

# Try importing external modules; if not available, use stubs
try:
    from r2r.builder import R2RBuilder
    from r2r.config import R2RConfig
    from r2r.agent import R2RAgent
    from r2r.providers import (
        CustomAuthProvider,
        CustomDatabaseProvider,
    )
    from r2r.pipes import (
        CustomParsingPipe,
        CustomEmbeddingPipe,
        MultiSearchPipe,
        QueryTransformPipe,
    )
    from r2r.pipelines import (
        CustomIngestionPipeline,
        CustomSearchPipeline,
    )
    from r2r.factory import (
        E2EPipelineFactory,
        CustomProviderFactory,
        CustomPipeFactory,
        CustomPipelineFactory,
    )
except ImportError as e:
    print(f"ImportError: {e}")
    # Creating stubs for testing purposes
    class R2RBuilder:
        def __init__(self, config=None):
            pass

        def with_embedding_provider(self, provider):
            return self

        def with_parsing_pipe(self, pipe):
            return self

        def with_ingestion_pipeline(self, pipeline):
            return self

        def with_search_pipeline(self, pipeline):
            return self

        def with_provider_factory(self, factory):
            return self

        def with_pipe_factory(self, factory):
            return self

        def with_pipeline_factory(self, factory):
            return self

        def build(self):
            return self

    class R2RConfig:
        @staticmethod
        def from_toml(path):
            print(f"Loading config from {path}")
            return R2RConfig()

    class R2RAgent:
        def __init__(self, config):
            pass

        def config(self):
            pass

    # Stub classes for providers, pipes, and pipelines
    class CustomEmbeddingPipe:
        pass

    class CustomParsingPipe:
        pass

    class CustomIngestionPipeline:
        pass

    class CustomSearchPipeline:
        pass

    class CustomProviderFactory:
        pass

    class CustomPipeFactory:
        pass

    class CustomPipelineFactory:
        pass

    class E2EPipelineFactory:
        pass

# Attempt to import LangChain modules
try:
    from langchain import LLMChain, PromptTemplate
    from langchain.llms import OpenAI
except ImportError as e:
    print(f"ImportError: {e}")
    # Create stubs for LangChain
    class OpenAI:
        def __init__(self, api_key):
            pass

    class PromptTemplate:
        def __init__(self, input_variables, template):
            pass

    class LLMChain:
        def __init__(self, llm, prompt):
            pass

        def run(self, instruction):
            # Return a dummy JSON string
            return '{"dummy_key": "dummy_value"}'

# Attempt to import utility functions
try:
    from utils.pipelines.main import (
        get_last_user_message,
        add_or_update_system_message,
        get_tools_specs,
    )
except ImportError as e:
    print(f"ImportError: {e}")

    def get_last_user_message(messages):
        return messages[-1]['content'] if messages else ''

    def add_or_update_system_message(system_message, messages):
        return messages + [{'role': 'system', 'content': system_message}]

    def get_tools_specs(tools):
        return {}

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = ["*"]
        priority: int = 0
        OPENAI_API_BASE_URL: str = os.getenv(
            "OPENAI_API_BASE_URL", "https://api.openai.com/v1"
        )
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
        TASK_MODEL: str = os.getenv("TASK_MODEL", "gpt-3.5-turbo")
        TEMPLATE: str = """Use the following context as your learned knowledge, inside <context></context> XML tags.
<context>
    {{CONTEXT}}
</context>
When answering to user:
- If you don't know, just say that you don't know.
- If you are not sure, ask for clarification.
Avoid mentioning that you obtained the information from the context.
Answer according to the language of the user's question."""

    def __init__(self):
        self.type = "pipe"
        self.name = "R2R Integration Pipeline (Test Version)"
        self.valves = self.Valves()
        self.setup_pipeline()

    def setup_pipeline(self):
        try:
            # Load the configuration
            config = R2RConfig.from_toml("path/to/your/r2r.toml")
            builder = R2RBuilder(config=config)

            # Set up the pipeline using stubs
            builder.with_embedding_provider(CustomEmbeddingPipe())
            builder.with_parsing_pipe(CustomParsingPipe())
            builder.with_ingestion_pipeline(CustomIngestionPipeline())
            builder.with_search_pipeline(CustomSearchPipeline())
            builder.with_provider_factory(CustomProviderFactory())
            builder.with_pipe_factory(CustomPipeFactory())
            builder.with_pipeline_factory(CustomPipelineFactory())

            # Build the R2R application
            self.r2r = builder.build()

            # Initialize the R2R agent
            self.agent = R2RAgent(config)

            # Access the app (e.g., FastAPI app) if needed
            self.app = getattr(self.r2r, 'app', None)

            print("Pipeline setup completed.")
        except Exception as e:
            print(f"Exception during setup_pipeline: {e}")
            # Additional error handling if necessary

    async def on_startup(self):
        print(f"on_startup: {__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown: {__name__}")

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        try:
            if "configure" in user_message.lower():
                self.configure_params(user_message)
                return "Configuration updated successfully."
            else:
                # For testing, return a fixed response
                return "Pipeline is working with stubs."
        except Exception as e:
            print(f"Error in pipe method: {e}")
            return "An error occurred while processing your request."

    def configure_params(self, instruction):
        try:
            prompt = PromptTemplate(
                input_variables=["instruction"],
                template="""
You are an assistant that extracts configuration parameters from user instructions.
Instruction:
{instruction}
Extracted Configuration Parameters (in JSON format):
""",
            )
            llm = OpenAI(api_key=self.valves.OPENAI_API_KEY)
            chain = LLMChain(llm=llm, prompt=prompt)
            response = chain.run(instruction=instruction)
            try:
                params = json.loads(response.strip())
                # Update the configuration
                # Assuming self.agent.config supports an update method
                self.agent.config.update(params)
                self.setup_pipeline()
            except json.JSONDecodeError:
                print("Failed to parse configuration parameters.")
        except Exception as e:
            print(f"Error in configure_params: {e}")
            # Error handling

# Main block to run the code locally
if __name__ == '__main__':
    # Create an instance of the Pipeline class
    pipeline = Pipeline()

    # Example user message
    user_message = "Test message"

    # Model ID (can be any string for testing)
    model_id = "test-model"

    # Example message history
    messages = [
        {'role': 'user', 'content': user_message}
    ]

    # Example request body (can be empty or include necessary data)
    body = {}

    # Call the pipe method
    response = pipeline.pipe(user_message, model_id, messages, body)

    # Print the response
    print(f"Response from pipeline: {response}")
