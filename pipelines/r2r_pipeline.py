"""
title: R2R Integration Pipeline
author: Your Name
date: 2023-10-05
version: 1.0
license: MIT
description: A pipeline integrating R2R for search, RAG, and agent functionalities with dynamic parameter configuration.
requirements: r2r, litellm, langchain, unstructured
"""

from typing import List, Optional, Union, Generator, Iterator
from pydantic import BaseModel
import os
import requests
import json

# Import R2R and LangChain libraries according to the documentation
from r2r import R2RBuilder, R2RConfig, R2RAgent
from r2r.providers import CustomAuthProvider, CustomDatabaseProvider
from r2r.pipes import (
    CustomParsingPipe,
    CustomEmbeddingPipe,
    MultiSearchPipe,
    QueryTransformPipe,
)
from r2r.pipelines import CustomIngestionPipeline, CustomSearchPipeline
from r2r.factory import CustomProviderFactory, CustomPipeFactory, CustomPipelineFactory
from r2r.factory import (
    E2EPipelineFactory,
)  # E2EPipelineFactory is imported from r2r.factory

from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI

# Remove or adjust this import if not necessary
# from schemas import OpenAIChatMessage

from utils.pipelines.main import (
    get_last_user_message,
    add_or_update_system_message,
    get_tools_specs,
)


class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
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
        self.name = "R2R Integration Pipeline"
        self.valves = self.Valves(
            pipelines=["*"],
            OPENAI_API_BASE_URL=self.Valves.OPENAI_API_BASE_URL,
            OPENAI_API_KEY=self.Valves.OPENAI_API_KEY,
            TASK_MODEL=self.Valves.TASK_MODEL,
            TEMPLATE=self.Valves.TEMPLATE,
        )
        self.setup_pipeline()

    def setup_pipeline(self):
        config = R2RConfig.from_toml(
            "path/to/your/r2r.toml"
        )  # Load your custom configuration
        builder = R2RBuilder(config=config)

        # Customize your providers, pipes, and pipelines as needed
        builder.with_embedding_provider(CustomEmbeddingPipe())
        builder.with_parsing_pipe(CustomParsingPipe())
        builder.with_ingestion_pipeline(CustomIngestionPipeline())
        builder.with_search_pipeline(CustomSearchPipeline())
        builder.with_provider_factory(CustomProviderFactory)
        builder.with_pipe_factory(CustomPipeFactory)
        builder.with_pipeline_factory(CustomPipelineFactory)

        # Build the R2R application
        self.r2r = builder.build()

        # Initialize the R2R agent
        self.agent = R2RAgent(config)

        # Access the app (FastAPI application) if needed
        self.app = self.r2r.app

    async def on_startup(self):
        print(f"on_startup: {__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown: {__name__}")

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        if "configure" in user_message.lower():
            self.configure_params(user_message)
            return "Configuration updated successfully."
        try:
            # Process the query using the R2R RAG pipeline
            response = self.r2r.search_pipeline.search(user_message)
            return response
        except Exception as e:
            print(f"Error processing RAG: {e}")
            return "An error occurred while processing your request."

    def configure_params(self, instruction):
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
            # Update the configuration with the extracted parameters
            # This may involve updating self.r2r.config or rebuilding the pipeline
            self.agent.config.update(params)
            self.setup_pipeline()
        except json.JSONDecodeError:
            print("Failed to parse configuration parameters.")
