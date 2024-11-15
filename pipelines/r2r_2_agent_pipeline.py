"""
title: Example Filter
author: open-webui
author_url: https://github.com/open-webui
funding_url: https://github.com/open-webui
version: 0.1
requirements: r2r, langchain
"""

from typing import List, Dict, Any, Union
import os
import json
from pydantic import BaseModel, Field
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI  # Replace with your preferred LLM provider
from r2r import R2RClient

class Pipeline:
    class Valves(BaseModel):
        R2R_API_URL: str = Field(
            default_factory=lambda: os.getenv("R2R_API_URL", "http://localhost:7272"),
            description="R2R API URL"
        )
        R2R_API_KEY: str = Field(
            default_factory=lambda: os.getenv("R2R_API_KEY", ""),
            description="API key for R2R"
        )

    class UserValves(BaseModel):
        # Define user-specific settings if needed
        pass

    def __init__(self):
        self.type = "pipe"
        self.id = "r2r_pipeline"
        self.name = "R2R Pipeline"
        self.valves = self.Valves()
        # Initialize R2R client
        self.client = R2RClient(
            api_base=self.valves.R2R_API_URL,
            api_key=self.valves.R2R_API_KEY
        )
        # Initialize LLM for dynamic parameter configuration
        self.llm = OpenAI(temperature=0.7)  # Configure with your API key and settings

    def get_provider_models(self) -> List[Dict[str, str]]:
        return [
            {"id": "agent", "name": "R2R Agent"},
            {"id": "rag", "name": "R2R RAG"},
            {"id": "search", "name": "R2R Search"},
        ]

    def pipe(self, body: Dict[str, Any]) -> Union[str, Dict[str, Any]]:
        # Get the selected model ID from the body, default to 'agent'
        model_id = body.get("model", "agent")
        # Get conversation history
        messages = body.get("messages", [])
        if not messages:
            return "No input messages found."

        user_message = messages[-1].get("content", "")

        # Use LangChain to dynamically configure parameters based on user message
        prompt_template = PromptTemplate(
            input_variables=["message", "model_id"],
            template="""
As an AI assistant, analyze the user's message below and generate a JSON configuration
for the R2R request parameters suitable for the '{model_id}' model.
Ensure the JSON is properly formatted.

User Message:
{message}
""".strip()
        )
        chain = LLMChain(prompt=prompt_template, llm=self.llm)
        config_prompt = chain.run(message=user_message, model_id=model_id)

        # Parse config_parameters from the LLM's output
        try:
            config_parameters = json.loads(config_prompt)
        except json.JSONDecodeError:
            config_parameters = {}

        # If the R2R agent needs to configure itself
        if not config_parameters:
            config_parameters = {}

        # Handle different models
        if model_id == "agent":
            response = self.client.agent(
                messages=messages,
                **config_parameters
            )
            # Return the assistant's reply
            return response.get('results', {}).get('completion', "No response.")
        elif model_id == "rag":
            response = self.client.rag(
                query=user_message,
                **config_parameters
            )
            return response.get('results', {}).get('completion', "No response.")
        elif model_id == "search":
            response = self.client.search(
                query=user_message,
                **config_parameters
            )
            return response.get('results', [])
        else:
            return f"Unknown model ID: {model_id}"