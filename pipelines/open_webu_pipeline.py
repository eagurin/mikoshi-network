"""
title: R2R OpenAPI LangChain
date: 2024-05-30
version: 1.0
license: MIT
author: evgeny
version: 1.0.0
description: >
  Этот pipeline выполняет запросы пользователей, используя OpenAPI и LangChain для
  интеграции с R2R API. Основные задачи — это извлечение информации через OpenAPI, 
  создание контекста с помощью поиска и обеспечение полного комплекса ответов через GPT-3.5 или GPT-4.
  Пример использования — генерация ответов на вопросы пользователей при помощи данных R2R.
requirements: >
  - pydantic < 2.0.0
  - fastapi >= 0.78.0
  - langchain >= 0.1.0
  - openai >= 0.27.0
  - r2r >= 1.0.0
  - supabase
  - litellm
"""

import json
from typing import List, Optional
import asyncio
import requests
from pydantic import BaseModel
from r2r import R2RClient
from langchain.agents import initialize_agent, AgentType
from langchain import OpenAI
from langchain.tools.openapi import OpenAPISpec

# 1. Загрузка спецификации OpenAPI
API_SPEC_URL = "https://api.mikoshi.company/v2/openapi_spec"
R2R_API_BASE_URL = "https://api.mikoshi.company"

def load_openapi_spec(api_spec_url):
    """Загрузить спецификацию OpenAPI"""
    try:
        response = requests.get(api_spec_url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Ошибка при загрузке API: {e}")
        return None

# 2. Модель данных для запросов
class PipelineRequest(BaseModel):
    """Модель для обработки входящих запросов в конвейере"""
    session_id: str
    messages: List[dict]

# 3. Основной класс конвейера, который интегрируется в Open-WebUI
class Pipeline:
    """Основной pipeline для Open-WebUI с использованием OpenAPI и R2R"""
    
    def __init__(self):
        # Загрузка спецификации OpenAPI и инициализация OpenAPI инструмента
        openapi_spec = load_openapi_spec(API_SPEC_URL)
        if openapi_spec:
            self.openapi_tool = OpenAPISpec.from_spec_dict(openapi_spec)
        else:
            raise ValueError("Не удалось загрузить спецификацию OpenAPI.")
        
        # Инициализация LangChain агента
        print("Инициализация агента LangChain...")
        self.llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        self.agent = initialize_agent(
            tools=[self.openapi_tool],
            agent=AgentType.OPENAI_FUNCTIONS,
            llm=self.llm,
            verbose=True
        )
        
        # Инициализация клиента R2R для работы с API
        self.r2r_client = R2RClient(R2R_API_BASE_URL)
    
    async def inbound(self, body: dict) -> dict:
        """Метод обработки входящих данных через Open-WebUI Pipeline"""
        user_message = body["messages"][-1]["content"]  # Последнее сообщение пользователя
        try:
            # 1. Получение контекста через OpenAPI и LangChain агент
            print(f"Запрос данных через OpenAPI и LangChain: {user_message}")
            # Используем асинхронный вызов, если arun является асинхронным
            openapi_result = await self.agent.arun(user_message)
            print(f"Ответ от агента: {openapi_result}")
            
            # 2. Поиск контекста через R2R Client
            print(f"Поиск через API R2R...")
            # Предполагается, что метод search_documents_async является асинхронным
            r2r_context = await self.r2r_client.search_documents_async(query=user_message)
            r2r_result = r2r_context.get("context", "Контекст не получен.")
            print(f"Ответ через R2R API: {r2r_result}")
            
            # Возвращение финального результата, включающего ответы от OpenAPI и R2R
            final_response = f"Ответ через OpenAPI: {openapi_result}, Контекст через R2R: {r2r_result}"
            body["messages"].append({"role": "assistant", "content": final_response})
            return body
        
        except Exception as e:
            print(f"Ошибка: {str(e)}")
            return {"error": str(e)}

# 4. Интеграция этого pipeline в Open-WebUI как отдельный слой
# Если требуется отдельный класс для интеграции, используем другое имя
class OpenWebUIPipeline:
    """Класс для интеграции в Open-WebUI как отдельный слой"""
    
    def __init__(self):
        print("OpenWebUIPipeline инициализирован")
        self.pipeline = Pipeline()
    
    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        """Метод входа данных для Open-WebUI Pipeline"""
        print("Обработка входящих данных через OpenWebUIPipeline...")
        result = await self.pipeline.inbound(body)
        return result

# 5. Пример теста для выполнения взаимодействия с pipeline и Open-WebUI
async def pipeline_test():
    """Пример теста системы через эмуляцию запросов"""
    request_body = {
        "session_id": "session_123456",
        "messages": [{"role": "user", "content": "Какая погода сегодня в Токио?"}]
    }
    
    # Инициализация OpenWebUIPipeline
    open_web_ui_pipeline = OpenWebUIPipeline()
    
    # Вызов inlet и обработка запроса
    result = await open_web_ui_pipeline.inlet(request_body)
    print(f"Финальный результат:\n{json.dumps(result, indent=2, ensure_ascii=False)}")

# Запуск теста с использованием `asyncio` в основной функции
if __name__ == "__main__":
    asyncio.run(pipeline_test())
