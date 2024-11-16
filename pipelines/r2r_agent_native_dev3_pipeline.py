import asyncio
import time
import os
import logging
from typing import Dict
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# Настройка логгера для отслеживания всех действий
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


# Асинхронная быстрая задача для быстрого ответа
async def quick_response_task(query: str) -> str:
    logging.info(f"Старт быстрой задачи для запроса: {query}")

    # Симуляция быстрого запроса, который просто занимает клиента
    await asyncio.sleep(1)  # Симулируем небольшое ожидание
    quick_reply = f"Мы получили ваш запрос: '{query}'. Пожалуйста, подождите, ответ готовится..."

    logging.info(f"Быстрая задача завершена. Ответ: {quick_reply}")
    return quick_reply


# Асинхронная задача агента для детального ответа
async def agent_response_task(query: str) -> str:
    logging.info(f"Запуск задачи агента для запроса: {query}")

    # Используем OpenAI через LangChain для получения более сложного ответа
    llm = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "")
    )  # Вставьте свой реальный API ключ
    prompt_template = PromptTemplate.from_template(
        "Сгенерируй детальный ответ для вопроса: {query}"
    )

    # Генерация детального ответа через цепочку LangChain
    chain = LLMChain(llm=llm, prompt=prompt_template)

    # Симулируем долгий процесс генерации ответа
    result = await chain.arun({"query": query})

    logging.info(f"Ответ агента подготовлен: {result}")
    return result


# Главная асинхронная функция для параллельной обработки запросов
async def main_pipeline(query: str) -> Dict[str, str]:
    logging.info("Запуск основного Pipeline")

    # Запускаем одновременно быструю и агентную задачу
    quick_response = asyncio.create_task(quick_response_task(query))
    agent_response = asyncio.create_task(agent_response_task(query))

    # Ожидаем завершения быстрой задачи
    quick_reply = await quick_response
    logging.info("Быстрая задача выполнена, отправляем быстрый ответ клиенту")
    print(f"Быстрый ответ клиенту: {quick_reply}")

    # Параллельно готовим детализированный ответ от агента
    detailed_reply = await agent_response
    logging.info(
        "Задача агента выполнена, отправляем детализированный ответ клиенту"
    )
    print(f"Детализированный ответ клиенту: {detailed_reply}")

    return {"quick_reply": quick_reply, "detailed_reply": detailed_reply}


# Функция запуска пайплайна
async def run_pipeline():
    query = "Расскажи мне о квантовых компьютерах и их использовании"
    logging.info(f"Получен запрос: {query}")

    # Запуск пайплайна с запросом и получением ответов
    responses = await main_pipeline(query)

    logging.info("Процесс завершён")
    print(
        f"Все ответы:\n- Быстрый: {responses['quick_reply']}\n- Детализированный: {responses['detailed_reply']}"
    )


# Старт программы
if __name__ == "__main__":
    # Используем asyncio для запуска пайплайна
    logging.info("Запуск программы...")

    start_time = time.time()  # Засекаем время начала
    asyncio.run(run_pipeline())  # Запускаем асинхронный пайплайн

    # Вывод финального времени исполнения
    total_time = time.time() - start_time
    logging.info(f"Программа завершена за {total_time:.2f} секунд")
