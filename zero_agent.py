import asyncio
from langchain.agents import initialize_agent, AgentType
from langchain_openai import OpenAI
from r2r import R2RClient

async def quick_search(query):
    # Инициализация клиента R2R для быстрого поиска
    r2r_client = R2RClient("https://api.mikoshi.company")
    results = r2r_client.search(query)
    return results

async def long_running_agent(query):
    # Создание агента LangChain для длительной задачи
    llm = OpenAI(model_name="gpt-4o-mini", temperature=0.7)
    agent = initialize_agent(
        tools=[],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    response = agent.run(query)
    return response

async def main(query):
    # Запуск быстрого поиска и вывод результата
    quick_result_task = asyncio.create_task(quick_search(query))
    quick_result = await quick_result_task
    print("Результаты быстрого поиска:")
    print(quick_result)

    # Запуск длительной задачи агента
    long_task = asyncio.create_task(long_running_agent(query))

    # Пока длительный агент выполняется, можно уведомлять пользователя или выполнять другие действия
    while not long_task.done():
        print("Длительный агент выполняется...")
        await asyncio.sleep(1)

    # Получение результата от длительного агента
    long_result = await long_task
    print("Результаты длительного агента:")
    print(long_result)

if __name__ == "__main__":
    user_query = "Расскажите мне об основных философских идеях Аристотеля."
    asyncio.run(main(user_query))
