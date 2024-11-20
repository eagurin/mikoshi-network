# === Импорты ==#
import os

import requests
from bs4 import BeautifulSoup
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.utilities import SearxSearchWrapper
from langchain.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# === API ключ для OpenAI ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_api_key")


# === Парсинг веб-сайтa с помощью BeautifulSoup ===
def parse_website(url: str) -> str:
    """
    Парсинг веб-сайта с извлечением заголовков и текста параграфов.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Проверка на ошибки
    except requests.HTTPError as e:
        return f"Ошибка HTTP: {e}"
    except requests.RequestException as e:
        return f"Ошибка сети: {e}"

    # Парсинг HTML страницы с помощью BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # Извлечение заголовков всех уровней
    titles = [
        heading.text.strip()
        for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
    ]

    # Извлечение абзацев
    paragraphs = [p.text.strip() for p in soup.find_all("p")]

    # Форматирование полученного содержимого
    result = "Заголовки:\n" + "\n".join(titles) + "\n\n"
    result += "Параграфы:\n" + "\n".join(paragraphs)

    return result


# === Парсинг с помощью Selenium для сайтов с JavaScript содержимым ===
def selenium_parse_website(url: str) -> str:
    """
    Использует Selenium для парсинга веб-сайтов с JavaScript содержимым.
    """
    options = webdriver.ChromeOptions()
    options.add_argument(
        "--headless"
    )  # Запускаем в фоновом режиме без UI браузера
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=options
    )

    # Перейти на сайт
    driver.get(url)

    # Получаем содержимое страницы
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    # Извлечение заголовков всех уровней
    titles = [
        heading.text.strip()
        for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
    ]

    # Извлечение абзацев
    paragraphs = [p.text.strip() for p in soup.find_all("p")]

    driver.quit()

    result = "Заголовки:\n" + "\n".join(titles) + "\n\n"
    result += "Параграфы:\n" + "\n".join(paragraphs)

    return result


# === Инструмент для LangChain для веб-скрапинга через BeautifulSoup ===
def web_scraping_tool(url: str) -> str:
    return parse_website(url)


# === Инструмент для LangChain для веб-скрапинга через Selenium ===
def selenium_tool(url: str) -> str:
    return selenium_parse_website(url)


# === Инициализация ChatOpenAI (LLM) ===
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY, model_name="gpt-4", temperature=0.5
)


# === Поисковый инструмент через Searx Search ===
class SearxNGSearch:
    def __init__(self, searx_host="https://search.durka.dndstudio.ru"):
        self.wrapper = SearxSearchWrapper(searx_host=searx_host)

    def search(self, query: str) -> str:
        return self.wrapper.run(query)


# Создаем инструмент для поиска через Searx
searx_tool = SearxNGSearch()


def searx_tool_wrap(query: str) -> str:
    return searx_tool.search(query)


# === Инструмент для использования RAG (Retrieval Augmented Generation) с FAISS ===
def create_faiss_index():
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    documents = [
        Document(
            page_content="Пример содержимого для поиска", metadata={"id": "1"}
        ),
        Document(
            page_content="Содержимое по архитектуре LangChain",
            metadata={"id": "2"},
        ),
    ]
    return FAISS.from_documents(documents, embeddings)


def rag_tool(query: str) -> str:
    vector_store = create_faiss_index()
    results = vector_store.similarity_search(query, k=5)
    return "\n".join([result.page_content for result in results])


# Интеграция всех инструментов через LangChain
tools = [
    Tool(
        name="Searx Search",
        func=searx_tool_wrap,
        description="Поиск информации через SearxNG.",
    ),
    Tool(
        name="WebScrapingTool",
        func=web_scraping_tool,
        description="Парсинг веб-сайтов через BeautifulSoup, извлечение заголовков и текста параграфов.",
    ),
    Tool(
        name="Selenium WebScraping Tool",
        func=selenium_tool,
        description="Парсинг динамических веб-сайтов с JavaScript содержанием через Selenium.",
    ),
    Tool(
        name="RAG Tool",
        func=rag_tool,
        description="Ищет документы через векторное хранилище (RAG).",
    ),
]


# === Создание агента ===
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,  # Поддержка реактивного агента.
    verbose=True,
)


# === Пример использования агента для тестов ===
def main():
    queries = [
        "Парси заголовки и параграфы с сайта https://example.com.",
        "Найди информацию по архитектуре LangChain через векторное хранилище.",
        "Найди последние новости по теме машинного обучения через Searx.",
    ]

    for query in queries:
        print(f"Тестируем запрос: {query}")
        response = agent_executor.run(query)
        print(f"Ответ агента:\n{response}\n")


if __name__ == "__main__":
    main()
