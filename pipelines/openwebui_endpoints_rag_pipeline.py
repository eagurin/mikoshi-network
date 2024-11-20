import os
import httpx
import asyncio
import logging
from typing import Dict, Any, Callable, Union, Optional

# #### Настройки логирования ####
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# #### API Клиент ####
class APIClient:
    """
    Базовый клиент для отправки запросов к R2R или RAG API.
    Позволяет настраивать проверку SSL-сертификатов.
    """
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        verify_ssl: Union[bool, str] = True,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.verify_ssl = verify_ssl  # Может быть True, False или путь к сертификату

    def _get_headers(self) -> dict:
        """Формируем заголовки для API запросов"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _make_post_request(self, endpoint: str, payload: dict) -> dict:
        """Асинхронный POST-запрос к API"""
        url = f"{self.base_url}/{endpoint}"
        async with httpx.AsyncClient(verify=self.verify_ssl) as client:
            try:
                response = await client.post(
                    url, json=payload, headers=self._get_headers()
                )
                response.raise_for_status()
                logger.info(f"Успешный POST запрос к {url}")
                return response.json()
            except httpx.HTTPStatusError as exc:
                logger.error(
                    f"HTTP Error при запросе к {url}: {exc.response.status_code} | {exc.response.text}"
                )
                raise
            except httpx.RequestError as exc:
                logger.error(f"Request Error при запросе к {url}: {exc}")
                raise
            except Exception as exc:
                logger.error(f"Unexpected Error при запросе к {url}: {exc}")
                raise

    async def _make_get_request(self, endpoint: str, params: dict = {}) -> dict:
        """Асинхронный GET-запрос к API"""
        url = f"{self.base_url}/{endpoint}"
        async with httpx.AsyncClient(verify=self.verify_ssl) as client:
            try:
                response = await client.get(
                    url, params=params, headers=self._get_headers()
                )
                response.raise_for_status()
                logger.info(f"Успешный GET запрос к {url}")
                return response.json()
            except httpx.HTTPStatusError as exc:
                logger.error(
                    f"HTTP Error при запросе к {url}: {exc.response.status_code} | {exc.response.text}"
                )
                raise
            except httpx.RequestError as exc:
                logger.error(f"Request Error при запросе к {url}: {exc}")
                raise
            except Exception as exc:
                logger.error(f"Unexpected Error при запросе к {url}: {exc}")
                raise

    async def _make_delete_request(self, endpoint: str, payload: dict) -> dict:
        """Асинхронный DELETE-запрос к API"""
        url = f"{self.base_url}/{endpoint}"
        async with httpx.AsyncClient(verify=self.verify_ssl) as client:
            try:
                # Используем 'json' вместо 'data' для DELETE запросов
                response = await client.delete(
                    url, json=payload, headers=self._get_headers()
                )
                response.raise_for_status()
                logger.info(f"Успешный DELETE запрос к {url}")
                return response.json()
            except httpx.HTTPStatusError as exc:
                logger.error(
                    f"HTTP Error при запросе к {url}: {exc.response.status_code} | {exc.response.text}"
                )
                raise
            except httpx.RequestError as exc:
                logger.error(f"Request Error при запросе к {url}: {exc}")
                raise
            except Exception as exc:
                logger.error(f"Unexpected Error при запросе к {url}: {exc}")
                raise

# #### R2R Pipeline ####
class R2RPipeline:
    """
    Pipeline для взаимодействия с R2R API
    """
    def __init__(self):
        self.r2r_base_url = os.getenv(
            "R2R_API_URL", "https://api.durka.dndstudio.ru/v2"
        )
        self.api_key = os.getenv("R2R_API_KEY")
        self.verify_ssl = (
            os.getenv("R2R_VERIFY_SSL", "false").lower() == "true"
        )
        self.ssl_cert_path = os.getenv("R2R_SSL_CERT_PATH", None)
        # Если указан путь к сертификату, используем его
        if self.ssl_cert_path:
            self.verify_ssl = self.ssl_cert_path
        self.client = APIClient(
            self.r2r_base_url, self.api_key, self.verify_ssl
        )

    async def retrieve_files(self, collection_id: str, query: str) -> dict:
        """Поиск файлов в коллекции через R2R"""
        payload = {"query": query, "collection_id": collection_id}
        return await self.client._make_post_request(
            "collections/retrieve", payload  # Убедитесь, что этот эндпоинт существует
        )

    async def create_collection(self, collection_name: str) -> dict:
        """Создание коллекции в R2R"""
        payload = {"name": collection_name}
        return await self.client._make_post_request(
            "collections/create", payload
        )

    async def add_files_to_collection(
        self, collection_id: str, file_paths: list
    ) -> dict:
        """Добавление файлов в коллекцию через R2R"""
        payload = {"collection_id": collection_id, "files": file_paths}
        return await self.client._make_post_request(
            "collections/add_files", payload
        )

    async def delete_files(self, collection_id: str, file_ids: list) -> dict:
        """Удаление файлов в R2R"""
        payload = {"collection_id": collection_id, "file_ids": file_ids}
        return await self.client._make_delete_request(
            "collections/delete", payload  # Убедитесь, что этот эндпоинт существует
        )

    async def list_collections(self) -> dict:
        """Получение списка коллекций в R2R"""
        return await self.client._make_get_request("collections/list")

    async def delete_collection(self, collection_id: str) -> dict:
        """Удаление коллекции через R2R"""
        payload = {"collection_id": collection_id}
        return await self.client._make_delete_request(
            "collections/delete", payload
        )

    async def update_collection(self, collection_id: str, new_name: str) -> dict:
        """Обновление названия коллекции через R2R"""
        payload = {"collection_id": collection_id, "new_name": new_name}
        return await self.client._make_post_request(
            "collections/update", payload
        )

    async def get_collection_details(self, collection_id: str) -> dict:
        """Получение подробной информации о коллекции через R2R"""
        params = {"collection_id": collection_id}
        return await self.client._make_get_request(
            "collections/details", params
        )

    # Добавьте дополнительные методы R2R API здесь по мере необходимости

# #### Middleware для подмены API ####

async def custom_middleware(
    request: Dict[str, Any], next_handler: Callable
) -> Union[Dict[str, Any], str]:
    """
    Middleware для перехвата запросов и их обработки через R2R или RAG API в зависимости от переменной окружения.
    """
    use_r2r = (
        os.getenv("USE_R2R", "true").lower() == "true"
    )  # Определяем используемый API: R2R или RAG
    # Определяем, какой pipeline использовать
    api_pipeline = R2RPipeline() if use_r2r else RAGPipeline()

    # Перехват запроса
    if "action" in request:
        action = request["action"]
        if action == "retrieve":
            collection_id = request.get("collection_id")
            query = request.get("query", "")
            logger.info(
                f"Запрос 'retrieve' через {'R2R' if use_r2r else 'RAG'}"
            )
            try:
                return await api_pipeline.retrieve_files(collection_id, query)
            except Exception as e:
                logger.error(f"Ошибка при выполнении 'retrieve': {e}")
                return {"error": str(e)}
        elif action == "create_collection":
            collection_name = request.get("collection_name")
            logger.info(
                f"Запрос 'create_collection' через {'R2R' if use_r2r else 'RAG'}"
            )
            try:
                return await api_pipeline.create_collection(collection_name)
            except Exception as e:
                logger.error(f"Ошибка при выполнении 'create_collection': {e}")
                return {"error": str(e)}
        elif action == "add_files":
            collection_id = request.get("collection_id")
            files = request.get("files", [])
            logger.info(
                f"Запрос 'add_files' через {'R2R' if use_r2r else 'RAG'}"
            )
            try:
                return await api_pipeline.add_files_to_collection(
                    collection_id, files
                )
            except Exception as e:
                logger.error(f"Ошибка при выполнении 'add_files': {e}")
                return {"error": str(e)}
        elif action == "delete_files":
            collection_id = request.get("collection_id")
            file_ids = request.get("file_ids", [])
            logger.info(
                f"Запрос 'delete_files' через {'R2R' if use_r2r else 'RAG'}"
            )
            try:
                return await api_pipeline.delete_files(collection_id, file_ids)
            except Exception as e:
                logger.error(f"Ошибка при выполнении 'delete_files': {e}")
                return {"error": str(e)}
        elif action == "list_collections":
            logger.info(
                f"Запрос 'list_collections' через {'R2R' if use_r2r else 'RAG'}"
            )
            try:
                return await api_pipeline.list_collections()
            except Exception as e:
                logger.error(f"Ошибка при выполнении 'list_collections': {e}")
                return {"error": str(e)}
        elif action == "delete_collection":
            collection_id = request.get("collection_id")
            logger.info(
                f"Запрос 'delete_collection' через {'R2R' if use_r2r else 'RAG'}"
            )
            try:
                return await api_pipeline.delete_collection(collection_id)
            except Exception as e:
                logger.error(f"Ошибка при выполнении 'delete_collection': {e}")
                return {"error": str(e)}
        elif action == "update_collection":
            collection_id = request.get("collection_id")
            new_name = request.get("new_name")
            logger.info(
                f"Запрос 'update_collection' через {'R2R' if use_r2r else 'RAG'}"
            )
            try:
                return await api_pipeline.update_collection(
                    collection_id, new_name
                )
            except Exception as e:
                logger.error(f"Ошибка при выполнении 'update_collection': {e}")
                return {"error": str(e)}
        elif action == "get_collection_details":
            collection_id = request.get("collection_id")
            logger.info(
                f"Запрос 'get_collection_details' через {'R2R' if use_r2r else 'RAG'}"
            )
            try:
                return await api_pipeline.get_collection_details(collection_id)
            except Exception as e:
                logger.error(
                    f"Ошибка при выполнении 'get_collection_details': {e}"
                )
                return {"error": str(e)}
        # Добавляйте дополнительные действия здесь по мере необходимости

    logger.warning(
        "Неизвестное действие, передача запроса следующему обработчику."
    )
    return await next_handler(request)

# #### Эмуляционные запросы для тестирования ####

async def simulate_client_request(request: Dict[str, Any]):
    """
    Эмуляция клиентских запросов для тестирования pipeline.
    """
    async def dummy_next_handler(req: Dict[str, Any]):
        logger.info(
            "Запрос передан следующему обработчику. Ответ возвращен напрямую."
        )
        return {"status": "handled directly", "request": req}
    try:
        response = await custom_middleware(request, dummy_next_handler)
    except Exception as e:
        logger.error(f"Ошибка при эмуляции запроса: {e}")
        response = {"error": str(e)}
    logger.info(f"Ответ: {response}")
    return response

# #### Основной блок для тестирования ####
if __name__ == "__main__":
    # Тестовые примеры работы с API через консоль
    test_requests = [
        {"action": "create_collection", "collection_name": "Test Collection"},
        {
            "action": "retrieve",
            "collection_id": "12345",
            "query": "test query",
        },
        {
            "action": "add_files",
            "collection_id": "12345",
            "files": ["file1.pdf", "file2.pdf"],
        },
        {
            "action": "delete_files",
            "collection_id": "12345",
            "file_ids": ["file1", "file2"],
        },
        {"action": "list_collections"},
        {"action": "delete_collection", "collection_id": "12345"},
        {
            "action": "update_collection",
            "collection_id": "12345",
            "new_name": "Updated Collection",
        },
        {"action": "get_collection_details", "collection_id": "12345"},
        # Пример запроса с неизвестным действием
        {"action": "unknown_action", "some_key": "some_value"},
    ]

    async def run_tests():
        for req in test_requests:
            await simulate_client_request(req)

    asyncio.run(run_tests())