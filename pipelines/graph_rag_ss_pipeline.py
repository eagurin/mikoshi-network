import os
import json
import sys
import asyncio
import aiohttp
import pandas as pd
import networkx as nx
import requests
import logging
import pickle
from typing import List, Dict, Optional, Callable, Awaitable, Any, Union, Generator, Iterator
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field, EmailStr
from langchain.llms.base import LLM
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
import uvicorn

# Так как OpenWebUI Pipelines предполагает использование асинхронных операций,
# убедимся, что все необходимые компоненты поддерживают асинхронность.

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Настройка SSL для Python версий < 3.11
if sys.version_info < (3, 11):
    if getattr(asyncio, "sslproto", None):
        setattr(asyncio.sslproto, "_is_sslproto_available", lambda: False)

# -----------------------------
# Pydantic Models для API
# -----------------------------

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str

class DocumentOverview(BaseModel):
    id: str
    title: str
    user_id: str
    type: str
    created_at: str
    updated_at: str
    ingestion_status: str
    restructuring_status: str
    version: str
    collection_ids: List[str]
    metadata: dict

class DocumentChunk(BaseModel):
    text: str
    title: str
    user_id: str
    version: str
    chunk_order: int
    document_id: str
    extraction_id: str
    fragment_id: str

class SearchSettings(BaseModel):
    query: str
    search_limit: int
    filters: Optional[dict] = {}

class RAGGenerationConfig(BaseModel):
    model: str
    temperature: float
    top_p: float = 1.0
    max_tokens_to_sample: int = 150
    stream: bool = True
    functions: Optional[Any] = None
    tools: Optional[Any] = None
    add_generation_kwargs: Optional[Any] = None
    api_base: Optional[str] = None

class RAGRequest(BaseModel):
    query: str
    rag_generation_config: RAGGenerationConfig
    vector_search_settings: Optional[dict] = {}
    kg_search_settings: Optional[dict] = {}

class UpdateDocumentRequest(BaseModel):
    document_id: str
    file_path: str
    metadata: Optional[dict] = {}

# -----------------------------
# Пользовательская модель языка (R2RLlm)
# -----------------------------

class R2RLlm(LLM):
    def __init__(self, api_endpoint: str, api_key: Optional[str] = None):
        self.api_endpoint = api_endpoint
        self.api_key = api_key or os.getenv('R2R_API_KEY')

    @property
    def _llm_type(self):
        return "R2R_RAG_Agent"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "messages": [
                {"role": "system", "content": "Вы эксперт по данным в графе знаний. Отвечайте подробно и точно."},
                {"role": "user", "content": prompt}
            ],
            "vector_search_settings": {"search_limit": 5, "filters": {}},
            "kg_search_settings": {"use_kg_search": False},
            "rag_generation_config": {"max_tokens": 300}
        }
        try:
            response = requests.post(self.api_endpoint, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except requests.RequestException as e:
            logger.error(f"Ошибка при обращении к R2R RAG Agent API: {e}")
            return "Извините, произошла ошибка при обработке вашего запроса."

# -----------------------------
# Модель конфигурации (Pipe)
# -----------------------------

class Pipe(BaseModel):
    AGENT_ID: str = Field(default="agent-app")
    AGENT_NAME: str = Field(default="Agent App")
    AGENT_API_URL: str = Field(default="https://api.mikoshi.company/v2/agent")
    AGENT_API_KEY: Optional[str] = Field(default=None)
    ENABLE_EMITTERS: bool = Field(default=True)
    INCLUDE_TITLE_IF_AVAILABLE: bool = Field(default=True)
    VECTOR_SEARCH_SETTINGS: Dict[str, Any] = Field(default_factory=lambda: {
        "use_vector_search": True,
        "use_hybrid_search": True,
        "filters": {},
        "search_limit": 15,
        "selected_group_ids": [],
        "index_measure": "cosine_distance",
        "include_values": True,
        "include_metadatas": True,
        "probes": 10,
        "ef_search": 40,
        "hybrid_search_settings": {
            "full_text_weight": 1,
            "semantic_weight": 5,
            "full_text_limit": 200,
            "rrf_k": 50,
        },
        "search_strategy": "fusion",
    })
    KG_SEARCH_SETTINGS: Dict[str, Any] = Field(default_factory=lambda: {
        "use_kg_search": False,
        "kg_search_type": "global",
        "kg_search_level": None,
        "kg_search_generation_config": None,
        "max_community_description_length": 65536,
        "max_llm_queries_for_global_search": 250,
        "local_search_limits": {
            "__Entity__": 20,
            "__Relationship__": 20,
            "__Community__": 20,
        },
    })
    RAG_GENERATION_CONFIG: Dict[str, Any] = Field(default_factory=lambda: {
        "model": "gpt-4o",
        "temperature": 0.7,
        "top_p": 1.0,
        "max_tokens_to_sample": 150,
        "stream": True,
        "functions": None,
        "tools": None,
        "add_generation_kwargs": None,
        "api_base": None,
    })
    TASK_PROMPT_OVERRIDE: Optional[str] = Field(default=None)

# -----------------------------
# Knowledge Graph Pipeline для OpenWebUI
# -----------------------------

class KnowledgeGraphPipelineOpenWebUI:
    def __init__(self):
        self.graph: Optional[nx.DiGraph] = None
        self.vectorstore: Optional[FAISS] = None
        self.qa_chain: Optional[RetrievalQA] = None
        self.agent: Optional[Any] = None
        self.memory: Optional[ConversationBufferMemory] = None
        self.valves = Pipe()

    async def on_startup(self):
        directory = os.getenv('PARQUET_FILES_PATH', '/app/backend/data')
        df = self.read_parquet_files(directory)
        if df.empty:
            logger.warning("В указанной директории нет данных.")
            return
        df = self.clean_dataframe(df)
        if df.empty:
            logger.warning("После очистки данных не осталось.")
            return
        self.graph = self.create_knowledge_graph(df)
        if self.graph.number_of_nodes() > 0:
            logger.info(f"Граф создан с {self.graph.number_of_nodes()} узлами и {self.graph.number_of_edges()} рёбрами.")
            documents = self.graph_to_documents(self.graph)
            embeddings = OpenAIEmbeddings()
            self.vectorstore = FAISS.from_texts(documents, embeddings)
            logger.info("Векторное хранилище создано.")
            r2r_llm = R2RLlm(api_endpoint=self.valves.AGENT_API_URL, api_key=self.valves.AGENT_API_KEY)
            retriever = self.vectorstore.as_retriever()
            self.qa_chain = RetrievalQA.from_chain_type(llm=r2r_llm, chain_type="stuff", retriever=retriever)
            logger.info("RetrievalQA Chain инициализирована.")
            self.memory = ConversationBufferMemory(memory_key="chat_history")
            tools = [
                Tool(
                    name="Knowledge Base QA",
                    func=self.qa_chain.run,
                    description="Отвечает на вопросы, используя граф знаний"
                ),
                Tool(
                    name="Path Finder",
                    func=lambda input: self.find_path(*input.split(" и ")),
                    description="Находит кратчайший путь между двумя узлами графа знаний. Введите два узла, разделённых ' и '."
                )
            ]
            self.agent = initialize_agent(
                tools,
                r2r_llm,
                AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                memory=self.memory
            )
            logger.info("Агент успешно инициализирован.")
            self.save_graph("knowledge_graph.pkl")
        else:
            logger.warning("Граф пуст. Невозможно инициализировать агента.")

    async def on_shutdown(self):
        logger.info("Остановка Pipeline. Освобождение ресурсов.")

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> str:
        if not self.graph or self.graph.number_of_nodes() == 0:
            return "Граф не инициализирован."
        if not self.agent:
            return "Агент не инициализирован."
        try:
            response = self.agent.run(user_message)
            return response
        except Exception as e:
            logger.error(f"Ошибка при обработке запроса: {e}")
            return "Извините, произошла ошибка при обработке вашего запроса."

    def read_parquet_files(self, directory: str) -> pd.DataFrame:
        dataframes = []
        for filename in os.listdir(directory):
            if filename.endswith('.parquet'):
                file_path = os.path.join(directory, filename)
                try:
                    df = pd.read_parquet(file_path)
                    dataframes.append(df)
                    logger.info(f"Файл {filename} успешно загружен.")
                except Exception as e:
                    logger.error(f"Ошибка при загрузке файла {filename}: {e}")
        combined_df = pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()
        logger.info(f"Загружено {len(combined_df)} записей данных.")
        return combined_df

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_length = len(df)
        df = df.dropna(subset=['source', 'target'])
        df['source'] = df['source'].astype(str)
        df['target'] = df['target'].astype(str)
        cleaned_length = len(df)
        logger.info(f"Данные очищены. Осталось {cleaned_length} строк из {initial_length}.")
        return df

    def create_knowledge_graph(self, df: pd.DataFrame) -> nx.DiGraph:
        G = nx.DiGraph()
        for _, row in df.iterrows():
            source = row['source']
            target = row['target']
            relation = row.get('relation', 'связан с')
            G.add_edge(source, target, relation=relation)
            G.nodes[source]['type'] = row.get('source_type', 'entity')
            G.nodes[target]['type'] = row.get('target_type', 'entity')
        logger.info(f"Граф создан с {G.number_of_nodes()} узлами и {G.number_of_edges()} рёбрами.")
        return G

    def graph_to_documents(self, graph: nx.DiGraph) -> List[str]:
        documents = []
        for node, data in graph.nodes(data=True):
            doc = f"Узел: {node}\nСвойства: {data}\n"
            neighbors = list(graph.neighbors(node))
            if neighbors:
                doc += f"Связанные узлы: {', '.join(neighbors)}\n"
            documents.append(doc)
        logger.info(f"Граф преобразован в {len(documents)} документов.")
        return documents

    def save_graph(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self.graph, f)
        logger.info(f"Граф сохранён в {filepath}")

    def load_graph(self, filepath: str):
        with open(filepath, 'rb') as f:
            self.graph = pickle.load(f)
        logger.info(f"Граф загружен из {filepath}")

    def find_path(self, source: str, target: str) -> str:
        try:
            path = nx.shortest_path(self.graph, source=source, target=target)
            return f"Путь между '{source}' и '{target}': {' -> '.join(path)}"
        except nx.NetworkXNoPath:
            return f"Путь между '{source}' и '{target}' не найден."
        except nx.NodeNotFound as e:
            return f"Узел не найден: {e}"
        except Exception as e:
            logger.error(f"Ошибка при поиске пути: {e}")
            return "Произошла ошибка при поиске пути."

# -----------------------------
# Модель событий для отправки запросов агенту
# -----------------------------

class EventModel(BaseModel):
    type: str
    data: dict

# -----------------------------
# R2R Клиент-обертка для взаимодействия с внешним API
# -----------------------------

class R2RClientWrapper:
    def __init__(self, base_url: str = "http://localhost:7272"):
        self.base_url = base_url
        self.session = aiohttp.ClientSession()

    async def register(self, email: str, password: str) -> dict:
        payload = {"email": email, "password": password}
        async with self.session.post(f"{self.base_url}/register", json=payload) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=resp.status, detail="Registration failed")
            return await resp.json()

    async def verify_email(self, verification_code: str) -> dict:
        payload = {"verification_code": verification_code}
        async with self.session.post(f"{self.base_url}/verify-email", json=payload) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=resp.status, detail="Verification failed")
            return await resp.json()

    async def login(self, email: str, password: str) -> dict:
        payload = {"email": email, "password": password}
        async with self.session.post(f"{self.base_url}/login", json=payload) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=resp.status, detail="Login failed")
            return await resp.json()

    async def logout(self) -> dict:
        async with self.session.post(f"{self.base_url}/logout") as resp:
            if resp.status != 200:
                raise HTTPException(status_code=resp.status, detail="Logout failed")
            return await resp.json()

    async def documents_overview(self) -> List[DocumentOverview]:
        async with self.session.get(f"{self.base_url}/documents-overview") as resp:
            if resp.status != 200:
                raise HTTPException(status_code=resp.status, detail="Failed to fetch documents overview")
            docs = await resp.json()
            return [DocumentOverview(**doc) for doc in docs.get("results", [])]

    async def document_chunks(self, document_id: str) -> List[DocumentChunk]:
        async with self.session.get(f"{self.base_url}/document-chunks/{document_id}") as resp:
            if resp.status != 200:
                raise HTTPException(status_code=resp.status, detail="Failed to fetch document chunks")
            chunks = await resp.json()
            return [DocumentChunk(**chunk) for chunk in chunks.get("results", [])]

    async def ingest_files(self, file_paths: List[str]) -> List[dict]:
        payload = {"file_paths": file_paths}
        async with self.session.post(f"{self.base_url}/ingest", json=payload) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=resp.status, detail="Ingestion failed")
            return await resp.json()

    async def search(self, query: str, search_settings: dict) -> dict:
        payload = {"query": query, **search_settings}
        async with self.session.post(f"{self.base_url}/search", json=payload) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=resp.status, detail="Search failed")
            return await resp.json()

    async def rag(self, query: str, rag_config: dict) -> dict:
        payload = {"query": query, "rag_generation_config": rag_config}
        async with self.session.post(f"{self.base_url}/rag", json=payload) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=resp.status, detail="RAG failed")
            return await resp.json()

    async def logs(self) -> dict:
        async with self.session.get(f"{self.base_url}/logs") as resp:
            if resp.status != 200:
                raise HTTPException(status_code=resp.status, detail="Failed to fetch logs")
            return await resp.json()

    async def analytics(self, filters: dict, analysis_types: dict) -> dict:
        payload = {"filters": filters, "analysis_types": analysis_types}
        async with self.session.post(f"{self.base_url}/analytics", json=payload) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=resp.status, detail="Analytics failed")
            return await resp.json()

    async def delete_document(self, document_id: str) -> dict:
        async with self.session.delete(f"{self.base_url}/documents/{document_id}") as resp:
            if resp.status != 200:
                raise HTTPException(status_code=resp.status, detail="Delete failed")
            return await resp.json()

    async def update_file(self, request: UpdateDocumentRequest) -> dict:
        payload = request.dict()
        async with self.session.put(f"{self.base_url}/documents/update", json=payload) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=resp.status, detail="Update failed")
            return await resp.json()

    async def close(self):
        await self.session.close()

# -----------------------------
# Инициализация R2R Клиента
# -----------------------------

r2r_client = R2RClientWrapper()

# -----------------------------
# Настройка FastAPI приложения
# -----------------------------

app = FastAPI()
pipeline = KnowledgeGraphPipelineOpenWebUI()

# -----------------------------
# API Эндпоинты
# -----------------------------

@app.post("/register", response_model=TokenResponse)
async def register(request: RegisterRequest):
    return await r2r_client.register(request.email, request.password)

@app.post("/verify-email", response_model=TokenResponse)
async def verify_email(verification_code: str):
    return await r2r_client.verify_email(verification_code)

@app.post("/login", response_model=TokenResponse)
async def login(form_data: LoginRequest):
    return await r2r_client.login(form_data.email, form_data.password)

@app.post("/logout")
async def logout():
    return await r2r_client.logout()

@app.get("/documents", response_model=List[DocumentOverview])
async def get_documents():
    return await r2r_client.documents_overview()

@app.get("/documents/{document_id}/chunks", response_model=List[DocumentChunk])
async def get_document_chunks(document_id: str):
    return await r2r_client.document_chunks(document_id)

@app.post("/documents/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    file_paths = []
    for file in files:
        path = f"/tmp/{file.filename}"
        with open(path, "wb") as f:
            f.write(await file.read())
        file_paths.append(path)
    return await r2r_client.ingest_files(file_paths)

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    return await r2r_client.delete_document(document_id)

@app.post("/documents/update")
async def update_document(request: UpdateDocumentRequest):
    return await r2r_client.update_file(request)

@app.post("/search")
async def search_documents(request: SearchSettings):
    return await r2r_client.search(request.query, request.dict())

@app.post("/rag")
async def perform_rag(request: RAGRequest):
    return await r2r_client.rag(request.query, request.rag_generation_config.dict())

@app.get("/logs")
async def get_logs():
    return await r2r_client.logs()

@app.post("/analytics")
async def get_analytics(filters: dict, analysis_types: dict):
    return await r2r_client.analytics(filters, analysis_types)

@app.post("/ask")
async def ask_question(question: str):
    response = pipeline.pipe(question, model_id='', messages=[], body={})
    return {"response": response}

# -----------------------------
# События запуска и остановки
# -----------------------------

@app.on_event("startup")
async def startup_event():
    await pipeline.on_startup()

@app.on_event("shutdown")
async def shutdown_event():
    await pipeline.on_shutdown()
    await r2r_client.close()

# -----------------------------
# Точка входа
# -----------------------------

if __name__ == "__main__":
    # Запускаем Pipeline и сервер FastAPI
    asyncio.run(pipeline.on_startup())
    uvicorn.run(app, host="0.0.0.0", port=8000)
