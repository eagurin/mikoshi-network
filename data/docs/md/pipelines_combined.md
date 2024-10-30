```python
# config.py
import os

####################################
# Load .env file
####################################

try:
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv("./.env"))
except ImportError:
    print("dotenv not installed, skipping...")

API_KEY = os.getenv("PIPELINES_API_KEY", "0p3n-w3bu!")
PIPELINES_DIR = os.getenv("PIPELINES_DIR", "./pipelines")

```

---

```python
# main.py
from fastapi import FastAPI, Request, Depends, status, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool


from starlette.responses import StreamingResponse, Response
from pydantic import BaseModel, ConfigDict
from typing import List, Union, Generator, Iterator


from utils.pipelines.auth import bearer_security, get_current_user
from utils.pipelines.main import get_last_user_message, stream_message_template
from utils.pipelines.misc import convert_to_raw_url

from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from schemas import FilterForm, OpenAIChatCompletionForm
from urllib.parse import urlparse

import shutil
import aiohttp
import os
import importlib.util
import logging
import time
import json
import uuid
import sys
import subprocess


from config import API_KEY, PIPELINES_DIR

if not os.path.exists(PIPELINES_DIR):
    os.makedirs(PIPELINES_DIR)


PIPELINES = {}
PIPELINE_MODULES = {}
PIPELINE_NAMES = {}


def get_all_pipelines():
    pipelines = {}
    for pipeline_id in PIPELINE_MODULES.keys():
        pipeline = PIPELINE_MODULES[pipeline_id]

        if hasattr(pipeline, "type"):
            if pipeline.type == "manifold":
                manifold_pipelines = []

                # Check if pipelines is a function or a list
                if callable(pipeline.pipelines):
                    manifold_pipelines = pipeline.pipelines()
                else:
                    manifold_pipelines = pipeline.pipelines

                for p in manifold_pipelines:
                    manifold_pipeline_id = f'{pipeline_id}.{p["id"]}'

                    manifold_pipeline_name = p["name"]
                    if hasattr(pipeline, "name"):
                        manifold_pipeline_name = (
                            f"{pipeline.name}{manifold_pipeline_name}"
                        )

                    pipelines[manifold_pipeline_id] = {
                        "module": pipeline_id,
                        "type": pipeline.type if hasattr(pipeline, "type") else "pipe",
                        "id": manifold_pipeline_id,
                        "name": manifold_pipeline_name,
                        "valves": (
                            pipeline.valves if hasattr(pipeline, "valves") else None
                        ),
                    }
            if pipeline.type == "filter":
                pipelines[pipeline_id] = {
                    "module": pipeline_id,
                    "type": (pipeline.type if hasattr(pipeline, "type") else "pipe"),
                    "id": pipeline_id,
                    "name": (
                        pipeline.name if hasattr(pipeline, "name") else pipeline_id
                    ),
                    "pipelines": (
                        pipeline.valves.pipelines
                        if hasattr(pipeline, "valves")
                        and hasattr(pipeline.valves, "pipelines")
                        else []
                    ),
                    "priority": (
                        pipeline.valves.priority
                        if hasattr(pipeline, "valves")
                        and hasattr(pipeline.valves, "priority")
                        else 0
                    ),
                    "valves": pipeline.valves if hasattr(pipeline, "valves") else None,
                }
        else:
            pipelines[pipeline_id] = {
                "module": pipeline_id,
                "type": (pipeline.type if hasattr(pipeline, "type") else "pipe"),
                "id": pipeline_id,
                "name": (pipeline.name if hasattr(pipeline, "name") else pipeline_id),
                "valves": pipeline.valves if hasattr(pipeline, "valves") else None,
            }

    return pipelines

def parse_frontmatter(content):
    frontmatter = {}
    for line in content.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            frontmatter[key.strip().lower()] = value.strip()
    return frontmatter

def install_frontmatter_requirements(requirements):
    if requirements:
        req_list = [req.strip() for req in requirements.split(',')]
        for req in req_list:
            print(f"Installing requirement: {req}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
    else:
        print("No requirements found in frontmatter.")

async def load_module_from_path(module_name, module_path):

    try:
        # Read the module content
        with open(module_path, 'r') as file:
            content = file.read()

        # Parse frontmatter
        frontmatter = {}
        if content.startswith('"""'):
            end = content.find('"""', 3)
            if end != -1:
                frontmatter_content = content[3:end]
                frontmatter = parse_frontmatter(frontmatter_content)

        # Install requirements if specified
        if 'requirements' in frontmatter:
            install_frontmatter_requirements(frontmatter['requirements'])

        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"Loaded module: {module.__name__}")
        if hasattr(module, "Pipeline"):
            return module.Pipeline()
        else:
            raise Exception("No Pipeline class found")
    except Exception as e:
        print(f"Error loading module: {module_name}")

        # Move the file to the error folder
        failed_pipelines_folder = os.path.join(PIPELINES_DIR, "failed")
        if not os.path.exists(failed_pipelines_folder):
            os.makedirs(failed_pipelines_folder)

        failed_file_path = os.path.join(failed_pipelines_folder, f"{module_name}.py")
        os.rename(module_path, failed_file_path)
        print(e)
    return None


async def load_modules_from_directory(directory):
    global PIPELINE_MODULES
    global PIPELINE_NAMES

    for filename in os.listdir(directory):
        if filename.endswith(".py"):
            module_name = filename[:-3]  # Remove the .py extension
            module_path = os.path.join(directory, filename)

            # Create subfolder matching the filename without the .py extension
            subfolder_path = os.path.join(directory, module_name)
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)
                logging.info(f"Created subfolder: {subfolder_path}")

            # Create a valves.json file if it doesn't exist
            valves_json_path = os.path.join(subfolder_path, "valves.json")
            if not os.path.exists(valves_json_path):
                with open(valves_json_path, "w") as f:
                    json.dump({}, f)
                logging.info(f"Created valves.json in: {subfolder_path}")

            pipeline = await load_module_from_path(module_name, module_path)
            if pipeline:
                # Overwrite pipeline.valves with values from valves.json
                if os.path.exists(valves_json_path):
                    with open(valves_json_path, "r") as f:
                        valves_json = json.load(f)
                        if hasattr(pipeline, "valves"):
                            ValvesModel = pipeline.valves.__class__
                            # Create a ValvesModel instance using default values and overwrite with valves_json
                            combined_valves = {
                                **pipeline.valves.model_dump(),
                                **valves_json,
                            }
                            valves = ValvesModel(**combined_valves)
                            pipeline.valves = valves

                            logging.info(f"Updated valves for module: {module_name}")

                pipeline_id = pipeline.id if hasattr(pipeline, "id") else module_name
                PIPELINE_MODULES[pipeline_id] = pipeline
                PIPELINE_NAMES[pipeline_id] = module_name
                logging.info(f"Loaded module: {module_name}")
            else:
                logging.warning(f"No Pipeline class found in {module_name}")

    global PIPELINES
    PIPELINES = get_all_pipelines()


async def on_startup():
    await load_modules_from_directory(PIPELINES_DIR)

    for module in PIPELINE_MODULES.values():
        if hasattr(module, "on_startup"):
            await module.on_startup()


async def on_shutdown():
    for module in PIPELINE_MODULES.values():
        if hasattr(module, "on_shutdown"):
            await module.on_shutdown()


async def reload():
    await on_shutdown()
    # Clear existing pipelines
    PIPELINES.clear()
    PIPELINE_MODULES.clear()
    PIPELINE_NAMES.clear()
    # Load pipelines afresh
    await on_startup()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await on_startup()
    yield
    await on_shutdown()


app = FastAPI(docs_url="/docs", redoc_url=None, lifespan=lifespan)

app.state.PIPELINES = PIPELINES


origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def check_url(request: Request, call_next):
    start_time = int(time.time())
    app.state.PIPELINES = get_all_pipelines()
    response = await call_next(request)
    process_time = int(time.time()) - start_time
    response.headers["X-Process-Time"] = str(process_time)

    return response


@app.get("/v1/models")
@app.get("/models")
async def get_models():
    """
    Returns the available pipelines
    """
    app.state.PIPELINES = get_all_pipelines()
    return {
        "data": [
            {
                "id": pipeline["id"],
                "name": pipeline["name"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "openai",
                "pipeline": {
                    "type": pipeline["type"],
                    **(
                        {
                            "pipelines": (
                                pipeline["valves"].pipelines
                                if pipeline.get("valves", None)
                                else []
                            ),
                            "priority": pipeline.get("priority", 0),
                        }
                        if pipeline.get("type", "pipe") == "filter"
                        else {}
                    ),
                    "valves": pipeline["valves"] != None,
                },
            }
            for pipeline in app.state.PIPELINES.values()
        ],
        "object": "list",
        "pipelines": True,
    }


@app.get("/v1")
@app.get("/")
async def get_status():
    return {"status": True}


@app.get("/v1/pipelines")
@app.get("/pipelines")
async def list_pipelines(user: str = Depends(get_current_user)):
    if user == API_KEY:
        return {
            "data": [
                {
                    "id": pipeline_id,
                    "name": PIPELINE_NAMES[pipeline_id],
                    "type": (
                        PIPELINE_MODULES[pipeline_id].type
                        if hasattr(PIPELINE_MODULES[pipeline_id], "type")
                        else "pipe"
                    ),
                    "valves": (
                        True
                        if hasattr(PIPELINE_MODULES[pipeline_id], "valves")
                        else False
                    ),
                }
                for pipeline_id in list(PIPELINE_MODULES.keys())
            ]
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


class AddPipelineForm(BaseModel):
    url: str


async def download_file(url: str, dest_folder: str):
    filename = os.path.basename(urlparse(url).path)
    if not filename.endswith(".py"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL must point to a Python file",
        )

    file_path = os.path.join(dest_folder, filename)

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to download file",
                )
            with open(file_path, "wb") as f:
                f.write(await response.read())

    return file_path


@app.post("/v1/pipelines/add")
@app.post("/pipelines/add")
async def add_pipeline(
    form_data: AddPipelineForm, user: str = Depends(get_current_user)
):
    if user != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    try:
        url = convert_to_raw_url(form_data.url)

        print(url)
        file_path = await download_file(url, dest_folder=PIPELINES_DIR)
        await reload()
        return {
            "status": True,
            "detail": f"Pipeline added successfully from {file_path}",
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@app.post("/v1/pipelines/upload")
@app.post("/pipelines/upload")
async def upload_pipeline(
    file: UploadFile = File(...), user: str = Depends(get_current_user)
):
    if user != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    file_ext = os.path.splitext(file.filename)[1]
    if file_ext != ".py":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only Python files are allowed.",
        )

    try:
        # Ensure the destination folder exists
        os.makedirs(PIPELINES_DIR, exist_ok=True)

        # Define the file path
        file_path = os.path.join(PIPELINES_DIR, file.filename)

        # Save the uploaded file to the specified directory
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Perform any necessary reload or processing
        await reload()

        return {
            "status": True,
            "detail": f"Pipeline uploaded successfully to {file_path}",
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


class DeletePipelineForm(BaseModel):
    id: str


@app.delete("/v1/pipelines/delete")
@app.delete("/pipelines/delete")
async def delete_pipeline(
    form_data: DeletePipelineForm, user: str = Depends(get_current_user)
):
    if user != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )

    pipeline_id = form_data.id
    pipeline_name = PIPELINE_NAMES.get(pipeline_id.split(".")[0], None)

    if PIPELINE_MODULES[pipeline_id]:
        if hasattr(PIPELINE_MODULES[pipeline_id], "on_shutdown"):
            await PIPELINE_MODULES[pipeline_id].on_shutdown()

    pipeline_path = os.path.join(PIPELINES_DIR, f"{pipeline_name}.py")
    if os.path.exists(pipeline_path):
        os.remove(pipeline_path)
        await reload()
        return {
            "status": True,
            "detail": f"Pipeline {pipeline_id} deleted successfully",
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline {pipeline_id} not found",
        )


@app.post("/v1/pipelines/reload")
@app.post("/pipelines/reload")
async def reload_pipelines(user: str = Depends(get_current_user)):
    if user == API_KEY:
        await reload()
        return {"message": "Pipelines reloaded successfully."}
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


@app.get("/v1/{pipeline_id}/valves")
@app.get("/{pipeline_id}/valves")
async def get_valves(pipeline_id: str):
    if pipeline_id not in PIPELINE_MODULES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline {pipeline_id} not found",
        )

    pipeline = PIPELINE_MODULES[pipeline_id]

    if hasattr(pipeline, "valves") is False:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Valves for {pipeline_id} not found",
        )

    return pipeline.valves


@app.get("/v1/{pipeline_id}/valves/spec")
@app.get("/{pipeline_id}/valves/spec")
async def get_valves_spec(pipeline_id: str):
    if pipeline_id not in PIPELINE_MODULES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline {pipeline_id} not found",
        )

    pipeline = PIPELINE_MODULES[pipeline_id]

    if hasattr(pipeline, "valves") is False:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Valves for {pipeline_id} not found",
        )

    return pipeline.valves.schema()


@app.post("/v1/{pipeline_id}/valves/update")
@app.post("/{pipeline_id}/valves/update")
async def update_valves(pipeline_id: str, form_data: dict):

    if pipeline_id not in PIPELINE_MODULES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline {pipeline_id} not found",
        )

    pipeline = PIPELINE_MODULES[pipeline_id]

    if hasattr(pipeline, "valves") is False:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Valves for {pipeline_id} not found",
        )

    try:
        ValvesModel = pipeline.valves.__class__
        valves = ValvesModel(**form_data)
        pipeline.valves = valves

        # Determine the directory path for the valves.json file
        subfolder_path = os.path.join(PIPELINES_DIR, PIPELINE_NAMES[pipeline_id])
        valves_json_path = os.path.join(subfolder_path, "valves.json")

        # Save the updated valves data back to the valves.json file
        with open(valves_json_path, "w") as f:
            json.dump(valves.model_dump(), f)

        if hasattr(pipeline, "on_valves_updated"):
            await pipeline.on_valves_updated()
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{str(e)}",
        )

    return pipeline.valves


@app.post("/v1/{pipeline_id}/filter/inlet")
@app.post("/{pipeline_id}/filter/inlet")
async def filter_inlet(pipeline_id: str, form_data: FilterForm):
    if pipeline_id not in app.state.PIPELINES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Filter {pipeline_id} not found",
        )

    try:
        pipeline = app.state.PIPELINES[form_data.body["model"]]
        if pipeline["type"] == "manifold":
            pipeline_id = pipeline_id.split(".")[0]
    except:
        pass

    pipeline = PIPELINE_MODULES[pipeline_id]

    try:
        if hasattr(pipeline, "inlet"):
            body = await pipeline.inlet(form_data.body, form_data.user)
            return body
        else:
            return form_data.body
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{str(e)}",
        )


@app.post("/v1/{pipeline_id}/filter/outlet")
@app.post("/{pipeline_id}/filter/outlet")
async def filter_outlet(pipeline_id: str, form_data: FilterForm):
    if pipeline_id not in app.state.PIPELINES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Filter {pipeline_id} not found",
        )

    try:
        pipeline = app.state.PIPELINES[form_data.body["model"]]
        if pipeline["type"] == "manifold":
            pipeline_id = pipeline_id.split(".")[0]
    except:
        pass

    pipeline = PIPELINE_MODULES[pipeline_id]

    try:
        if hasattr(pipeline, "outlet"):
            body = await pipeline.outlet(form_data.body, form_data.user)
            return body
        else:
            return form_data.body
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{str(e)}",
        )


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def generate_openai_chat_completion(form_data: OpenAIChatCompletionForm):
    messages = [message.model_dump() for message in form_data.messages]
    user_message = get_last_user_message(messages)

    if (
        form_data.model not in app.state.PIPELINES
        or app.state.PIPELINES[form_data.model]["type"] == "filter"
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline {form_data.model} not found",
        )

    def job():
        print(form_data.model)

        pipeline = app.state.PIPELINES[form_data.model]
        pipeline_id = form_data.model

        print(pipeline_id)

        if pipeline["type"] == "manifold":
            manifold_id, pipeline_id = pipeline_id.split(".", 1)
            pipe = PIPELINE_MODULES[manifold_id].pipe
        else:
            pipe = PIPELINE_MODULES[pipeline_id].pipe

        if form_data.stream:

            def stream_content():
                res = pipe(
                    user_message=user_message,
                    model_id=pipeline_id,
                    messages=messages,
                    body=form_data.model_dump(),
                )

                logging.info(f"stream:true:{res}")

                if isinstance(res, str):
                    message = stream_message_template(form_data.model, res)
                    logging.info(f"stream_content:str:{message}")
                    yield f"data: {json.dumps(message)}\n\n"

                if isinstance(res, Iterator):
                    for line in res:
                        if isinstance(line, BaseModel):
                            line = line.model_dump_json()
                            line = f"data: {line}"

                        try:
                            line = line.decode("utf-8")
                        except:
                            pass

                        logging.info(f"stream_content:Generator:{line}")

                        if line.startswith("data:"):
                            yield f"{line}\n\n"
                        else:
                            line = stream_message_template(form_data.model, line)
                            yield f"data: {json.dumps(line)}\n\n"

                if isinstance(res, str) or isinstance(res, Generator):
                    finish_message = {
                        "id": f"{form_data.model}-{str(uuid.uuid4())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": form_data.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "logprobs": None,
                                "finish_reason": "stop",
                            }
                        ],
                    }

                    yield f"data: {json.dumps(finish_message)}\n\n"
                    yield f"data: [DONE]"

            return StreamingResponse(stream_content(), media_type="text/event-stream")
        else:
            res = pipe(
                user_message=user_message,
                model_id=pipeline_id,
                messages=messages,
                body=form_data.model_dump(),
            )
            logging.info(f"stream:false:{res}")

            if isinstance(res, dict):
                return res
            elif isinstance(res, BaseModel):
                return res.model_dump()
            else:

                message = ""

                if isinstance(res, str):
                    message = res

                if isinstance(res, Generator):
                    for stream in res:
                        message = f"{message}{stream}"

                logging.info(f"stream:false:{message}")
                return {
                    "id": f"{form_data.model}-{str(uuid.uuid4())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": form_data.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": message,
                            },
                            "logprobs": None,
                            "finish_reason": "stop",
                        }
                    ],
                }

    return await run_in_threadpool(job)

```

---

<p align="center">
  <a href="#"><img src="./header.png" alt="Pipelines Logo"></a>
</p>

# Pipelines: UI-Agnostic OpenAI API Plugin Framework

Welcome to **Pipelines**, an [Open WebUI](https://github.com/open-webui) initiative. Pipelines bring modular, customizable workflows to any UI client supporting OpenAI API specs – and much more! Easily extend functionalities, integrate unique logic, and create dynamic workflows with just a few lines of code.

## 🚀 Why Choose Pipelines?

- **Limitless Possibilities:** Easily add custom logic and integrate Python libraries, from AI agents to home automation APIs.
- **Seamless Integration:** Compatible with any UI/client supporting OpenAI API specs. (Only pipe-type pipelines are supported; filter types require clients with Pipelines support.)
- **Custom Hooks:** Build and integrate custom pipelines.

### Examples of What You Can Achieve:

- [**Function Calling Pipeline**](/examples/filters/function_calling_filter_pipeline.py): Easily handle function calls and enhance your applications with custom logic.
- [**Custom RAG Pipeline**](/examples/pipelines/rag/llamaindex_pipeline.py): Implement sophisticated Retrieval-Augmented Generation pipelines tailored to your needs.
- [**Message Monitoring Using Langfuse**](/examples/filters/langfuse_filter_pipeline.py): Monitor and analyze message interactions in real-time using Langfuse.
- [**Rate Limit Filter**](/examples/filters/rate_limit_filter_pipeline.py): Control the flow of requests to prevent exceeding rate limits.
- [**Real-Time Translation Filter with LibreTranslate**](/examples/filters/libretranslate_filter_pipeline.py): Seamlessly integrate real-time translations into your LLM interactions.
- [**Toxic Message Filter**](/examples/filters/detoxify_filter_pipeline.py): Implement filters to detect and handle toxic messages effectively.
- **And Much More!**: The sky is the limit for what you can accomplish with Pipelines and Python. [Check out our scaffolds](/examples/scaffolds) to get a head start on your projects and see how you can streamline your development process!

## 🔧 How It Works

<p align="center">
  <a href="./docs/images/workflow.png"><img src="./docs/images/workflow.png" alt="Pipelines Workflow"></a>
</p>

Integrating Pipelines with any OpenAI API-compatible UI client is simple. Launch your Pipelines instance and set the OpenAI URL on your client to the Pipelines URL. That's it! You're ready to leverage any Python library for your needs.

## ⚡ Quick Start with Docker

> [!WARNING]
> Pipelines are a plugin system with arbitrary code execution — **don't fetch random pipelines from sources you don't trust**.

For a streamlined setup using Docker:

1. **Run the Pipelines container:**

   ```sh
   docker run -d -p 9099:9099 --add-host=host.docker.internal:host-gateway -v pipelines:/app/pipelines --name pipelines --restart always ghcr.io/open-webui/pipelines:main
   ```

2. **Connect to Open WebUI:**

   - Navigate to the **Settings > Connections > OpenAI API** section in Open WebUI.
   - Set the API URL to `http://localhost:9099` and the API key to `0p3n-w3bu!`. Your pipelines should now be active.

> [!NOTE]
> If your Open WebUI is running in a Docker container, replace `localhost` with `host.docker.internal` in the API URL.

3. **Manage Configurations:**

   - In the admin panel, go to **Admin Settings > Pipelines tab**.
   - Select your desired pipeline and modify the valve values directly from the WebUI.

> [!TIP]
> If you are unable to connect, it is most likely a Docker networking issue. We encourage you to troubleshoot on your own and share your methods and solutions in the discussions forum.

If you need to install a custom pipeline with additional dependencies:

- **Run the following command:**

  ```sh
  docker run -d -p 9099:9099 --add-host=host.docker.internal:host-gateway -e PIPELINES_URLS="https://github.com/open-webui/pipelines/blob/main/examples/filters/detoxify_filter_pipeline.py" -v pipelines:/app/pipelines --name pipelines --restart always ghcr.io/open-webui/pipelines:main
  ```

Alternatively, you can directly install pipelines from the admin settings by copying and pasting the pipeline URL, provided it doesn't have additional dependencies.

That's it! You're now ready to build customizable AI integrations effortlessly with Pipelines. Enjoy!

## 📦 Installation and Setup

Get started with Pipelines in a few easy steps:

1. **Ensure Python 3.11 is installed.**
2. **Clone the Pipelines repository:**

   ```sh
   git clone https://github.com/open-webui/pipelines.git
   cd pipelines
   ```

3. **Install the required dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

4. **Start the Pipelines server:**

   ```sh
   sh ./start.sh
   ```

Once the server is running, set the OpenAI URL on your client to the Pipelines URL. This unlocks the full capabilities of Pipelines, integrating any Python library and creating custom workflows tailored to your needs.

## 📂 Directory Structure and Examples

The `/pipelines` directory is the core of your setup. Add new modules, customize existing ones, and manage your workflows here. All the pipelines in the `/pipelines` directory will be **automatically loaded** when the server launches.

You can change this directory from `/pipelines` to another location using the `PIPELINES_DIR` env variable.

### Integration Examples

Find various integration examples in the `/examples` directory. These examples show how to integrate different functionalities, providing a foundation for building your own custom pipelines.

## 🎉 Work in Progress

We’re continuously evolving! We'd love to hear your feedback and understand which hooks and features would best suit your use case. Feel free to reach out and become a part of our Open WebUI community!

Our vision is to push **Pipelines** to become the ultimate plugin framework for our AI interface, **Open WebUI**. Imagine **Open WebUI** as the WordPress of AI interfaces, with **Pipelines** being its diverse range of plugins. Join us on this exciting journey! 🌍

---

```python
# examples/filters/datadog_filter_pipeline.py
"""
title: DataDog Filter Pipeline
author: 0xThresh
date: 2024-06-06
version: 1.0
license: MIT
description: A filter pipeline that sends traces to DataDog.
requirements: ddtrace
environment_variables: DD_LLMOBS_AGENTLESS_ENABLED, DD_LLMOBS_ENABLED, DD_LLMOBS_APP_NAME, DD_API_KEY, DD_SITE 
"""

from typing import List, Optional
import os

from utils.pipelines.main import get_last_user_message, get_last_assistant_message
from pydantic import BaseModel
from ddtrace.llmobs import LLMObs


class Pipeline:
    class Valves(BaseModel):
        # List target pipeline ids (models) that this filter will be connected to.
        # If you want to connect this filter to all pipelines, you can set pipelines to ["*"]
        # e.g. ["llama3:latest", "gpt-3.5-turbo"]
        pipelines: List[str] = []

        # Assign a priority level to the filter pipeline.
        # The priority level determines the order in which the filter pipelines are executed.
        # The lower the number, the higher the priority.
        priority: int = 0

        # Valves
        dd_api_key: str
        dd_site: str
        ml_app: str

    def __init__(self):
        # Pipeline filters are only compatible with Open WebUI
        # You can think of filter pipeline as a middleware that can be used to edit the form data before it is sent to the OpenAI API.
        self.type = "filter"

        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "datadog_filter_pipeline"
        self.name = "DataDog Filter"

        # Initialize
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],  # Connect to all pipelines
                "dd_api_key": os.getenv("DD_API_KEY"),
                "dd_site": os.getenv("DD_SITE", "datadoghq.com"),
                "ml_app": os.getenv("ML_APP", "pipelines-test"),
            }
        )

        # DataDog LLMOBS docs: https://docs.datadoghq.com/tracing/llm_observability/sdk/
        self.LLMObs = LLMObs()
        self.llm_span = None
        self.chat_generations = {}
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        self.set_dd()
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        self.LLMObs.flush()
        pass

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        self.set_dd()
        pass

    def set_dd(self):
        self.LLMObs.enable(
            ml_app=self.valves.ml_app,
            api_key=self.valves.dd_api_key,
            site=self.valves.dd_site,
            agentless_enabled=True,
            integrations_enabled=True,
        )

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"inlet:{__name__}")

        self.llm_span = self.LLMObs.llm(
            model_name=body["model"],
            name=f"filter:{__name__}",
            model_provider="open-webui",
            session_id=body["chat_id"],
            ml_app=self.valves.ml_app
        )

        self.LLMObs.annotate(
            span = self.llm_span,
            input_data = get_last_user_message(body["messages"]),
        )

        return body


    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"outlet:{__name__}")

        self.LLMObs.annotate(
            span = self.llm_span,
            output_data = get_last_assistant_message(body["messages"]),
        )

        self.llm_span.finish()
        self.LLMObs.flush()

        return body

```

---

```python
# schemas.py
from typing import List, Union, Optional
from pydantic import BaseModel, RootModel, ConfigDict

class ImageContent(BaseModel):
    type: str
    image_url: dict

class TextContent(BaseModel):
    type: str
    text: str

class MessageContent(RootModel):
    root: Union[TextContent, ImageContent]

class OpenAIChatMessage(BaseModel):
    role: str
    content: Union[str, List[MessageContent]]

    model_config = ConfigDict(extra="allow")

class OpenAIChatCompletionForm(BaseModel):
    stream: bool = True
    model: str
    messages: List[OpenAIChatMessage]

    model_config = ConfigDict(extra="allow")

class FilterForm(BaseModel):
    body: dict
    user: Optional[dict] = None
    model_config = ConfigDict(extra="allow")
```

---

# Contributor Covenant Code of Conduct

## Our Pledge

We as members, contributors, and leaders pledge to make participation in our
community a harassment-free experience for everyone, regardless of age, body
size, visible or invisible disability, ethnicity, sex characteristics, gender
identity and expression, level of experience, education, socio-economic status,
nationality, personal appearance, race, religion, or sexual identity
and orientation.

We pledge to act and interact in ways that contribute to an open, welcoming, diverse, inclusive, and healthy community.

## Our Standards

Examples of behavior that contribute to a positive environment for our community include:

- Demonstrating empathy and kindness toward other people
- Being respectful of differing opinions, viewpoints, and experiences
- Giving and gracefully accepting constructive feedback
- Accepting responsibility and apologizing to those affected by our mistakes, and learning from the experience
- Focusing on what is best not just for us as individuals, but for the overall community

Examples of unacceptable behavior include:

- The use of sexualized language or imagery, and sexual attention or advances of any kind
- Trolling, insulting or derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information, such as a physical or email address, without their explicit permission
- **Spamming of any kind**
- Other conduct which could reasonably be considered inappropriate in a professional setting

## Enforcement Responsibilities

Community leaders are responsible for clarifying and enforcing our standards of acceptable behavior and will take appropriate and fair corrective action in response to any behavior that they deem inappropriate, threatening, offensive, or harmful.

## Scope

This Code of Conduct applies within all community spaces and also applies when an individual is officially representing the community in public spaces. Examples of representing our community include using an official e-mail address, posting via an official social media account, or acting as an appointed representative at an online or offline event.

## Enforcement

Instances of abusive, harassing, spamming, or otherwise unacceptable behavior may be reported to the community leaders responsible for enforcement at hello@openwebui.com. All complaints will be reviewed and investigated promptly and fairly.

All community leaders are obligated to respect the privacy and security of the reporter of any incident.

## Enforcement Guidelines

Community leaders will follow these Community Impact Guidelines in determining the consequences for any action they deem in violation of this Code of Conduct:

### 1. Temporary Ban

**Community Impact**: Any violation of community standards, including but not limited to inappropriate language, unprofessional behavior, harassment, or spamming.

**Consequence**: A temporary ban from any sort of interaction or public communication with the community for a specified period of time. No public or private interaction with the people involved, including unsolicited interaction with those enforcing the Code of Conduct, is allowed during this period. Violating these terms may lead to a permanent ban.

### 2. Permanent Ban

**Community Impact**: Repeated or severe violations of community standards, including sustained inappropriate behavior, harassment of an individual, or aggression toward or disparagement of classes of individuals.

**Consequence**: A permanent ban from any sort of public interaction within the community.
## Attribution

This Code of Conduct is adapted from the [Contributor Covenant][homepage],
version 2.0, available at
https://www.contributor-covenant.org/version/2/0/code_of_conduct.html.

Community Impact Guidelines were inspired by [Mozilla's code of conduct
enforcement ladder](https://github.com/mozilla/diversity).

[homepage]: https://www.contributor-covenant.org

For answers to common questions about this code of conduct, see the FAQ at
https://www.contributor-covenant.org/faq. Translations are available at
https://www.contributor-covenant.org/translations.

---

```python
# examples/filters/mem0_memory_filter_pipeline.py
"""
title: Long Term Memory Filter
author: Anton Nilsson
date: 2024-08-23
version: 1.0
license: MIT
description: A filter that processes user messages and stores them as long term memory by utilizing the mem0 framework together with qdrant and ollama
requirements: pydantic, ollama, mem0ai
"""

from typing import List, Optional
from pydantic import BaseModel
import json
from mem0 import Memory
import threading

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0

        store_cycles: int = 5 # Number of messages from the user before the data is processed and added to the memory
        mem_zero_user: str = "user" # Memories belongs to this user, only used by mem0 for internal organization of memories

        # Default values for the mem0 vector store
        vector_store_qdrant_name: str = "memories"
        vector_store_qdrant_url: str = "host.docker.internal"
        vector_store_qdrant_port: int = 6333
        vector_store_qdrant_dims: int = 768 # Need to match the vector dimensions of the embedder model

        # Default values for the mem0 language model
        ollama_llm_model: str = "llama3.1:latest" # This model need to exist in ollama
        ollama_llm_temperature: float = 0
        ollama_llm_tokens: int = 8000
        ollama_llm_url: str = "http://host.docker.internal:11434"

        # Default values for the mem0 embedding model
        ollama_embedder_model: str = "nomic-embed-text:latest" # This model need to exist in ollama
        ollama_embedder_url: str = "http://host.docker.internal:11434"

    def __init__(self):
        self.type = "filter"
        self.name = "Memory Filter"
        self.user_messages = []
        self.thread = None
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],  # Connect to all pipelines
            }
        )
        self.m = self.init_mem_zero()

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"pipe:{__name__}")

        user = self.valves.mem_zero_user
        store_cycles = self.valves.store_cycles

        if isinstance(body, str):
            body = json.loads(body)

        all_messages = body["messages"]
        last_message = all_messages[-1]["content"]

        self.user_messages.append(last_message)

        if len(self.user_messages) == store_cycles:

            message_text = ""
            for message in self.user_messages:
                message_text += message + " "

            if self.thread and self.thread.is_alive():
                print("Waiting for previous memory to be done")
                self.thread.join()

            self.thread = threading.Thread(target=self.m.add, kwargs={"data":message_text,"user_id":user})

            print("Text to be processed in to a memory:")
            print(message_text)

            self.thread.start()
            self.user_messages.clear()

        memories = self.m.search(last_message, user_id=user)

        if(memories):
            fetched_memory = memories[0]["memory"]
        else:
            fetched_memory = ""

        print("Memory added to the context:")
        print(fetched_memory)

        if fetched_memory:
            all_messages.insert(0, {"role":"system", "content":"This is your inner voice talking, you remember this about the person you chatting with "+str(fetched_memory)})

        print("Final body to send to the LLM:")
        print(body)

        return body

    def init_mem_zero(self):
        config = {
                "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": self.valves.vector_store_qdrant_name,
                    "host": self.valves.vector_store_qdrant_url,
                    "port": self.valves.vector_store_qdrant_port,
                    "embedding_model_dims": self.valves.vector_store_qdrant_dims,
                },
            },
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": self.valves.ollama_llm_model,
                    "temperature": self.valves.ollama_llm_temperature,
                    "max_tokens": self.valves.ollama_llm_tokens,
                    "ollama_base_url": self.valves.ollama_llm_url,
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": self.valves.ollama_embedder_model,
                    "ollama_base_url": self.valves.ollama_embedder_url,
                },
            },
        }

        return Memory.from_config(config)
```

---

```python
# examples/filters/dynamic_ollama_vision_filter_pipeline.py
"""
title: Ollama Dynamic Vision Pipeline
author: Andrew Tait Gehrhardt
date: 2024-06-18
version: 1.0
license: MIT
description: A pipeline for dynamically processing images when current model is a text only model
requirements: pydantic, aiohttp
"""

from typing import List, Optional
from pydantic import BaseModel
import json
import aiohttp
from utils.pipelines.main import get_last_user_message

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        vision_model: str = "llava"
        ollama_base_url: str = ""
        model_to_override: str = ""

    def __init__(self):
        self.type = "filter"
        self.name = "Interception Filter"
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],  # Connect to all pipelines
            }
        )

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

    async def process_images_with_llava(self, images: List[str], content: str, vision_model: str, ollama_base_url: str) -> str:
        url = f"{ollama_base_url}/api/chat"
        payload = {
            "model": vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                    "images": images
                }
            ]
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    content = []
                    async for line in response.content:
                        data = json.loads(line)
                        content.append(data.get("message", {}).get("content", ""))
                    return "".join(content)
                else:
                    print(f"Failed to process images with LLava, status code: {response.status}")
                    return ""

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"pipe:{__name__}")

        images = []

        # Ensure the body is a dictionary
        if isinstance(body, str):
            body = json.loads(body)
        
        model = body.get("model", "")

        # Get the content of the most recent message
        user_message = get_last_user_message(body["messages"])

        if model in self.valves.model_to_override:
            messages = body.get("messages", [])
            for message in messages:
                if "images" in message:
                    images.extend(message["images"])
                    raw_llava_response = await self.process_images_with_llava(images, user_message, self.valves.vision_model,self.valves.ollama_base_url)
                    llava_response = f"REPEAT THIS BACK: {raw_llava_response}"
                    message["content"] = llava_response
                    message.pop("images", None)  # This will safely remove the 'images' key if it exists
        
        return body

```

---

```python
# examples/filters/google_translation_filter_pipeline.py
"""
title: Google Translate Filter
author: SimonOriginal
date: 2024-06-28
version: 1.0
license: MIT
description: This pipeline integrates Google Translate for automatic translation of user and assistant messages 
without requiring an API key. It supports multilingual communication by translating based on specified source 
and target languages.
"""

import re
from typing import List, Optional
from schemas import OpenAIChatMessage
from pydantic import BaseModel
import requests
import os
import time
import asyncio
from functools import lru_cache

from utils.pipelines.main import get_last_user_message, get_last_assistant_message

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        source_user: Optional[str] = "auto"
        target_user: Optional[str] = "en"
        source_assistant: Optional[str] = "en"
        target_assistant: Optional[str] = "uk"

    def __init__(self):
        self.type = "filter"
        self.name = "Google Translate Filter"
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],
            }
        )

        # Initialize translation cache
        self.translation_cache = {}
        self.code_blocks = []  # List to store code blocks

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        pass

    # @lru_cache(maxsize=128)  # LRU cache to store translation results
    def translate(self, text: str, source: str, target: str) -> str:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            "client": "gtx",
            "sl": source,
            "tl": target,
            "dt": "t",
            "q": text,
        }
        
        try:
            r = requests.get(url, params=params)
            r.raise_for_status()
            result = r.json()
            translated_text = ''.join([sentence[0] for sentence in result[0]])
            return translated_text
        except requests.exceptions.RequestException as e:
            print(f"Network error: {e}")
            time.sleep(1)
            return self.translate(text, source, target)
        except Exception as e:
            print(f"Error translating text: {e}")
            return text

    def split_text_around_table(self, text: str) -> List[str]:
        table_regex = r'((?:^.*?\|.*?\n)+)(?=\n[^\|\s].*?\|)'
        matches = re.split(table_regex, text, flags=re.MULTILINE)

        if len(matches) > 1:
            return [matches[0], matches[1]]
        else:
            return [text, ""]

    def clean_table_delimiters(self, text: str) -> str:
        # Remove extra spaces from table delimiters
        return re.sub(r'(\|\s*-+\s*)+', lambda m: m.group(0).replace(' ', '-'), text)

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"inlet:{__name__}")

        messages = body["messages"]
        user_message = get_last_user_message(messages)

        print(f"User message: {user_message}")

        # Find and store code blocks
        code_block_regex = r'```[\s\S]+?```'
        self.code_blocks = re.findall(code_block_regex, user_message)
        # Replace code blocks with placeholders
        user_message_no_code = re.sub(code_block_regex, '__CODE_BLOCK__', user_message)

        parts = self.split_text_around_table(user_message_no_code)
        text_before_table, table_text = parts

        # Check translation cache for text before table
        translated_before_table = self.translation_cache.get(text_before_table)
        if translated_before_table is None:
            translated_before_table = self.translate(
                text_before_table,
                self.valves.source_user,
                self.valves.target_user,
            )
            self.translation_cache[text_before_table] = translated_before_table

        translated_user_message = translated_before_table + table_text

        # Clean table delimiters
        translated_user_message = self.clean_table_delimiters(translated_user_message)

        # Restore code blocks
        for code_block in self.code_blocks:
            translated_user_message = translated_user_message.replace('__CODE_BLOCK__', code_block, 1)

        print(f"Translated user message: {translated_user_message}")

        for message in reversed(messages):
            if message["role"] == "user":
                message["content"] = translated_user_message
                break

        body = {**body, "messages": messages}
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"outlet:{__name__}")

        messages = body["messages"]
        assistant_message = get_last_assistant_message(messages)

        print(f"Assistant message: {assistant_message}")

        # Find and store code blocks
        code_block_regex = r'```[\s\S]+?```'
        self.code_blocks = re.findall(code_block_regex, assistant_message)
        # Replace code blocks with placeholders
        assistant_message_no_code = re.sub(code_block_regex, '__CODE_BLOCK__', assistant_message)

        parts = self.split_text_around_table(assistant_message_no_code)
        text_before_table, table_text = parts

        # Check translation cache for text before table
        translated_before_table = self.translation_cache.get(text_before_table)
        if translated_before_table is None:
            translated_before_table = self.translate(
                text_before_table,
                self.valves.source_assistant,
                self.valves.target_assistant,
            )
            self.translation_cache[text_before_table] = translated_before_table

        translated_assistant_message = translated_before_table + table_text

        # Clean table delimiters
        translated_assistant_message = self.clean_table_delimiters(translated_assistant_message)

        # Restore code blocks
        for code_block in self.code_blocks:
            translated_assistant_message = translated_assistant_message.replace('__CODE_BLOCK__', code_block, 1)

        print(f"Translated assistant message: {translated_assistant_message}")

        for message in reversed(messages):
            if message["role"] == "assistant":
                message["content"] = translated_assistant_message
                break

        body = {**body, "messages": messages}
        return body

```

---

## Contributing to Pipelines

🚀 **Welcome, Contributors!** 🚀

We are thrilled to have you join the Pipelines community! Your contributions are essential to making Pipelines a powerful and versatile framework for extending OpenAI-compatible applications' capabilities. This document provides guidelines to ensure your contributions are smooth and effective.

### 📌 Key Points

- **Scope of Pipelines:** Remember that Pipelines is a framework designed to enhance OpenAI interactions, specifically through a plugin-like approach. Focus your contributions on making Pipelines more robust, flexible, and user-friendly within this context.
- **Open WebUI Integration:** Pipelines is primarily designed to work with Open WebUI. While contributions that expand compatibility with other platforms are welcome, prioritize functionalities that seamlessly integrate with Open WebUI's ecosystem.

### 🚨 Reporting Issues

Encountered a bug or have an idea for improvement? We encourage you to report it! Here's how:

1. **Check Existing Issues:**  Browse the [Issues tab](https://github.com/open-webui/pipelines/issues) to see if the issue or suggestion has already been reported.
2. **Open a New Issue:** If it's a new issue, feel free to open one. Follow the issue template for clear and concise reporting. Provide detailed descriptions, steps to reproduce, expected outcomes, and actual results. This helps us understand and resolve the issue efficiently.

### 🧭 Scope of Support

- **Python Fundamentals:** Pipelines leverages Python. Basic Python knowledge is essential for contributing effectively.

## 💡 Contributing

Ready to make a difference? Here's how you can contribute to Pipelines:

### 🛠 Pull Requests

We encourage pull requests to improve Pipelines! Here's the process:

1. **Discuss Your Idea:** If your contribution involves significant changes, discuss it in the [Issues tab](https://github.com/open-webui/pipelines/issues) first. This ensures your idea aligns with the project's vision.
2. **Coding Standards:** Follow the project's coding standards and write clear, descriptive commit messages.
3. **Update Documentation:**  If your contribution impacts documentation, update it accordingly.
4. **Submit Your Pull Request:** Submit your pull request and provide a clear summary of your changes.

### 📚 Documentation

Help make Pipelines more accessible by:

- **Writing Tutorials:** Create guides for setting up, using, and customizing Pipelines.
- **Improving Documentation:**  Enhance existing documentation for clarity, completeness, and accuracy.
- **Adding Examples:**  Contribute pipelines examples that showcase different functionalities and use cases.

### 🤔 Questions & Feedback

Got questions or feedback? Join our [Discord community](https://discord.gg/5rJgQTnV4s) or open an issue. We're here to help!

## 🙏 Thank You!

Your contributions are invaluable to Pipelines' success! We are excited to see what you bring to the project. Together, we can create a powerful and versatile framework for extending OpenAI capabilities. 🌟

---

```python
# examples/filters/home_assistant_filter.py
"""
title: HomeAssistant Filter Pipeline
author: Andrew Tait Gehrhardt
date: 2024-06-15
version: 1.0
license: MIT
description: A pipeline for controlling Home Assistant entities based on their easy names. Only supports lights at the moment.
requirements: pytz, difflib
"""
import requests
from typing import Literal, Dict, Any
from datetime import datetime
import pytz
from difflib import get_close_matches

from blueprints.function_calling_blueprint import Pipeline as FunctionCallingBlueprint

class Pipeline(FunctionCallingBlueprint):
    class Valves(FunctionCallingBlueprint.Valves):
        HOME_ASSISTANT_URL: str = ""
        HOME_ASSISTANT_TOKEN: str = ""

    class Tools:
        def __init__(self, pipeline) -> None:
            self.pipeline = pipeline

        def get_current_time(self) -> str:
            """
            Get the current time in EST.

            :return: The current time in EST.
            """
            now_est = datetime.now(pytz.timezone('US/Eastern'))  # Get the current time in EST
            current_time = now_est.strftime("%I:%M %p")  # %I for 12-hour clock, %M for minutes, %p for am/pm
            return f"ONLY RESPOND 'Current time is {current_time}'"

        def get_all_lights(self) -> Dict[str, Any]:
            """
            Lists my lights.
            Shows me my lights.
            Get a dictionary of all lights in my home.

            :return: A dictionary of light entity names and their IDs.
            """
            if not self.pipeline.valves.HOME_ASSISTANT_URL or not self.pipeline.valves.HOME_ASSISTANT_TOKEN:
                return {"error": "Home Assistant URL or token not set, ask the user to set it up."}
            else:
                url = f"{self.pipeline.valves.HOME_ASSISTANT_URL}/api/states"
                headers = {
                    "Authorization": f"Bearer {self.pipeline.valves.HOME_ASSISTANT_TOKEN}",
                    "Content-Type": "application/json",
                }

                response = requests.get(url, headers=headers)
                response.raise_for_status()  # Raises an HTTPError for bad responses
                data = response.json()

                lights = {entity["attributes"]["friendly_name"]: entity["entity_id"]
                          for entity in data if entity["entity_id"].startswith("light.")}

                return lights

        def control_light(self, name: str, state: Literal['on', 'off']) -> str:
            """
            Turn a light on or off based on its name.

            :param name: The friendly name of the light.
            :param state: The desired state ('on' or 'off').
            :return: The result of the operation.
            """
            if not self.pipeline.valves.HOME_ASSISTANT_URL or not self.pipeline.valves.HOME_ASSISTANT_TOKEN:
                return "Home Assistant URL or token not set, ask the user to set it up."

            # Normalize the light name by converting to lowercase and stripping extra spaces
            normalized_name = " ".join(name.lower().split())

            # Get a dictionary of all lights
            lights = self.get_all_lights()
            if "error" in lights:
                return lights["error"]

            # Find the closest matching light name
            light_names = list(lights.keys())
            closest_matches = get_close_matches(normalized_name, light_names, n=1, cutoff=0.6)

            if not closest_matches:
                return f"Light named '{name}' not found."

            best_match = closest_matches[0]
            light_id = lights[best_match]

            url = f"{self.pipeline.valves.HOME_ASSISTANT_URL}/api/services/light/turn_{state}"
            headers = {
                "Authorization": f"Bearer {self.pipeline.valves.HOME_ASSISTANT_TOKEN}",
                "Content-Type": "application/json",
            }
            payload = {
                "entity_id": light_id
            }

            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                return f"ONLY RESPOND 'Will do' TO THE USER. DO NOT SAY ANYTHING ELSE!"
            else:
                return f"ONLY RESPOND 'Couldn't find light' TO THE USER. DO NOT SAY ANYTHING ELSE!"

    def __init__(self):
        super().__init__()
        self.name = "My Tools Pipeline"
        self.valves = self.Valves(
            **{
                **self.valves.model_dump(),
                "pipelines": ["*"],  # Connect to all pipelines
            },
        )
        self.tools = self.Tools(self)

```

---

```python
# examples/filters/function_calling_filter_pipeline.py
import os
import requests
from typing import Literal, List, Optional
from datetime import datetime


from blueprints.function_calling_blueprint import Pipeline as FunctionCallingBlueprint


class Pipeline(FunctionCallingBlueprint):
    class Valves(FunctionCallingBlueprint.Valves):
        # Add your custom parameters here
        OPENWEATHERMAP_API_KEY: str = ""
        pass

    class Tools:
        def __init__(self, pipeline) -> None:
            self.pipeline = pipeline

        def get_current_time(
            self,
        ) -> str:
            """
            Get the current time.

            :return: The current time.
            """

            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            return f"Current Time = {current_time}"

        def get_current_weather(
            self,
            location: str,
            unit: Literal["metric", "fahrenheit"] = "fahrenheit",
        ) -> str:
            """
            Get the current weather for a location. If the location is not found, return an empty string.

            :param location: The location to get the weather for.
            :param unit: The unit to get the weather in. Default is fahrenheit.
            :return: The current weather for the location.
            """

            # https://openweathermap.org/api

            if self.pipeline.valves.OPENWEATHERMAP_API_KEY == "":
                return "OpenWeatherMap API Key not set, ask the user to set it up."
            else:
                units = "imperial" if unit == "fahrenheit" else "metric"
                params = {
                    "q": location,
                    "appid": self.pipeline.valves.OPENWEATHERMAP_API_KEY,
                    "units": units,
                }

                response = requests.get(
                    "http://api.openweathermap.org/data/2.5/weather", params=params
                )
                response.raise_for_status()  # Raises an HTTPError for bad responses
                data = response.json()

                weather_description = data["weather"][0]["description"]
                temperature = data["main"]["temp"]

                return f"{location}: {weather_description.capitalize()}, {temperature}°{unit.capitalize()[0]}"

        def calculator(self, equation: str) -> str:
            """
            Calculate the result of an equation.

            :param equation: The equation to calculate.
            """

            # Avoid using eval in production code
            # https://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html
            try:
                result = eval(equation)
                return f"{equation} = {result}"
            except Exception as e:
                print(e)
                return "Invalid equation"

    def __init__(self):
        super().__init__()
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "my_tools_pipeline"
        self.name = "My Tools Pipeline"
        self.valves = self.Valves(
            **{
                **self.valves.model_dump(),
                "pipelines": ["*"],  # Connect to all pipelines
                "OPENWEATHERMAP_API_KEY": os.getenv("OPENWEATHERMAP_API_KEY", ""),
            },
        )
        self.tools = self.Tools(self)

```

---

```python
# examples/filters/rate_limit_filter_pipeline.py
import os
from typing import List, Optional
from pydantic import BaseModel
from schemas import OpenAIChatMessage
import time


class Pipeline:
    class Valves(BaseModel):
        # List target pipeline ids (models) that this filter will be connected to.
        # If you want to connect this filter to all pipelines, you can set pipelines to ["*"]
        pipelines: List[str] = []

        # Assign a priority level to the filter pipeline.
        # The priority level determines the order in which the filter pipelines are executed.
        # The lower the number, the higher the priority.
        priority: int = 0

        # Valves for rate limiting
        requests_per_minute: Optional[int] = None
        requests_per_hour: Optional[int] = None
        sliding_window_limit: Optional[int] = None
        sliding_window_minutes: Optional[int] = None

    def __init__(self):
        # Pipeline filters are only compatible with Open WebUI
        # You can think of filter pipeline as a middleware that can be used to edit the form data before it is sent to the OpenAI API.
        self.type = "filter"

        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "rate_limit_filter_pipeline"
        self.name = "Rate Limit Filter"

        # Initialize rate limits
        self.valves = self.Valves(
            **{
                "pipelines": os.getenv("RATE_LIMIT_PIPELINES", "*").split(","),
                "requests_per_minute": int(
                    os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", 10)
                ),
                "requests_per_hour": int(
                    os.getenv("RATE_LIMIT_REQUESTS_PER_HOUR", 1000)
                ),
                "sliding_window_limit": int(
                    os.getenv("RATE_LIMIT_SLIDING_WINDOW_LIMIT", 100)
                ),
                "sliding_window_minutes": int(
                    os.getenv("RATE_LIMIT_SLIDING_WINDOW_MINUTES", 15)
                ),
            }
        )

        # Tracking data - user_id -> (timestamps of requests)
        self.user_requests = {}

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    def prune_requests(self, user_id: str):
        """Prune old requests that are outside of the sliding window period."""
        now = time.time()
        if user_id in self.user_requests:
            self.user_requests[user_id] = [
                req
                for req in self.user_requests[user_id]
                if (
                    (self.valves.requests_per_minute is not None and now - req < 60)
                    or (self.valves.requests_per_hour is not None and now - req < 3600)
                    or (
                        self.valves.sliding_window_limit is not None
                        and now - req < self.valves.sliding_window_minutes * 60
                    )
                )
            ]

    def log_request(self, user_id: str):
        """Log a new request for a user."""
        now = time.time()
        if user_id not in self.user_requests:
            self.user_requests[user_id] = []
        self.user_requests[user_id].append(now)

    def rate_limited(self, user_id: str) -> bool:
        """Check if a user is rate limited."""
        self.prune_requests(user_id)

        user_reqs = self.user_requests.get(user_id, [])

        if self.valves.requests_per_minute is not None:
            requests_last_minute = sum(1 for req in user_reqs if time.time() - req < 60)
            if requests_last_minute >= self.valves.requests_per_minute:
                return True

        if self.valves.requests_per_hour is not None:
            requests_last_hour = sum(1 for req in user_reqs if time.time() - req < 3600)
            if requests_last_hour >= self.valves.requests_per_hour:
                return True

        if self.valves.sliding_window_limit is not None:
            requests_in_window = len(user_reqs)
            if requests_in_window >= self.valves.sliding_window_limit:
                return True

        return False

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"pipe:{__name__}")
        print(body)
        print(user)

        if user.get("role", "admin") == "user":
            user_id = user["id"] if user and "id" in user else "default_user"
            if self.rate_limited(user_id):
                raise Exception("Rate limit exceeded. Please try again later.")

            self.log_request(user_id)
        return body

```

---

```python
# examples/filters/libretranslate_filter_pipeline.py
from typing import List, Optional
from schemas import OpenAIChatMessage
from pydantic import BaseModel
import requests
import os

from utils.pipelines.main import get_last_user_message, get_last_assistant_message


class Pipeline:

    class Valves(BaseModel):
        # List target pipeline ids (models) that this filter will be connected to.
        # If you want to connect this filter to all pipelines, you can set pipelines to ["*"]
        # e.g. ["llama3:latest", "gpt-3.5-turbo"]
        pipelines: List[str] = []

        # Assign a priority level to the filter pipeline.
        # The priority level determines the order in which the filter pipelines are executed.
        # The lower the number, the higher the priority.
        priority: int = 0

        # Valves
        libretranslate_url: str

        # Source and target languages
        # User message will be translated from source_user to target_user
        source_user: Optional[str] = "auto"
        target_user: Optional[str] = "en"

        # Assistant languages
        # Assistant message will be translated from source_assistant to target_assistant
        source_assistant: Optional[str] = "en"
        target_assistant: Optional[str] = "es"

    def __init__(self):
        # Pipeline filters are only compatible with Open WebUI
        # You can think of filter pipeline as a middleware that can be used to edit the form data before it is sent to the OpenAI API.
        self.type = "filter"

        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "libretranslate_filter_pipeline"
        self.name = "LibreTranslate Filter"

        # Initialize
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],  # Connect to all pipelines
                "libretranslate_url": os.getenv(
                    "LIBRETRANSLATE_API_BASE_URL", "http://localhost:5000"
                ),
            }
        )

        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        pass

    def translate(self, text: str, source: str, target: str) -> str:
        payload = {
            "q": text,
            "source": source,
            "target": target,
        }

        try:
            r = requests.post(
                f"{self.valves.libretranslate_url}/translate", json=payload
            )
            r.raise_for_status()

            data = r.json()
            return data["translatedText"]
        except Exception as e:
            print(f"Error translating text: {e}")
            return text

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"inlet:{__name__}")

        messages = body["messages"]
        user_message = get_last_user_message(messages)

        print(f"User message: {user_message}")

        # Translate user message
        translated_user_message = self.translate(
            user_message,
            self.valves.source_user,
            self.valves.target_user,
        )

        print(f"Translated user message: {translated_user_message}")

        for message in reversed(messages):
            if message["role"] == "user":
                message["content"] = translated_user_message
                break

        body = {**body, "messages": messages}
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"outlet:{__name__}")

        messages = body["messages"]
        assistant_message = get_last_assistant_message(messages)

        print(f"Assistant message: {assistant_message}")

        # Translate assistant message
        translated_assistant_message = self.translate(
            assistant_message,
            self.valves.source_assistant,
            self.valves.target_assistant,
        )

        print(f"Translated assistant message: {translated_assistant_message}")

        for message in reversed(messages):
            if message["role"] == "assistant":
                message["content"] = translated_assistant_message
                break

        body = {**body, "messages": messages}
        return body

```

---

```python
# examples/filters/llm_translate_filter_pipeline.py
from typing import List, Optional
from schemas import OpenAIChatMessage
from pydantic import BaseModel
import requests
import os

from utils.pipelines.main import get_last_user_message, get_last_assistant_message


class Pipeline:
    class Valves(BaseModel):
        # List target pipeline ids (models) that this filter will be connected to.
        # If you want to connect this filter to all pipelines, you can set pipelines to ["*"]
        # e.g. ["llama3:latest", "gpt-3.5-turbo"]
        pipelines: List[str] = []

        # Assign a priority level to the filter pipeline.
        # The priority level determines the order in which the filter pipelines are executed.
        # The lower the number, the higher the priority.
        priority: int = 0

        OPENAI_API_BASE_URL: str = "https://api.openai.com/v1"
        OPENAI_API_KEY: str = ""
        TASK_MODEL: str = "gpt-3.5-turbo"

        # Source and target languages
        # User message will be translated from source_user to target_user
        source_user: Optional[str] = "auto"
        target_user: Optional[str] = "en"

        # Assistant languages
        # Assistant message will be translated from source_assistant to target_assistant
        source_assistant: Optional[str] = "en"
        target_assistant: Optional[str] = "es"

    def __init__(self):
        # Pipeline filters are only compatible with Open WebUI
        # You can think of filter pipeline as a middleware that can be used to edit the form data before it is sent to the OpenAI API.
        self.type = "filter"

        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "libretranslate_filter_pipeline"
        self.name = "LLM Translate Filter"

        # Initialize
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],  # Connect to all pipelines
                "OPENAI_API_KEY": os.getenv(
                    "OPENAI_API_KEY", "your-openai-api-key-here"
                ),
            }
        )

        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        pass

    def translate(self, text: str, source: str, target: str) -> str:
        headers = {}
        headers["Authorization"] = f"Bearer {self.valves.OPENAI_API_KEY}"
        headers["Content-Type"] = "application/json"

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": f"Translate the following text to {target}. Provide only the translated text and nothing else.",
                },
                {"role": "user", "content": text},
            ],
            "model": self.valves.TASK_MODEL,
        }
        print(payload)

        try:
            r = requests.post(
                url=f"{self.valves.OPENAI_API_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                stream=False,
            )

            r.raise_for_status()
            response = r.json()
            print(response)
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error: {e}"

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"inlet:{__name__}")

        messages = body["messages"]
        user_message = get_last_user_message(messages)

        print(f"User message: {user_message}")

        # Translate user message
        translated_user_message = self.translate(
            user_message,
            self.valves.source_user,
            self.valves.target_user,
        )

        print(f"Translated user message: {translated_user_message}")

        for message in reversed(messages):
            if message["role"] == "user":
                message["content"] = translated_user_message
                break

        body = {**body, "messages": messages}
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if "title" in body:
            return body

        print(f"outlet:{__name__}")

        messages = body["messages"]
        assistant_message = get_last_assistant_message(messages)

        print(f"Assistant message: {assistant_message}")

        # Translate assistant message
        translated_assistant_message = self.translate(
            assistant_message,
            self.valves.source_assistant,
            self.valves.target_assistant,
        )

        print(f"Translated assistant message: {translated_assistant_message}")

        for message in reversed(messages):
            if message["role"] == "assistant":
                message["content"] = translated_assistant_message
                break

        body = {**body, "messages": messages}
        return body

```

---

```python
# examples/filters/detoxify_filter_pipeline.py
"""
title: Detoxify Filter Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for filtering out toxic messages using the Detoxify library.
requirements: detoxify
"""

from typing import List, Optional
from schemas import OpenAIChatMessage
from pydantic import BaseModel
from detoxify import Detoxify
import os


class Pipeline:
    class Valves(BaseModel):
        # List target pipeline ids (models) that this filter will be connected to.
        # If you want to connect this filter to all pipelines, you can set pipelines to ["*"]
        # e.g. ["llama3:latest", "gpt-3.5-turbo"]
        pipelines: List[str] = []

        # Assign a priority level to the filter pipeline.
        # The priority level determines the order in which the filter pipelines are executed.
        # The lower the number, the higher the priority.
        priority: int = 0

    def __init__(self):
        # Pipeline filters are only compatible with Open WebUI
        # You can think of filter pipeline as a middleware that can be used to edit the form data before it is sent to the OpenAI API.
        self.type = "filter"

        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "detoxify_filter_pipeline"
        self.name = "Detoxify Filter"

        # Initialize
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],  # Connect to all pipelines
            }
        )

        self.model = None

        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")

        self.model = Detoxify("original")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        pass

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # This filter is applied to the form data before it is sent to the OpenAI API.
        print(f"inlet:{__name__}")

        print(body)
        user_message = body["messages"][-1]["content"]

        # Filter out toxic messages
        toxicity = self.model.predict(user_message)
        print(toxicity)

        if toxicity["toxicity"] > 0.5:
            raise Exception("Toxic message detected")

        return body

```

---

```python
# examples/pipelines/providers/ollama_pipeline.py
from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import requests


class Pipeline:
    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "ollama_pipeline"
        self.name = "Ollama Pipeline"
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        OLLAMA_BASE_URL = "http://localhost:11434"
        MODEL = "llama3"

        if "user" in body:
            print("######################################")
            print(f'# User: {body["user"]["name"]} ({body["user"]["id"]})')
            print(f"# Message: {user_message}")
            print("######################################")

        try:
            r = requests.post(
                url=f"{OLLAMA_BASE_URL}/v1/chat/completions",
                json={**body, "model": MODEL},
                stream=True,
            )

            r.raise_for_status()

            if body["stream"]:
                return r.iter_lines()
            else:
                return r.json()
        except Exception as e:
            return f"Error: {e}"

```

---

```python
# examples/filters/llmguard_prompt_injection_filter_pipeline.py
"""
title: LLM Guard Filter Pipeline
author: jannikstdl
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for filtering out potential prompt injections using the LLM Guard library.
requirements: llm-guard
"""

from typing import List, Optional
from schemas import OpenAIChatMessage
from pydantic import BaseModel
from llm_guard.input_scanners import PromptInjection
from llm_guard.input_scanners.prompt_injection import MatchType
import os

class Pipeline:
    def __init__(self):
        # Pipeline filters are only compatible with Open WebUI
        # You can think of filter pipeline as a middleware that can be used to edit the form data before it is sent to the OpenAI API.
        self.type = "filter"

        # Optionally, you can set the id and name of the pipeline.
        # Assign a unique identifier to the pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        self.id = "llmguard_prompt_injection_filter_pipeline"
        self.name = "LLMGuard Prompt Injection Filter"

        class Valves(BaseModel):
            # List target pipeline ids (models) that this filter will be connected to.
            # If you want to connect this filter to all pipelines, you can set pipelines to ["*"]
            # e.g. ["llama3:latest", "gpt-3.5-turbo"]
            pipelines: List[str] = []

            # Assign a priority level to the filter pipeline.
            # The priority level determines the order in which the filter pipelines are executed.
            # The lower the number, the higher the priority.
            priority: int = 0

        # Initialize
        self.valves = Valves(
            **{
                "pipelines": ["*"],  # Connect to all pipelines
            }
        )

        self.model = None

        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")

        self.model = PromptInjection(threshold=0.8, match_type=MatchType.FULL)
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        pass

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # This filter is applied to the form data before it is sent to the OpenAI API.
        print(f"inlet:{__name__}")

        user_message = body["messages"][-1]["content"]

        # Filter out prompt injection messages
        sanitized_prompt, is_valid, risk_score = self.model.scan(user_message)

        if risk_score > 0.8: 
            raise Exception("Prompt injection detected")

        return body

```

---

```python
# examples/pipelines/providers/cohere_manifold_pipeline.py
"""
title: Cohere Manifold Pipeline
author: justinh-rahb
date: 2024-05-28
version: 1.0
license: MIT
description: A pipeline for generating text using the Anthropic API.
requirements: requests
environment_variables: COHERE_API_KEY
"""

import os
import json
from schemas import OpenAIChatMessage
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import requests


class Pipeline:
    class Valves(BaseModel):
        COHERE_API_BASE_URL: str = "https://api.cohere.com/v1"
        COHERE_API_KEY: str = ""

    def __init__(self):
        self.type = "manifold"

        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.

        self.id = "cohere"

        self.name = "cohere/"

        self.valves = self.Valves(
            **{"COHERE_API_KEY": os.getenv("COHERE_API_KEY", "your-api-key-here")}
        )

        self.pipelines = self.get_cohere_models()

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        # This function is called when the valves are updated.

        self.pipelines = self.get_cohere_models()

        pass

    def get_cohere_models(self):
        if self.valves.COHERE_API_KEY:
            try:
                headers = {}
                headers["Authorization"] = f"Bearer {self.valves.COHERE_API_KEY}"
                headers["Content-Type"] = "application/json"

                r = requests.get(
                    f"{self.valves.COHERE_API_BASE_URL}/models", headers=headers
                )

                models = r.json()
                return [
                    {
                        "id": model["name"],
                        "name": model["name"] if "name" in model else model["name"],
                    }
                    for model in models["models"]
                ]
            except Exception as e:

                print(f"Error: {e}")
                return [
                    {
                        "id": self.id,
                        "name": "Could not fetch models from Cohere, please update the API Key in the valves.",
                    },
                ]
        else:
            return []

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        try:
            if body.get("stream", False):
                return self.stream_response(user_message, model_id, messages, body)
            else:
                return self.get_completion(user_message, model_id, messages, body)
        except Exception as e:
            return f"Error: {e}"

    def stream_response(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Generator:

        headers = {}
        headers["Authorization"] = f"Bearer {self.valves.COHERE_API_KEY}"
        headers["Content-Type"] = "application/json"

        r = requests.post(
            url=f"{self.valves.COHERE_API_BASE_URL}/chat",
            json={
                "model": model_id,
                "chat_history": [
                    {
                        "role": "USER" if message["role"] == "user" else "CHATBOT",
                        "message": message["content"],
                    }
                    for message in messages[:-1]
                ],
                "message": user_message,
                "stream": True,
            },
            headers=headers,
            stream=True,
        )

        r.raise_for_status()

        for line in r.iter_lines():
            if line:
                try:
                    line = json.loads(line)
                    if line["event_type"] == "text-generation":
                        yield line["text"]
                except:
                    pass

    def get_completion(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> str:
        headers = {}
        headers["Authorization"] = f"Bearer {self.valves.COHERE_API_KEY}"
        headers["Content-Type"] = "application/json"

        r = requests.post(
            url=f"{self.valves.COHERE_API_BASE_URL}/chat",
            json={
                "model": model_id,
                "chat_history": [
                    {
                        "role": "USER" if message["role"] == "user" else "CHATBOT",
                        "message": message["content"],
                    }
                    for message in messages[:-1]
                ],
                "message": user_message,
            },
            headers=headers,
        )

        r.raise_for_status()
        data = r.json()

        return data["text"] if "text" in data else "No response from Cohere."

```

---

```python
# examples/scaffolds/example_pipeline_scaffold.py
from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
from pydantic import BaseModel


class Pipeline:
    class Valves(BaseModel):
        pass

    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "pipeline_example"

        # The name of the pipeline.
        self.name = "Pipeline Example"
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        pass

    async def inlet(self, body: dict, user: dict) -> dict:
        # This function is called before the OpenAI API request is made. You can modify the form data before it is sent to the OpenAI API.
        print(f"inlet:{__name__}")

        print(body)
        print(user)

        return body

    async def outlet(self, body: dict, user: dict) -> dict:
        # This function is called after the OpenAI API response is completed. You can modify the messages after they are received from the OpenAI API.
        print(f"outlet:{__name__}")

        print(body)
        print(user)

        return body

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        # If you'd like to check for title generation, you can add the following check
        if body.get("title", False):
            print("Title Generation Request")

        print(messages)
        print(user_message)
        print(body)

        return f"{__name__} response to: {user_message}"

```

---

```python
# examples/pipelines/providers/ollama_manifold_pipeline.py
from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import os

from pydantic import BaseModel
import requests


class Pipeline:

    class Valves(BaseModel):
        OLLAMA_BASE_URL: str

    def __init__(self):
        # You can also set the pipelines that are available in this pipeline.
        # Set manifold to True if you want to use this pipeline as a manifold.
        # Manifold pipelines can have multiple pipelines.
        self.type = "manifold"

        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "ollama_manifold"

        # Optionally, you can set the name of the manifold pipeline.
        self.name = "Ollama: "

        self.valves = self.Valves(
            **{
                "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL", "http://localhost:11435"),
            }
        )
        self.pipelines = []
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        self.pipelines = self.get_ollama_models()
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        print(f"on_valves_updated:{__name__}")
        self.pipelines = self.get_ollama_models()
        pass

    def get_ollama_models(self):
        if self.valves.OLLAMA_BASE_URL:
            try:
                r = requests.get(f"{self.valves.OLLAMA_BASE_URL}/api/tags")
                models = r.json()
                return [
                    {"id": model["model"], "name": model["name"]}
                    for model in models["models"]
                ]
            except Exception as e:
                print(f"Error: {e}")
                return [
                    {
                        "id": "error",
                        "name": "Could not fetch models from Ollama, please update the URL in the valves.",
                    },
                ]
        else:
            return []

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.

        if "user" in body:
            print("######################################")
            print(f'# User: {body["user"]["name"]} ({body["user"]["id"]})')
            print(f"# Message: {user_message}")
            print("######################################")

        try:
            r = requests.post(
                url=f"{self.valves.OLLAMA_BASE_URL}/v1/chat/completions",
                json={**body, "model": model_id},
                stream=True,
            )

            r.raise_for_status()

            if body["stream"]:
                return r.iter_lines()
            else:
                return r.json()
        except Exception as e:
            return f"Error: {e}"

```

---

```python
# examples/filters/presidio_filter_pipeline.py
"""
title: Presidio PII Redaction Pipeline
author: justinh-rahb
date: 2024-07-07
version: 0.1.0
license: MIT
description: A pipeline for redacting personally identifiable information (PII) using the Presidio library.
requirements: presidio-analyzer, presidio-anonymizer
"""

import os
from typing import List, Optional
from pydantic import BaseModel
from schemas import OpenAIChatMessage
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = ["*"]
        priority: int = 0
        enabled_for_admins: bool = False
        entities_to_redact: List[str] = [
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", 
            "CREDIT_CARD", "IP_ADDRESS", "US_PASSPORT", "LOCATION",
            "DATE_TIME", "NRP", "MEDICAL_LICENSE", "URL"
        ]
        language: str = "en"

    def __init__(self):
        self.type = "filter"
        self.name = "Presidio PII Redaction Pipeline"

        self.valves = self.Valves(
            **{
                "pipelines": os.getenv("PII_REDACT_PIPELINES", "*").split(","),
                "enabled_for_admins": os.getenv("PII_REDACT_ENABLED_FOR_ADMINS", "false").lower() == "true",
                "entities_to_redact": os.getenv("PII_REDACT_ENTITIES", ",".join(self.Valves().entities_to_redact)).split(","),
                "language": os.getenv("PII_REDACT_LANGUAGE", "en"),
            }
        )

        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    async def on_startup(self):
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    def redact_pii(self, text: str) -> str:
        results = self.analyzer.analyze(
            text=text,
            language=self.valves.language,
            entities=self.valves.entities_to_redact
        )

        anonymized_text = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators={
                "DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"})
            }
        )

        return anonymized_text.text

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"pipe:{__name__}")
        print(body)
        print(user)

        if user is None or user.get("role") != "admin" or self.valves.enabled_for_admins:
            messages = body.get("messages", [])
            for message in messages:
                if message.get("role") == "user":
                    message["content"] = self.redact_pii(message["content"])

        return body

```

---

```python
# examples/filters/langfuse_filter_pipeline.py
"""
title: Langfuse Filter Pipeline
author: open-webui
date: 2024-09-27
version: 1.4
license: MIT
description: A filter pipeline that uses Langfuse.
requirements: langfuse
"""

from typing import List, Optional
import os
import uuid

from utils.pipelines.main import get_last_assistant_message
from pydantic import BaseModel
from langfuse import Langfuse
from langfuse.api.resources.commons.errors.unauthorized_error import UnauthorizedError

def get_last_assistant_message_obj(messages: List[dict]) -> dict:
    for message in reversed(messages):
        if message["role"] == "assistant":
            return message
    return {}


class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        secret_key: str
        public_key: str
        host: str

    def __init__(self):
        self.type = "filter"
        self.name = "Langfuse Filter"
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],
                "secret_key": os.getenv("LANGFUSE_SECRET_KEY", "your-secret-key-here"),
                "public_key": os.getenv("LANGFUSE_PUBLIC_KEY", "your-public-key-here"),
                "host": os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            }
        )
        self.langfuse = None
        self.chat_generations = {}

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        self.set_langfuse()

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        self.langfuse.flush()

    async def on_valves_updated(self):
        self.set_langfuse()

    def set_langfuse(self):
        try:
            self.langfuse = Langfuse(
                secret_key=self.valves.secret_key,
                public_key=self.valves.public_key,
                host=self.valves.host,
                debug=False,
            )
            self.langfuse.auth_check()
        except UnauthorizedError:
            print(
                "Langfuse credentials incorrect. Please re-enter your Langfuse credentials in the pipeline settings."
            )
        except Exception as e:
            print(f"Langfuse error: {e} Please re-enter your Langfuse credentials in the pipeline settings.")

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"inlet:{__name__}")
        print(f"Received body: {body}")
        print(f"User: {user}")

        # Check for presence of required keys and generate chat_id if missing
        if "chat_id" not in body:
            unique_id = f"SYSTEM MESSAGE {uuid.uuid4()}"
            body["chat_id"] = unique_id
            print(f"chat_id was missing, set to: {unique_id}")

        required_keys = ["model", "messages"]
        missing_keys = [key for key in required_keys if key not in body]
        
        if missing_keys:
            error_message = f"Error: Missing keys in the request body: {', '.join(missing_keys)}"
            print(error_message)
            raise ValueError(error_message)

        trace = self.langfuse.trace(
            name=f"filter:{__name__}",
            input=body,
            user_id=user["email"],
            metadata={"user_name": user["name"], "user_id": user["id"]},
            session_id=body["chat_id"],
        )

        generation = trace.generation(
            name=body["chat_id"],
            model=body["model"],
            input=body["messages"],
            metadata={"interface": "open-webui"},
        )

        self.chat_generations[body["chat_id"]] = generation
        print(trace.get_trace_url())

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"outlet:{__name__}")
        print(f"Received body: {body}")
        if body["chat_id"] not in self.chat_generations:
            return body

        generation = self.chat_generations[body["chat_id"]]
        assistant_message = get_last_assistant_message(body["messages"])

        
        # Extract usage information for models that support it
        usage = None
        assistant_message_obj = get_last_assistant_message_obj(body["messages"])
        if assistant_message_obj:
            info = assistant_message_obj.get("info", {})
            if isinstance(info, dict):
                input_tokens = info.get("prompt_eval_count") or info.get("prompt_tokens")
                output_tokens = info.get("eval_count") or info.get("completion_tokens")
                if input_tokens is not None and output_tokens is not None:
                    usage = {
                        "input": input_tokens,
                        "output": output_tokens,
                        "unit": "TOKENS",
                    }

        # Update generation
        generation.end(
            output=assistant_message,
            metadata={"interface": "open-webui"},
            usage=usage,
        )

        # Clean up the chat_generations dictionary
        del self.chat_generations[body["chat_id"]]

        return body

```

---

```python
# examples/pipelines/providers/openai_pipeline.py
from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
from pydantic import BaseModel
import os
import requests


class Pipeline:
    class Valves(BaseModel):
        OPENAI_API_KEY: str = ""
        pass

    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "openai_pipeline"
        self.name = "OpenAI Pipeline"
        self.valves = self.Valves(
            **{
                "OPENAI_API_KEY": os.getenv(
                    "OPENAI_API_KEY", "your-openai-api-key-here"
                )
            }
        )
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        print(messages)
        print(user_message)

        OPENAI_API_KEY = self.valves.OPENAI_API_KEY
        MODEL = "gpt-3.5-turbo"

        headers = {}
        headers["Authorization"] = f"Bearer {OPENAI_API_KEY}"
        headers["Content-Type"] = "application/json"

        payload = {**body, "model": MODEL}

        if "user" in payload:
            del payload["user"]
        if "chat_id" in payload:
            del payload["chat_id"]
        if "title" in payload:
            del payload["title"]

        print(payload)

        try:
            r = requests.post(
                url="https://api.openai.com/v1/chat/completions",
                json=payload,
                headers=headers,
                stream=True,
            )

            r.raise_for_status()

            if body["stream"]:
                return r.iter_lines()
            else:
                return r.json()
        except Exception as e:
            return f"Error: {e}"

```

---

```python
# examples/pipelines/providers/anthropic_manifold_pipeline.py
"""
title: Anthropic Manifold Pipeline
author: justinh-rahb, sriparashiva
date: 2024-06-20
version: 1.4
license: MIT
description: A pipeline for generating text and processing images using the Anthropic API.
requirements: requests, sseclient-py
environment_variables: ANTHROPIC_API_KEY
"""

import os
import requests
import json
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import sseclient

from utils.pipelines.main import pop_system_message


class Pipeline:
    class Valves(BaseModel):
        ANTHROPIC_API_KEY: str = ""

    def __init__(self):
        self.type = "manifold"
        self.id = "anthropic"
        self.name = "anthropic/"

        self.valves = self.Valves(
            **{"ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", "your-api-key-here")}
        )
        self.url = 'https://api.anthropic.com/v1/messages'
        self.update_headers()

    def update_headers(self):
        self.headers = {
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
            'x-api-key': self.valves.ANTHROPIC_API_KEY
        }

    def get_anthropic_models(self):
        return [
            {"id": "claude-3-haiku-20240307", "name": "claude-3-haiku"},
            {"id": "claude-3-opus-20240229", "name": "claude-3-opus"},
            {"id": "claude-3-sonnet-20240229", "name": "claude-3-sonnet"},
            {"id": "claude-3-5-sonnet-20240620", "name": "claude-3.5-sonnet"},
        ]

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        self.update_headers()

    def pipelines(self) -> List[dict]:
        return self.get_anthropic_models()

    def process_image(self, image_data):
        if image_data["url"].startswith("data:image"):
            mime_type, base64_data = image_data["url"].split(",", 1)
            media_type = mime_type.split(":")[1].split(";")[0]
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data,
                },
            }
        else:
            return {
                "type": "image",
                "source": {"type": "url", "url": image_data["url"]},
            }

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        try:
            # Remove unnecessary keys
            for key in ['user', 'chat_id', 'title']:
                body.pop(key, None)

            system_message, messages = pop_system_message(messages)

            processed_messages = []
            image_count = 0
            total_image_size = 0

            for message in messages:
                processed_content = []
                if isinstance(message.get("content"), list):
                    for item in message["content"]:
                        if item["type"] == "text":
                            processed_content.append({"type": "text", "text": item["text"]})
                        elif item["type"] == "image_url":
                            if image_count >= 5:
                                raise ValueError("Maximum of 5 images per API call exceeded")

                            processed_image = self.process_image(item["image_url"])
                            processed_content.append(processed_image)

                            if processed_image["source"]["type"] == "base64":
                                image_size = len(processed_image["source"]["data"]) * 3 / 4
                            else:
                                image_size = 0

                            total_image_size += image_size
                            if total_image_size > 100 * 1024 * 1024:
                                raise ValueError("Total size of images exceeds 100 MB limit")

                            image_count += 1
                else:
                    processed_content = [{"type": "text", "text": message.get("content", "")}]

                processed_messages.append({"role": message["role"], "content": processed_content})

            # Prepare the payload
            payload = {
                "model": model_id,
                "messages": processed_messages,
                "max_tokens": body.get("max_tokens", 4096),
                "temperature": body.get("temperature", 0.8),
                "top_k": body.get("top_k", 40),
                "top_p": body.get("top_p", 0.9),
                "stop_sequences": body.get("stop", []),
                **({"system": str(system_message)} if system_message else {}),
                "stream": body.get("stream", False),
            }

            if body.get("stream", False):
                return self.stream_response(payload)
            else:
                return self.get_completion(payload)
        except Exception as e:
            return f"Error: {e}"

    def stream_response(self, payload: dict) -> Generator:
        response = requests.post(self.url, headers=self.headers, json=payload, stream=True)

        if response.status_code == 200:
            client = sseclient.SSEClient(response)
            for event in client.events():
                try:
                    data = json.loads(event.data)
                    if data["type"] == "content_block_start":
                        yield data["content_block"]["text"]
                    elif data["type"] == "content_block_delta":
                        yield data["delta"]["text"]
                    elif data["type"] == "message_stop":
                        break
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON: {event.data}")
                except KeyError as e:
                    print(f"Unexpected data structure: {e}")
                    print(f"Full data: {data}")
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")

    def get_completion(self, payload: dict) -> str:
        response = requests.post(self.url, headers=self.headers, json=payload)
        if response.status_code == 200:
            res = response.json()
            return res["content"][0]["text"] if "content" in res and res["content"] else ""
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")

```

---

```python
# examples/scaffolds/manifold_pipeline_scaffold.py
from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage


class Pipeline:
    def __init__(self):
        # You can also set the pipelines that are available in this pipeline.
        # Set manifold to True if you want to use this pipeline as a manifold.
        # Manifold pipelines can have multiple pipelines.
        self.type = "manifold"

        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "manifold_pipeline"

        # Optionally, you can set the name of the manifold pipeline.
        self.name = "Manifold: "

        # Define pipelines that are available in this manifold pipeline.
        # This is a list of dictionaries where each dictionary has an id and name.
        self.pipelines = [
            {
                "id": "pipeline-1",  # This will turn into `manifold_pipeline.pipeline-1`
                "name": "Pipeline 1",  # This will turn into `Manifold: Pipeline 1`
            },
            {
                "id": "pipeline-2",
                "name": "Pipeline 2",
            },
        ]
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        # If you'd like to check for title generation, you can add the following check
        if body.get("title", False):
            print("Title Generation Request")

        print(messages)
        print(user_message)
        print(body)

        return f"{model_id} response to: {user_message}"

```

---

```python
# examples/pipelines/providers/litellm_subprocess_manifold_pipeline.py
"""
title: LiteLLM Subprocess Manifold Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A manifold pipeline that uses LiteLLM as a subprocess.
requirements: yaml, litellm[proxy]
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
from pydantic import BaseModel
import requests


import os
import asyncio
import subprocess
import yaml


class Pipeline:
    class Valves(BaseModel):
        LITELLM_CONFIG_DIR: str = "./litellm/config.yaml"
        LITELLM_PROXY_PORT: int = 4001
        LITELLM_PROXY_HOST: str = "127.0.0.1"
        litellm_config: dict = {}

    def __init__(self):
        # You can also set the pipelines that are available in this pipeline.
        # Set manifold to True if you want to use this pipeline as a manifold.
        # Manifold pipelines can have multiple pipelines.
        self.type = "manifold"

        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "litellm_subprocess_manifold"

        # Optionally, you can set the name of the manifold pipeline.
        self.name = "LiteLLM: "

        # Initialize Valves
        self.valves = self.Valves(**{"LITELLM_CONFIG_DIR": f"./litellm/config.yaml"})
        self.background_process = None
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")

        # Check if the config file exists
        if not os.path.exists(self.valves.LITELLM_CONFIG_DIR):
            with open(self.valves.LITELLM_CONFIG_DIR, "w") as file:
                yaml.dump(
                    {
                        "general_settings": {},
                        "litellm_settings": {},
                        "model_list": [],
                        "router_settings": {},
                    },
                    file,
                )

            print(
                f"Config file not found. Created a default config file at {self.valves.LITELLM_CONFIG_DIR}"
            )

        with open(self.valves.LITELLM_CONFIG_DIR, "r") as file:
            litellm_config = yaml.safe_load(file)

        self.valves.litellm_config = litellm_config

        asyncio.create_task(self.start_litellm_background())
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        await self.shutdown_litellm_background()
        pass

    async def on_valves_updated(self):
        # This function is called when the valves are updated.

        print(f"on_valves_updated:{__name__}")

        with open(self.valves.LITELLM_CONFIG_DIR, "r") as file:
            litellm_config = yaml.safe_load(file)

        self.valves.litellm_config = litellm_config

        await self.shutdown_litellm_background()
        await self.start_litellm_background()
        pass

    async def run_background_process(self, command):
        print("run_background_process")

        try:
            # Log the command to be executed
            print(f"Executing command: {command}")

            # Execute the command and create a subprocess
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.background_process = process
            print("Subprocess started successfully.")

            # Capture STDERR for debugging purposes
            stderr_output = await process.stderr.read()
            stderr_text = stderr_output.decode().strip()
            if stderr_text:
                print(f"Subprocess STDERR: {stderr_text}")

            # log.info output line by line
            async for line in process.stdout:
                print(line.decode().strip())

            # Wait for the process to finish
            returncode = await process.wait()
            print(f"Subprocess exited with return code {returncode}")
        except Exception as e:
            print(f"Failed to start subprocess: {e}")
            raise  # Optionally re-raise the exception if you want it to propagate

    async def start_litellm_background(self):
        print("start_litellm_background")
        # Command to run in the background
        command = [
            "litellm",
            "--port",
            str(self.valves.LITELLM_PROXY_PORT),
            "--host",
            self.valves.LITELLM_PROXY_HOST,
            "--telemetry",
            "False",
            "--config",
            self.valves.LITELLM_CONFIG_DIR,
        ]

        await self.run_background_process(command)

    async def shutdown_litellm_background(self):
        print("shutdown_litellm_background")

        if self.background_process:
            self.background_process.terminate()
            await self.background_process.wait()  # Ensure the process has terminated
            print("Subprocess terminated")
            self.background_process = None

    def get_litellm_models(self):
        if self.background_process:
            try:
                r = requests.get(
                    f"http://{self.valves.LITELLM_PROXY_HOST}:{self.valves.LITELLM_PROXY_PORT}/v1/models"
                )
                models = r.json()
                return [
                    {
                        "id": model["id"],
                        "name": model["name"] if "name" in model else model["id"],
                    }
                    for model in models["data"]
                ]
            except Exception as e:
                print(f"Error: {e}")
                return [
                    {
                        "id": "error",
                        "name": "Could not fetch models from LiteLLM, please update the URL in the valves.",
                    },
                ]
        else:
            return []

    # Pipelines are the models that are available in the manifold.
    # It can be a list or a function that returns a list.
    def pipelines(self) -> List[dict]:
        return self.get_litellm_models()

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        if "user" in body:
            print("######################################")
            print(f'# User: {body["user"]["name"]} ({body["user"]["id"]})')
            print(f"# Message: {user_message}")
            print("######################################")

        try:
            r = requests.post(
                url=f"http://{self.valves.LITELLM_PROXY_HOST}:{self.valves.LITELLM_PROXY_PORT}/v1/chat/completions",
                json={**body, "model": model_id, "user": body["user"]["id"]},
                stream=True,
            )

            r.raise_for_status()

            if body["stream"]:
                return r.iter_lines()
            else:
                return r.json()
        except Exception as e:
            return f"Error: {e}"

```

---

```python
# examples/pipelines/providers/azure_openai_pipeline.py
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import requests
import os


class Pipeline:
    class Valves(BaseModel):
        # You can add your custom valves here.
        AZURE_OPENAI_API_KEY: str
        AZURE_OPENAI_ENDPOINT: str
        AZURE_OPENAI_DEPLOYMENT_NAME: str
        AZURE_OPENAI_API_VERSION: str

    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "azure_openai_pipeline"
        self.name = "Azure OpenAI Pipeline"
        self.valves = self.Valves(
            **{
                "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY", "your-azure-openai-api-key-here"),
                "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT", "your-azure-openai-endpoint-here"),
                "AZURE_OPENAI_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "your-deployment-name-here"),
                "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            }
        )
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    def pipe(
            self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        print(messages)
        print(user_message)

        headers = {
            "api-key": self.valves.AZURE_OPENAI_API_KEY,
            "Content-Type": "application/json",
        }

        url = f"{self.valves.AZURE_OPENAI_ENDPOINT}/openai/deployments/{self.valves.AZURE_OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version={self.valves.AZURE_OPENAI_API_VERSION}"

        allowed_params = {'messages', 'temperature', 'role', 'content', 'contentPart', 'contentPartImage',
                          'enhancements', 'data_sources', 'n', 'stream', 'stop', 'max_tokens', 'presence_penalty',
                          'frequency_penalty', 'logit_bias', 'user', 'function_call', 'functions', 'tools',
                          'tool_choice', 'top_p', 'log_probs', 'top_logprobs', 'response_format', 'seed'}
        # remap user field
        if "user" in body and not isinstance(body["user"], str):
            body["user"] = body["user"]["id"] if "id" in body["user"] else str(body["user"])
        filtered_body = {k: v for k, v in body.items() if k in allowed_params}
        # log fields that were filtered out as a single line
        if len(body) != len(filtered_body):
            print(f"Dropped params: {', '.join(set(body.keys()) - set(filtered_body.keys()))}")

        # Initialize the response variable to None.
        r = None
        try:
            r = requests.post(
                url=url,
                json=filtered_body,
                headers=headers,
                stream=True,
            )

            r.raise_for_status()
            if body["stream"]:
                return r.iter_lines()
            else:
                return r.json()
        except Exception as e:
            if r:
                text = r.text
                return f"Error: {e} ({text})"
            else:
                return f"Error: {e}"

```

---

```python
# examples/pipelines/providers/google_vertexai_manifold_pipeline.py
"""
title: Google GenAI (Vertex AI) Manifold Pipeline
author: Hiromasa Kakehashi
date: 2024-09-19
version: 1.0
license: MIT
description: A pipeline for generating text using Google's GenAI models in Open-WebUI.
requirements: vertexai
environment_variables: GOOGLE_PROJECT_ID, GOOGLE_CLOUD_REGION
usage_instructions:
  To use Gemini with the Vertex AI API, a service account with the appropriate role (e.g., `roles/aiplatform.user`) is required.
  - For deployment on Google Cloud: Associate the service account with the deployment.
  - For use outside of Google Cloud: Set the GOOGLE_APPLICATION_CREDENTIALS environment variable to the path of the service account key file.
"""

import os
from typing import Iterator, List, Union

import vertexai
from pydantic import BaseModel, Field
from vertexai.generative_models import (
    Content,
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)


class Pipeline:
    """Google GenAI pipeline"""

    class Valves(BaseModel):
        """Options to change from the WebUI"""

        GOOGLE_PROJECT_ID: str = ""
        GOOGLE_CLOUD_REGION: str = ""
        USE_PERMISSIVE_SAFETY: bool = Field(default=False)

    def __init__(self):
        self.type = "manifold"
        self.name = "vertexai: "

        self.valves = self.Valves(
            **{
                "GOOGLE_PROJECT_ID": os.getenv("GOOGLE_PROJECT_ID", ""),
                "GOOGLE_CLOUD_REGION": os.getenv("GOOGLE_CLOUD_REGION", ""),
                "USE_PERMISSIVE_SAFETY": False,
            }
        )
        self.pipelines = [
            {"id": "gemini-1.5-flash-001", "name": "Gemini 1.5 Flash"},
            {"id": "gemini-1.5-pro-001", "name": "Gemini 1.5 Pro"},
            {"id": "gemini-flash-experimental", "name": "Gemini 1.5 Flash Experimental"},
            {"id": "gemini-pro-experimental", "name": "Gemini 1.5 Pro Experimental"},
        ]

    async def on_startup(self) -> None:
        """This function is called when the server is started."""

        print(f"on_startup:{__name__}")
        vertexai.init(
            project=self.valves.GOOGLE_PROJECT_ID,
            location=self.valves.GOOGLE_CLOUD_REGION,
        )

    async def on_shutdown(self) -> None:
        """This function is called when the server is stopped."""
        print(f"on_shutdown:{__name__}")

    async def on_valves_updated(self) -> None:
        """This function is called when the valves are updated."""
        print(f"on_valves_updated:{__name__}")
        vertexai.init(
            project=self.valves.GOOGLE_PROJECT_ID,
            location=self.valves.GOOGLE_CLOUD_REGION,
        )

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Iterator]:
        try:
            if not model_id.startswith("gemini-"):
                return f"Error: Invalid model name format: {model_id}"

            print(f"Pipe function called for model: {model_id}")
            print(f"Stream mode: {body.get('stream', False)}")

            system_message = next(
                (msg["content"] for msg in messages if msg["role"] == "system"), None
            )

            model = GenerativeModel(
                model_name=model_id,
                system_instruction=system_message,
            )

            if body.get("title", False):  # If chat title generation is requested
                contents = [Content(role="user", parts=[Part.from_text(user_message)])]
            else:
                contents = self.build_conversation_history(messages)

            generation_config = GenerationConfig(
                temperature=body.get("temperature", 0.7),
                top_p=body.get("top_p", 0.9),
                top_k=body.get("top_k", 40),
                max_output_tokens=body.get("max_tokens", 8192),
                stop_sequences=body.get("stop", []),
            )

            if self.valves.USE_PERMISSIVE_SAFETY:
                safety_settings = {
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                }
            else:
                safety_settings = body.get("safety_settings")

            response = model.generate_content(
                contents,
                stream=body.get("stream", False),
                generation_config=generation_config,
                safety_settings=safety_settings,
            )

            if body.get("stream", False):
                return self.stream_response(response)
            else:
                return response.text

        except Exception as e:
            print(f"Error generating content: {e}")
            return f"An error occurred: {str(e)}"

    def stream_response(self, response):
        for chunk in response:
            if chunk.text:
                print(f"Chunk: {chunk.text}")
                yield chunk.text

    def build_conversation_history(self, messages: List[dict]) -> List[Content]:
        contents = []

        for message in messages:
            if message["role"] == "system":
                continue

            parts = []

            if isinstance(message.get("content"), list):
                for content in message["content"]:
                    if content["type"] == "text":
                        parts.append(Part.from_text(content["text"]))
                    elif content["type"] == "image_url":
                        image_url = content["image_url"]["url"]
                        if image_url.startswith("data:image"):
                            image_data = image_url.split(",")[1]
                            parts.append(Part.from_image(image_data))
                        else:
                            parts.append(Part.from_uri(image_url))
            else:
                parts = [Part.from_text(message["content"])]

            role = "user" if message["role"] == "user" else "model"
            contents.append(Content(role=role, parts=parts))

        return contents

```

---

```python
# examples/scaffolds/function_calling_scaffold.py
from blueprints.function_calling_blueprint import Pipeline as FunctionCallingBlueprint


class Pipeline(FunctionCallingBlueprint):
    class Valves(FunctionCallingBlueprint.Valves):
        # Add your custom valves here
        pass

    class Tools:
        def __init__(self, pipeline) -> None:
            self.pipeline = pipeline

        # Add your custom tools using pure Python code here, make sure to add type hints
        # Use Sphinx-style docstrings to document your tools, they will be used for generating tools specifications
        # Please refer to function_calling_filter_pipeline.py for an example
        pass

    def __init__(self):
        super().__init__()

        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "my_tools_pipeline"
        self.name = "My Tools Pipeline"
        self.valves = self.Valves(
            **{
                **self.valves.model_dump(),
                "pipelines": ["*"],  # Connect to all pipelines
            },
        )
        self.tools = self.Tools(self)

```

---

```python
# examples/scaffolds/filter_pipeline_scaffold.py
"""
title: Filter Pipeline
author: open-webui
date: 2024-05-30
version: 1.1
license: MIT
description: Example of a filter pipeline that can be used to edit the form data before it is sent to the OpenAI API.
requirements: requests
"""

from typing import List, Optional
from pydantic import BaseModel
from schemas import OpenAIChatMessage


class Pipeline:
    class Valves(BaseModel):
        # List target pipeline ids (models) that this filter will be connected to.
        # If you want to connect this filter to all pipelines, you can set pipelines to ["*"]
        pipelines: List[str] = []

        # Assign a priority level to the filter pipeline.
        # The priority level determines the order in which the filter pipelines are executed.
        # The lower the number, the higher the priority.
        priority: int = 0

        # Add your custom parameters here
        pass

    def __init__(self):
        # Pipeline filters are only compatible with Open WebUI
        # You can think of filter pipeline as a middleware that can be used to edit the form data before it is sent to the OpenAI API.
        self.type = "filter"

        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "filter_pipeline"

        self.name = "Filter"

        self.valves = self.Valves(**{"pipelines": ["llama3:latest"]})

        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # This filter is applied to the form data before it is sent to the OpenAI API.
        print(f"inlet:{__name__}")

        # If you'd like to check for title generation, you can add the following check
        if body.get("title", False):
            print("Title Generation Request")

        print(body)
        print(user)

        return body

```

---

```python
# examples/pipelines/providers/perplexity_manifold_pipeline.py
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
import os
import requests

from utils.pipelines.main import pop_system_message


class Pipeline:
    class Valves(BaseModel):
        PERPLEXITY_API_BASE_URL: str = "https://api.perplexity.ai"
        PERPLEXITY_API_KEY: str = ""
        pass

    def __init__(self):
        self.type = "manifold"
        self.name = "Perplexity: "

        self.valves = self.Valves(
            **{
                "PERPLEXITY_API_KEY": os.getenv(
                    "PERPLEXITY_API_KEY", "your-perplexity-api-key-here"
                )
            }
        )

        # Debugging: print the API key to ensure it's loaded
        print(f"Loaded API Key: {self.valves.PERPLEXITY_API_KEY}")

        # List of models
        self.pipelines = [
            {
                "id": "llama-3.1-sonar-large-128k-online",
                "name": "Llama 3.1 Sonar Large 128k Online"
            },
            {
                "id": "llama-3.1-sonar-small-128k-online",
                "name": "Llama 3.1 Sonar Small 128k Online"
            },
            {
                "id": "llama-3.1-sonar-large-128k-chat",
                "name": "Llama 3.1 Sonar Large 128k Chat"
            },
            {
                "id": "llama-3.1-sonar-small-128k-chat",
                "name": "Llama 3.1 Sonar Small 128k Chat"
            },
            {
                "id": "llama-3.1-8b-instruct", "name": "Llama 3.1 8B Instruct"
            },
            {
                "id": "llama-3.1-70b-instruct", "name": "Llama 3.1 70B Instruct"
            }
        ]
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        print(f"on_valves_updated:{__name__}")
        # No models to fetch, static setup
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        system_message, messages = pop_system_message(messages)
        system_prompt = "You are a helpful assistant."
        if system_message is not None:
            system_prompt = system_message["content"]

        print(system_prompt)
        print(messages)
        print(user_message)

        headers = {
            "Authorization": f"Bearer {self.valves.PERPLEXITY_API_KEY}",
            "Content-Type": "application/json",
            "accept": "application/json"
        }

        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                *messages
            ],
            "stream": body.get("stream", True),
            "return_citations": True,
            "return_images": True
        }

        if "user" in payload:
            del payload["user"]
        if "chat_id" in payload:
            del payload["chat_id"]
        if "title" in payload:
            del payload["title"]

        print(payload)

        try:
            r = requests.post(
                url=f"{self.valves.PERPLEXITY_API_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                stream=True,
            )

            r.raise_for_status()

            if body.get("stream", False):
                return r.iter_lines()
            else:
                response = r.json()
                formatted_response = {
                    "id": response["id"],
                    "model": response["model"],
                    "created": response["created"],
                    "usage": response["usage"],
                    "object": response["object"],
                    "choices": [
                        {
                            "index": choice["index"],
                            "finish_reason": choice["finish_reason"],
                            "message": {
                                "role": choice["message"]["role"],
                                "content": choice["message"]["content"]
                            },
                            "delta": {"role": "assistant", "content": ""}
                        } for choice in response["choices"]
                    ]
                }
                return formatted_response
        except Exception as e:
            return f"Error: {e}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Perplexity API Client")
    parser.add_argument("--api-key", type=str, required=True,
                        help="API key for Perplexity")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Prompt to send to the Perplexity API")

    args = parser.parse_args()

    pipeline = Pipeline()
    pipeline.valves.PERPLEXITY_API_KEY = args.api_key
    response = pipeline.pipe(
        user_message=args.prompt, model_id="llama-3-sonar-large-32k-online", messages=[], body={"stream": False})

    print("Response:", response)

```

---

```python
# examples/pipelines/providers/mlx_manifold_pipeline.py
"""
title: MLX Manifold Pipeline
author: justinh-rahb
date: 2024-05-28
version: 2.0
license: MIT
description: A pipeline for generating text using Apple MLX Framework with dynamic model loading.
requirements: requests, mlx-lm, huggingface-hub, psutil
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
from pydantic import BaseModel
import requests
import subprocess
import logging
from huggingface_hub import login
import time
import psutil

class Pipeline:
    class Valves(BaseModel):
        MLX_DEFAULT_MODEL: str = "mlx-community/Meta-Llama-3-8B-Instruct-8bit"
        MLX_MODEL_FILTER: str = "mlx-community"
        MLX_STOP: str = "<|start_header_id|>,<|end_header_id|>,<|eot_id|>"
        MLX_CHAT_TEMPLATE: str | None = None
        MLX_USE_DEFAULT_CHAT_TEMPLATE: bool | None = False
        HUGGINGFACE_TOKEN: str | None = None

    def __init__(self):
        # Pipeline identification
        self.type = "manifold"
        self.id = "mlx"
        self.name = "MLX/"

        # Initialize valves and update them
        self.valves = self.Valves()
        self.update_valves()

        # Server configuration
        self.host = "localhost"  # Always use localhost for security
        self.port = None  # Port will be dynamically assigned

        # Model management
        self.models = self.get_mlx_models()
        self.current_model = None
        self.server_process = None

        # Start the MLX server with the default model
        self.start_mlx_server(self.valves.MLX_DEFAULT_MODEL)

    def update_valves(self):
        """Update pipeline configuration based on valve settings."""
        if self.valves.HUGGINGFACE_TOKEN:
            login(self.valves.HUGGINGFACE_TOKEN)
        self.stop_sequence = self.valves.MLX_STOP.split(",")

    def get_mlx_models(self):
        """Fetch available MLX models based on the specified pattern."""
        try:
            cmd = [
                'mlx_lm.manage',
                '--scan',
                '--pattern', self.valves.MLX_MODEL_FILTER,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            lines = result.stdout.strip().split('\n')
            
            content_lines = [line for line in lines if line and not line.startswith('-')]
            
            models = []
            for line in content_lines[2:]:  # Skip header lines
                parts = line.split()
                if len(parts) >= 2:
                    repo_id = parts[0]
                    models.append({
                        "id": f"{repo_id.split('/')[-1].lower()}",
                        "name": repo_id
                    })
            if not models:
                # Add default model if no models are found
                models.append({
                    "id": f"mlx.{self.valves.MLX_DEFAULT_MODEL.split('/')[-1].lower()}",
                    "name": self.valves.MLX_DEFAULT_MODEL
                })
            return models
        except Exception as e:
            logging.error(f"Error fetching MLX models: {e}")
            # Return default model on error
            return [{
                "id": f"mlx.{self.valves.MLX_DEFAULT_MODEL.split('/')[-1].lower()}",
                "name": self.valves.MLX_DEFAULT_MODEL
            }]

    def pipelines(self) -> List[dict]:
        """Return the list of available models as pipelines."""
        return self.models

    def start_mlx_server(self, model_name):
        """Start the MLX server with the specified model."""
        model_id = f"mlx.{model_name.split('/')[-1].lower()}"
        if self.current_model == model_id and self.server_process and self.server_process.poll() is None:
            logging.info(f"MLX server already running with model {model_name}")
            return

        self.stop_mlx_server()

        self.port = self.find_free_port()

        command = [
            "mlx_lm.server",
            "--model", model_name,
            "--port", str(self.port),
        ]

        # Add chat template options if specified
        if self.valves.MLX_CHAT_TEMPLATE:
            command.extend(["--chat-template", self.valves.MLX_CHAT_TEMPLATE])
        elif self.valves.MLX_USE_DEFAULT_CHAT_TEMPLATE:
            command.append("--use-default-chat-template")

        logging.info(f"Starting MLX server with command: {' '.join(command)}")
        self.server_process = subprocess.Popen(command)
        self.current_model = model_id
        logging.info(f"Started MLX server for model {model_name} on port {self.port}")
        time.sleep(5)  # Give the server some time to start up

    def stop_mlx_server(self):
        """Stop the currently running MLX server."""
        if self.server_process:
            try:
                process = psutil.Process(self.server_process.pid)
                for proc in process.children(recursive=True):
                    proc.terminate()
                process.terminate()
                process.wait(timeout=10)  # Wait for the process to terminate
            except psutil.NoSuchProcess:
                pass  # Process already terminated
            except psutil.TimeoutExpired:
                logging.warning("Timeout while terminating MLX server process")
            finally:
                self.server_process = None
                self.current_model = None
                self.port = None
                logging.info("Stopped MLX server")

    def find_free_port(self):
        """Find and return a free port to use for the MLX server."""
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    async def on_startup(self):
        """Perform any necessary startup operations."""
        logging.info(f"on_startup:{__name__}")

    async def on_shutdown(self):
        """Perform cleanup operations on shutdown."""
        self.stop_mlx_server()

    async def on_valves_updated(self):
        """Handle updates to the pipeline configuration."""
        self.update_valves()
        self.models = self.get_mlx_models()
        self.start_mlx_server(self.valves.MLX_DEFAULT_MODEL)

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """Process a request through the MLX pipeline."""
        logging.info(f"pipe:{__name__}")

        # Switch model if necessary
        if model_id != self.current_model:
            model_name = next((model['name'] for model in self.models if model['id'] == model_id), self.valves.MLX_DEFAULT_MODEL)
            self.start_mlx_server(model_name)

        url = f"http://{self.host}:{self.port}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        # Prepare the payload for the MLX server
        max_tokens = body.get("max_tokens", 4096)
        temperature = body.get("temperature", 0.8)
        repeat_penalty = body.get("repeat_penalty", 1.0)

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "repetition_penalty": repeat_penalty,
            "stop": self.stop_sequence,
            "stream": body.get("stream", False),
        }

        try:
            # Send request to MLX server
            r = requests.post(
                url, headers=headers, json=payload, stream=body.get("stream", False)
            )
            r.raise_for_status()

            # Return streamed response or full JSON response
            if body.get("stream", False):
                return r.iter_lines()
            else:
                return r.json()
        except Exception as e:
            return f"Error: {e}"

```

---

```python
# examples/pipelines/providers/llama_cpp_pipeline.py
"""
title: Llama C++ Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for generating responses using the Llama C++ library.
requirements: llama-cpp-python
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage


class Pipeline:
    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "llama_cpp_pipeline"

        self.name = "Llama C++ Pipeline"
        self.llm = None
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        from llama_cpp import Llama

        self.llm = Llama(
            model_path="./models/llama3.gguf",
            # n_gpu_layers=-1, # Uncomment to use GPU acceleration
            # seed=1337, # Uncomment to set a specific seed
            # n_ctx=2048, # Uncomment to increase the context window
        )

        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        print(messages)
        print(user_message)
        print(body)

        response = self.llm.create_chat_completion_openai_v1(
            messages=messages,
            stream=body["stream"],
        )

        return response

```

---

```python
# examples/pipelines/providers/openai_manifold_pipeline.py
from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
from pydantic import BaseModel

import os
import requests


class Pipeline:
    class Valves(BaseModel):
        OPENAI_API_BASE_URL: str = "https://api.openai.com/v1"
        OPENAI_API_KEY: str = ""
        pass

    def __init__(self):
        self.type = "manifold"
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "openai_pipeline"
        self.name = "OpenAI: "

        self.valves = self.Valves(
            **{
                "OPENAI_API_KEY": os.getenv(
                    "OPENAI_API_KEY", "your-openai-api-key-here"
                )
            }
        )

        self.pipelines = self.get_openai_models()
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        print(f"on_valves_updated:{__name__}")
        self.pipelines = self.get_openai_models()
        pass

    def get_openai_models(self):
        if self.valves.OPENAI_API_KEY:
            try:
                headers = {}
                headers["Authorization"] = f"Bearer {self.valves.OPENAI_API_KEY}"
                headers["Content-Type"] = "application/json"

                r = requests.get(
                    f"{self.valves.OPENAI_API_BASE_URL}/models", headers=headers
                )

                models = r.json()
                return [
                    {
                        "id": model["id"],
                        "name": model["name"] if "name" in model else model["id"],
                    }
                    for model in models["data"]
                    if "gpt" in model["id"]
                ]

            except Exception as e:

                print(f"Error: {e}")
                return [
                    {
                        "id": "error",
                        "name": "Could not fetch models from OpenAI, please update the API Key in the valves.",
                    },
                ]
        else:
            return []

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        print(messages)
        print(user_message)

        headers = {}
        headers["Authorization"] = f"Bearer {self.valves.OPENAI_API_KEY}"
        headers["Content-Type"] = "application/json"

        payload = {**body, "model": model_id}

        if "user" in payload:
            del payload["user"]
        if "chat_id" in payload:
            del payload["chat_id"]
        if "title" in payload:
            del payload["title"]

        print(payload)

        try:
            r = requests.post(
                url=f"{self.valves.OPENAI_API_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                stream=True,
            )

            r.raise_for_status()

            if body["stream"]:
                return r.iter_lines()
            else:
                return r.json()
        except Exception as e:
            return f"Error: {e}"

```

---

```python
# examples/pipelines/providers/google_manifold_pipeline.py
"""
title: Google GenAI Manifold Pipeline
author: Marc Lopez (refactor by justinh-rahb)
date: 2024-06-06
version: 1.3
license: MIT
description: A pipeline for generating text using Google's GenAI models in Open-WebUI.
requirements: google-generativeai
environment_variables: GOOGLE_API_KEY
"""

from typing import List, Union, Iterator
import os

from pydantic import BaseModel, Field

import google.generativeai as genai
from google.generativeai.types import GenerationConfig


class Pipeline:
    """Google GenAI pipeline"""

    class Valves(BaseModel):
        """Options to change from the WebUI"""

        GOOGLE_API_KEY: str = ""
        USE_PERMISSIVE_SAFETY: bool = Field(default=False)

    def __init__(self):
        self.type = "manifold"
        self.id = "google_genai"
        self.name = "Google: "

        self.valves = self.Valves(**{
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", ""),
            "USE_PERMISSIVE_SAFETY": False
        })
        self.pipelines = []

        genai.configure(api_key=self.valves.GOOGLE_API_KEY)
        self.update_pipelines()

    async def on_startup(self) -> None:
        """This function is called when the server is started."""

        print(f"on_startup:{__name__}")
        genai.configure(api_key=self.valves.GOOGLE_API_KEY)
        self.update_pipelines()

    async def on_shutdown(self) -> None:
        """This function is called when the server is stopped."""

        print(f"on_shutdown:{__name__}")

    async def on_valves_updated(self) -> None:
        """This function is called when the valves are updated."""

        print(f"on_valves_updated:{__name__}")
        genai.configure(api_key=self.valves.GOOGLE_API_KEY)
        self.update_pipelines()

    def update_pipelines(self) -> None:
        """Update the available models from Google GenAI"""

        if self.valves.GOOGLE_API_KEY:
            try:
                models = genai.list_models()
                self.pipelines = [
                    {
                        "id": model.name[7:],  # the "models/" part messeses up the URL
                        "name": model.display_name,
                    }
                    for model in models
                    if "generateContent" in model.supported_generation_methods
                    if model.name[:7] == "models/"
                ]
            except Exception:
                self.pipelines = [
                    {
                        "id": "error",
                        "name": "Could not fetch models from Google, please update the API Key in the valves.",
                    }
                ]
        else:
            self.pipelines = []

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Iterator]:
        if not self.valves.GOOGLE_API_KEY:
            return "Error: GOOGLE_API_KEY is not set"

        try:
            genai.configure(api_key=self.valves.GOOGLE_API_KEY)

            if model_id.startswith("google_genai."):
                model_id = model_id[12:]
            model_id = model_id.lstrip(".")

            if not model_id.startswith("gemini-"):
                return f"Error: Invalid model name format: {model_id}"

            print(f"Pipe function called for model: {model_id}")
            print(f"Stream mode: {body.get('stream', False)}")

            system_message = next((msg["content"] for msg in messages if msg["role"] == "system"), None)
            
            contents = []
            for message in messages:
                if message["role"] != "system":
                    if isinstance(message.get("content"), list):
                        parts = []
                        for content in message["content"]:
                            if content["type"] == "text":
                                parts.append({"text": content["text"]})
                            elif content["type"] == "image_url":
                                image_url = content["image_url"]["url"]
                                if image_url.startswith("data:image"):
                                    image_data = image_url.split(",")[1]
                                    parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_data}})
                                else:
                                    parts.append({"image_url": image_url})
                        contents.append({"role": message["role"], "parts": parts})
                    else:
                        contents.append({
                            "role": "user" if message["role"] == "user" else "model",
                            "parts": [{"text": message["content"]}]
                        })
            
            if "gemini-1.5" in model_id:
                model = genai.GenerativeModel(model_name=model_id, system_instruction=system_message)
            else:
                if system_message:
                    contents.insert(0, {"role": "user", "parts": [{"text": f"System: {system_message}"}]})
                
                model = genai.GenerativeModel(model_name=model_id)

            generation_config = GenerationConfig(
                temperature=body.get("temperature", 0.7),
                top_p=body.get("top_p", 0.9),
                top_k=body.get("top_k", 40),
                max_output_tokens=body.get("max_tokens", 8192),
                stop_sequences=body.get("stop", []),
            )

            if self.valves.USE_PERMISSIVE_SAFETY:
                safety_settings = {
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                }
            else:
                safety_settings = body.get("safety_settings")

            response = model.generate_content(
                contents,
                generation_config=generation_config,
                safety_settings=safety_settings,
                stream=body.get("stream", False),
            )

            if body.get("stream", False):
                return self.stream_response(response)
            else:
                return response.text

        except Exception as e:
            print(f"Error generating content: {e}")
            return f"An error occurred: {str(e)}"

    def stream_response(self, response):
        for chunk in response:
            if chunk.text:
                yield chunk.text

```

---

```python
# examples/pipelines/providers/groq_manifold_pipeline.py
from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
from pydantic import BaseModel

import os
import requests


class Pipeline:
    class Valves(BaseModel):
        GROQ_API_BASE_URL: str = "https://api.groq.com/openai/v1"
        GROQ_API_KEY: str = ""
        pass

    def __init__(self):
        self.type = "manifold"
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        self.id = "groq"
        self.name = "Groq: "

        self.valves = self.Valves(
            **{
                "GROQ_API_KEY": os.getenv(
                    "GROQ_API_KEY", "your-groq-api-key-here"
                )
            }
        )

        self.pipelines = self.get_models()
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        print(f"on_valves_updated:{__name__}")
        self.pipelines = self.get_models()
        pass

    def get_models(self):
        if self.valves.GROQ_API_KEY:
            try:
                headers = {}
                headers["Authorization"] = f"Bearer {self.valves.GROQ_API_KEY}"
                headers["Content-Type"] = "application/json"

                r = requests.get(
                    f"{self.valves.GROQ_API_BASE_URL}/models", headers=headers
                )

                models = r.json()
                return [
                    {
                        "id": model["id"],
                        "name": model["name"] if "name" in model else model["id"],
                    }
                    for model in models["data"]
                ]

            except Exception as e:

                print(f"Error: {e}")
                return [
                    {
                        "id": "error",
                        "name": "Could not fetch models from Groq, please update the API Key in the valves.",
                    },
                ]
        else:
            return []

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        print(messages)
        print(user_message)

        headers = {}
        headers["Authorization"] = f"Bearer {self.valves.GROQ_API_KEY}"
        headers["Content-Type"] = "application/json"

        payload = {**body, "model": model_id}

        if "user" in payload:
            del payload["user"]
        if "chat_id" in payload:
            del payload["chat_id"]
        if "title" in payload:
            del payload["title"]

        print(payload)

        try:
            r = requests.post(
                url=f"{self.valves.GROQ_API_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                stream=True,
            )

            r.raise_for_status()

            if body["stream"]:
                return r.iter_lines()
            else:
                return r.json()
        except Exception as e:
            return f"Error: {e}"
```

---

```python
# examples/pipelines/rag/llamaindex_ollama_pipeline.py
"""
title: Llama Index Ollama Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library with Ollama embeddings.
requirements: llama-index, llama-index-llms-ollama, llama-index-embeddings-ollama
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import os

from pydantic import BaseModel


class Pipeline:

    class Valves(BaseModel):
        LLAMAINDEX_OLLAMA_BASE_URL: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str

    def __init__(self):
        self.documents = None
        self.index = None

        self.valves = self.Valves(
            **{
                "LLAMAINDEX_OLLAMA_BASE_URL": os.getenv("LLAMAINDEX_OLLAMA_BASE_URL", "http://localhost:11434"),
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "llama3"),
                "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "nomic-embed-text"),
            }
        )

    async def on_startup(self):
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.llms.ollama import Ollama
        from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader

        Settings.embed_model = OllamaEmbedding(
            model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )
        Settings.llm = Ollama(
            model=self.valves.LLAMAINDEX_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )

        # This function is called when the server is started.
        global documents, index

        self.documents = SimpleDirectoryReader("/app/backend/data").load_data()
        self.index = VectorStoreIndex.from_documents(self.documents)
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        print(messages)
        print(user_message)

        query_engine = self.index.as_query_engine(streaming=True)
        response = query_engine.query(user_message)

        return response.response_gen

```

---

```python
# blueprints/function_calling_blueprint.py
from typing import List, Optional
from pydantic import BaseModel
from schemas import OpenAIChatMessage
import os
import requests
import json

from utils.pipelines.main import (
    get_last_user_message,
    add_or_update_system_message,
    get_tools_specs,
)


class Pipeline:
    class Valves(BaseModel):
        # List target pipeline ids (models) that this filter will be connected to.
        # If you want to connect this filter to all pipelines, you can set pipelines to ["*"]
        pipelines: List[str] = []

        # Assign a priority level to the filter pipeline.
        # The priority level determines the order in which the filter pipelines are executed.
        # The lower the number, the higher the priority.
        priority: int = 0

        # Valves for function calling
        OPENAI_API_BASE_URL: str
        OPENAI_API_KEY: str
        TASK_MODEL: str
        TEMPLATE: str

    def __init__(self):
        # Pipeline filters are only compatible with Open WebUI
        # You can think of filter pipeline as a middleware that can be used to edit the form data before it is sent to the OpenAI API.
        self.type = "filter"

        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "function_calling_blueprint"
        self.name = "Function Calling Blueprint"

        # Initialize valves
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],  # Connect to all pipelines
                "OPENAI_API_BASE_URL": os.getenv(
                    "OPENAI_API_BASE_URL", "https://api.openai.com/v1"
                ),
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY"),
                "TASK_MODEL": os.getenv("TASK_MODEL", "gpt-3.5-turbo"),
                "TEMPLATE": """Use the following context as your learned knowledge, inside <context></context> XML tags.
<context>
    {{CONTEXT}}
</context>

When answer to user:
- If you don't know, just say that you don't know.
- If you don't know when you are not sure, ask for clarification.
Avoid mentioning that you obtained the information from the context.
And answer according to the language of the user's question.""",
            }
        )

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # If title generation is requested, skip the function calling filter
        if body.get("title", False):
            return body

        print(f"pipe:{__name__}")
        print(user)

        # Get the last user message
        user_message = get_last_user_message(body["messages"])

        # Get the tools specs
        tools_specs = get_tools_specs(self.tools)

        # System prompt for function calling
        fc_system_prompt = (
            f"Tools: {json.dumps(tools_specs, indent=2)}"
            + """
If a function tool doesn't match the query, return an empty string. Else, pick a function tool, fill in the parameters from the function tool's schema, and return it in the format { "name": \"functionName\", "parameters": { "key": "value" } }. Only pick a function if the user asks.  Only return the object. Do not return any other text."
"""
        )

        r = None
        try:
            # Call the OpenAI API to get the function response
            r = requests.post(
                url=f"{self.valves.OPENAI_API_BASE_URL}/chat/completions",
                json={
                    "model": self.valves.TASK_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": fc_system_prompt,
                        },
                        {
                            "role": "user",
                            "content": "History:\n"
                            + "\n".join(
                                [
                                    f"{message['role']}: {message['content']}"
                                    for message in body["messages"][::-1][:4]
                                ]
                            )
                            + f"Query: {user_message}",
                        },
                    ],
                    # TODO: dynamically add response_format?
                    # "response_format": {"type": "json_object"},
                },
                headers={
                    "Authorization": f"Bearer {self.valves.OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                stream=False,
            )
            r.raise_for_status()

            response = r.json()
            content = response["choices"][0]["message"]["content"]

            # Parse the function response
            if content != "":
                result = json.loads(content)
                print(result)

                # Call the function
                if "name" in result:
                    function = getattr(self.tools, result["name"])
                    function_result = None
                    try:
                        function_result = function(**result["parameters"])
                    except Exception as e:
                        print(e)

                    # Add the function result to the system prompt
                    if function_result:
                        system_prompt = self.valves.TEMPLATE.replace(
                            "{{CONTEXT}}", function_result
                        )

                        print(system_prompt)
                        messages = add_or_update_system_message(
                            system_prompt, body["messages"]
                        )

                        # Return the updated messages
                        return {**body, "messages": messages}

        except Exception as e:
            print(f"Error: {e}")

            if r:
                try:
                    print(r.json())
                except:
                    pass

        return body

```

---

# Security Policy

Our primary goal is to ensure the protection and confidentiality of sensitive data stored by users on Pipelines. Additionally, we aim to maintain a secure and trusted environment for executing Pipelines, which effectively function as a plugin system with arbitrary code execution capabilities.

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |
| others  | :x:                |

## Secure Pipelines Execution

To mitigate risks associated with the Pipelines plugin system, we recommend the following best practices:

1. **Trusted Sources**: Only fetch and execute Pipelines from trusted sources. Do not retrieve or run Pipelines from untrusted or unknown origins.

2. **Fixed Versions**: Instead of pulling the latest version of a Pipeline, consider using a fixed, audited version to ensure stability and security.

3. **Sandboxing**: Pipelines are executed in a sandboxed environment to limit their access to system resources and prevent potential harm.

4. **Code Review**: All Pipelines undergo a thorough code review process before being approved for execution on our platform.

5. **Monitoring**: We continuously monitor the execution of Pipelines for any suspicious or malicious activities.

## Reporting a Vulnerability

If you discover a security issue within our system, please notify us immediately via a pull request or contact us on discord. We take all security reports seriously and will respond promptly.

## Product Security

We regularly audit our internal processes and system's architecture for vulnerabilities using a combination of automated and manual testing techniques. We are committed to implementing Static Application Security Testing (SAST) and Software Composition Analysis (SCA) scans in our project to further enhance our security posture.

---

```python
# examples/pipelines/providers/mlx_pipeline.py
"""
title: MLX Pipeline
author: justinh-rahb
date: 2024-05-27
version: 1.2
license: MIT
description: A pipeline for generating text using Apple MLX Framework.
requirements: requests, mlx-lm, huggingface-hub
environment_variables: MLX_HOST, MLX_PORT, MLX_SUBPROCESS
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
from pydantic import BaseModel
import requests
import os
import subprocess
import logging
from huggingface_hub import login

class Pipeline:
    class Valves(BaseModel):
        MLX_MODEL: str = "mistralai/Mistral-7B-Instruct-v0.3"
        MLX_STOP: str = "[INST]"
        HUGGINGFACE_TOKEN: str = ""

    def __init__(self):
        self.id = "mlx_pipeline"
        self.name = "MLX Pipeline"
        self.valves = self.Valves()
        self.update_valves()

        self.host = os.getenv("MLX_HOST", "localhost")
        self.port = os.getenv("MLX_PORT", "8080")
        self.subprocess = os.getenv("MLX_SUBPROCESS", "true").lower() == "true"

        if self.subprocess:
            self.start_mlx_server()

    def update_valves(self):
        if self.valves.HUGGINGFACE_TOKEN:
            login(self.valves.HUGGINGFACE_TOKEN)
        self.stop_sequence = self.valves.MLX_STOP.split(",")

    def start_mlx_server(self):
        if not os.getenv("MLX_PORT"):
            self.port = self.find_free_port()
        command = f"mlx_lm.server --model {self.valves.MLX_MODEL} --port {self.port}"
        self.server_process = subprocess.Popen(command, shell=True)
        logging.info(f"Started MLX server on port {self.port}")

    def find_free_port(self):
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    async def on_startup(self):
        logging.info(f"on_startup:{__name__}")

    async def on_shutdown(self):
        if self.subprocess and hasattr(self, "server_process"):
            self.server_process.terminate()
            logging.info(f"Terminated MLX server on port {self.port}")

    async def on_valves_updated(self):
        self.update_valves()
        if self.subprocess and hasattr(self, "server_process"):
            self.server_process.terminate()
            logging.info(f"Terminated MLX server on port {self.port}")
            self.start_mlx_server()

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        logging.info(f"pipe:{__name__}")

        url = f"http://{self.host}:{self.port}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        max_tokens = body.get("max_tokens", 4096)
        if not isinstance(max_tokens, int) or max_tokens < 0:
            max_tokens = 4096

        temperature = body.get("temperature", 0.8)
        if not isinstance(temperature, (int, float)) or temperature < 0:
            temperature = 0.8

        repeat_penalty = body.get("repeat_penalty", 1.0)
        if not isinstance(repeat_penalty, (int, float)) or repeat_penalty < 0:
            repeat_penalty = 1.0

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "repetition_penalty": repeat_penalty,
            "stop": self.stop_sequence,
            "stream": body.get("stream", False),
        }

        try:
            r = requests.post(
                url, headers=headers, json=payload, stream=body.get("stream", False)
            )
            r.raise_for_status()

            if body.get("stream", False):
                return r.iter_lines()
            else:
                return r.json()
        except Exception as e:
            return f"Error: {e}"
```

---

```python
# examples/pipelines/providers/openai_dalle_manifold_pipeline.py
"""A manifold to integrate OpenAI's ImageGen models into Open-WebUI"""

from typing import List, Union, Generator, Iterator

from pydantic import BaseModel

from openai import OpenAI

class Pipeline:
    """OpenAI ImageGen pipeline"""

    class Valves(BaseModel):
        """Options to change from the WebUI"""

        OPENAI_API_BASE_URL: str = "https://api.openai.com/v1"
        OPENAI_API_KEY: str = ""
        IMAGE_SIZE: str = "1024x1024"
        NUM_IMAGES: int = 1

    def __init__(self):
        self.type = "manifold"
        self.name = "ImageGen: "

        self.valves = self.Valves()
        self.client = OpenAI(
            base_url=self.valves.OPENAI_API_BASE_URL,
            api_key=self.valves.OPENAI_API_KEY,
        )

        self.pipelines = self.get_openai_assistants()

    async def on_startup(self) -> None:
        """This function is called when the server is started."""
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        """This function is called when the server is stopped."""
        print(f"on_shutdown:{__name__}")

    async def on_valves_updated(self):
        """This function is called when the valves are updated."""
        print(f"on_valves_updated:{__name__}")
        self.client = OpenAI(
            base_url=self.valves.OPENAI_API_BASE_URL,
            api_key=self.valves.OPENAI_API_KEY,
        )
        self.pipelines = self.get_openai_assistants()

    def get_openai_assistants(self) -> List[dict]:
        """Get the available ImageGen models from OpenAI

        Returns:
            List[dict]: The list of ImageGen models
        """

        if self.valves.OPENAI_API_KEY:
            models = self.client.models.list()
            return [
                {
                    "id": model.id,
                    "name": model.id,
                }
                for model in models
                if "dall-e" in model.id
            ]

        return []

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")

        response = self.client.images.generate(
            model=model_id,
            prompt=user_message,
            size=self.valves.IMAGE_SIZE,
            n=self.valves.NUM_IMAGES,
        )

        message = ""
        for image in response.data:
            if image.url:
                message += "![image](" + image.url + ")\n"

        yield message

```

---

```python
# examples/pipelines/providers/cloudflare_ai_pipeline.py
from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
from pydantic import BaseModel
import os
import requests


class Pipeline:
    class Valves(BaseModel):
        CLOUDFLARE_ACCOUNT_ID: str = ""
        CLOUDFLARE_API_KEY: str = ""
        CLOUDFLARE_MODEL: str = ""
        pass

    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "openai_pipeline"
        self.name = "Cloudlfare AI"
        self.valves = self.Valves(
            **{
                "CLOUDFLARE_ACCOUNT_ID": os.getenv(
                    "CLOUDFLARE_ACCOUNT_ID",
                    "your-account-id",
                ),
                "CLOUDFLARE_API_KEY": os.getenv(
                    "CLOUDFLARE_API_KEY", "your-cloudflare-api-key"
                ),
                "CLOUDFLARE_MODEL": os.getenv(
                    "CLOUDFLARE_MODELS",
                    "@cf/meta/llama-3.1-8b-instruct",
                ),
            }
        )
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        headers = {}
        headers["Authorization"] = f"Bearer {self.valves.CLOUDFLARE_API_KEY}"
        headers["Content-Type"] = "application/json"

        payload = {**body, "model": self.valves.CLOUDFLARE_MODEL}

        if "user" in payload:
            del payload["user"]
        if "chat_id" in payload:
            del payload["chat_id"]
        if "title" in payload:
            del payload["title"]

        try:
            r = requests.post(
                url=f"https://api.cloudflare.com/client/v4/accounts/{self.valves.CLOUDFLARE_ACCOUNT_ID}/ai/v1/chat/completions",
                json=payload,
                headers=headers,
                stream=True,
            )

            r.raise_for_status()

            if body["stream"]:
                return r.iter_lines()
            else:
                return r.json()
        except Exception as e:
            return f"Error: {e}"

```

---

```python
# examples/pipelines/integrations/python_code_pipeline.py
from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import subprocess


class Pipeline:
    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "python_code_pipeline"
        self.name = "Python Code Pipeline"
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    def execute_python_code(self, code):
        try:
            result = subprocess.run(
                ["python", "-c", code], capture_output=True, text=True, check=True
            )
            stdout = result.stdout.strip()
            return stdout, result.returncode
        except subprocess.CalledProcessError as e:
            return e.output.strip(), e.returncode

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        print(messages)
        print(user_message)

        if body.get("title", False):
            print("Title Generation")
            return "Python Code Pipeline"
        else:
            stdout, return_code = self.execute_python_code(user_message)
            return stdout

```

---

```python
# examples/pipelines/rag/haystack_pipeline.py
"""
title: Haystack Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Haystack library.
requirements: haystack-ai, datasets>=2.6.1, sentence-transformers>=2.2.0
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import os
import asyncio


class Pipeline:
    def __init__(self):
        self.basic_rag_pipeline = None

    async def on_startup(self):
        os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

        from haystack.components.embedders import SentenceTransformersDocumentEmbedder
        from haystack.components.embedders import SentenceTransformersTextEmbedder
        from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
        from haystack.components.builders import PromptBuilder
        from haystack.components.generators import OpenAIGenerator

        from haystack.document_stores.in_memory import InMemoryDocumentStore

        from datasets import load_dataset
        from haystack import Document
        from haystack import Pipeline

        document_store = InMemoryDocumentStore()

        dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
        docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]

        doc_embedder = SentenceTransformersDocumentEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        doc_embedder.warm_up()

        docs_with_embeddings = doc_embedder.run(docs)
        document_store.write_documents(docs_with_embeddings["documents"])

        text_embedder = SentenceTransformersTextEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )

        retriever = InMemoryEmbeddingRetriever(document_store)

        template = """
        Given the following information, answer the question.

        Context:
        {% for document in documents %}
            {{ document.content }}
        {% endfor %}

        Question: {{question}}
        Answer:
        """

        prompt_builder = PromptBuilder(template=template)

        generator = OpenAIGenerator(model="gpt-3.5-turbo")

        self.basic_rag_pipeline = Pipeline()
        # Add components to your pipeline
        self.basic_rag_pipeline.add_component("text_embedder", text_embedder)
        self.basic_rag_pipeline.add_component("retriever", retriever)
        self.basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
        self.basic_rag_pipeline.add_component("llm", generator)

        # Now, connect the components to each other
        self.basic_rag_pipeline.connect(
            "text_embedder.embedding", "retriever.query_embedding"
        )
        self.basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
        self.basic_rag_pipeline.connect("prompt_builder", "llm")

        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        print(messages)
        print(user_message)

        question = user_message
        response = self.basic_rag_pipeline.run(
            {
                "text_embedder": {"text": question},
                "prompt_builder": {"question": question},
            }
        )

        return response["llm"]["replies"][0]

```

---

```python
# examples/pipelines/providers/litellm_manifold_pipeline.py
"""
title: LiteLLM Manifold Pipeline
author: open-webui
date: 2024-05-30
version: 1.0.1
license: MIT
description: A manifold pipeline that uses LiteLLM.
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
from pydantic import BaseModel
import requests
import os


class Pipeline:

    class Valves(BaseModel):
        LITELLM_BASE_URL: str = ""
        LITELLM_API_KEY: str = ""
        LITELLM_PIPELINE_DEBUG: bool = False

    def __init__(self):
        # You can also set the pipelines that are available in this pipeline.
        # Set manifold to True if you want to use this pipeline as a manifold.
        # Manifold pipelines can have multiple pipelines.
        self.type = "manifold"

        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "litellm_manifold"

        # Optionally, you can set the name of the manifold pipeline.
        self.name = "LiteLLM: "

        # Initialize rate limits
        self.valves = self.Valves(
            **{
                "LITELLM_BASE_URL": os.getenv(
                    "LITELLM_BASE_URL", "http://localhost:4001"
                ),
                "LITELLM_API_KEY": os.getenv("LITELLM_API_KEY", "your-api-key-here"),
                "LITELLM_PIPELINE_DEBUG": os.getenv("LITELLM_PIPELINE_DEBUG", False),
            }
        )
        # Get models on initialization
        self.pipelines = self.get_litellm_models()
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        # Get models on startup
        self.pipelines = self.get_litellm_models()
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        # This function is called when the valves are updated.

        self.pipelines = self.get_litellm_models()
        pass

    def get_litellm_models(self):

        headers = {}
        if self.valves.LITELLM_API_KEY:
            headers["Authorization"] = f"Bearer {self.valves.LITELLM_API_KEY}"

        if self.valves.LITELLM_BASE_URL:
            try:
                r = requests.get(
                    f"{self.valves.LITELLM_BASE_URL}/v1/models", headers=headers
                )
                models = r.json()
                return [
                    {
                        "id": model["id"],
                        "name": model["name"] if "name" in model else model["id"],
                    }
                    for model in models["data"]
                ]
            except Exception as e:
                print(f"Error fetching models from LiteLLM: {e}")
                return [
                    {
                        "id": "error",
                        "name": "Could not fetch models from LiteLLM, please update the URL in the valves.",
                    },
                ]
        else:
            print("LITELLM_BASE_URL not set. Please configure it in the valves.")
            return []

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        if "user" in body:
            print("######################################")
            print(f'# User: {body["user"]["name"]} ({body["user"]["id"]})')
            print(f"# Message: {user_message}")
            print("######################################")

        headers = {}
        if self.valves.LITELLM_API_KEY:
            headers["Authorization"] = f"Bearer {self.valves.LITELLM_API_KEY}"

        try:
            payload = {**body, "model": model_id, "user": body["user"]["id"]}
            payload.pop("chat_id", None)
            payload.pop("user", None)
            payload.pop("title", None)

            r = requests.post(
                url=f"{self.valves.LITELLM_BASE_URL}/v1/chat/completions",
                json=payload,
                headers=headers,
                stream=True,
            )

            r.raise_for_status()

            if body["stream"]:
                return r.iter_lines()
            else:
                return r.json()
        except Exception as e:
            return f"Error: {e}"

```

---

```python
# examples/pipelines/providers/aws_bedrock_claude_pipeline.py
"""
title: AWS Bedrock Claude Pipeline
author: G-mario
date: 2024-08-18
version: 1.0
license: MIT
description: A pipeline for generating text and processing images using the AWS Bedrock API(By Anthropic claude).
requirements: requests, boto3
environment_variables: AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION_NAME
"""
import base64
import json
import logging
from io import BytesIO
from typing import List, Union, Generator, Iterator

import boto3

from pydantic import BaseModel

import os
import requests

from utils.pipelines.main import pop_system_message


class Pipeline:
    class Valves(BaseModel):
        AWS_ACCESS_KEY: str = ""
        AWS_SECRET_KEY: str = ""
        AWS_REGION_NAME: str = ""

    def __init__(self):
        self.type = "manifold"
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "openai_pipeline"
        self.name = "Bedrock: "

        self.valves = self.Valves(
            **{
                "AWS_ACCESS_KEY": os.getenv("AWS_ACCESS_KEY", "your-aws-access-key-here"),
                "AWS_SECRET_KEY": os.getenv("AWS_SECRET_KEY", "your-aws-secret-key-here"),
                "AWS_REGION_NAME": os.getenv("AWS_REGION_NAME", "your-aws-region-name-here"),
            }
        )

        self.bedrock = boto3.client(aws_access_key_id=self.valves.AWS_ACCESS_KEY,
                                    aws_secret_access_key=self.valves.AWS_SECRET_KEY,
                                    service_name="bedrock",
                                    region_name=self.valves.AWS_REGION_NAME)
        self.bedrock_runtime = boto3.client(aws_access_key_id=self.valves.AWS_ACCESS_KEY,
                                            aws_secret_access_key=self.valves.AWS_SECRET_KEY,
                                            service_name="bedrock-runtime",
                                            region_name=self.valves.AWS_REGION_NAME)

        self.pipelines = self.get_models()


    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        print(f"on_valves_updated:{__name__}")
        self.bedrock = boto3.client(aws_access_key_id=self.valves.AWS_ACCESS_KEY,
                                    aws_secret_access_key=self.valves.AWS_SECRET_KEY,
                                    service_name="bedrock",
                                    region_name=self.valves.AWS_REGION_NAME)
        self.bedrock_runtime = boto3.client(aws_access_key_id=self.valves.AWS_ACCESS_KEY,
                                            aws_secret_access_key=self.valves.AWS_SECRET_KEY,
                                            service_name="bedrock-runtime",
                                            region_name=self.valves.AWS_REGION_NAME)
        self.pipelines = self.get_models()

    def pipelines(self) -> List[dict]:
        return self.get_models()

    def get_models(self):
        if self.valves.AWS_ACCESS_KEY and self.valves.AWS_SECRET_KEY:
            try:
                response = self.bedrock.list_foundation_models(byProvider='Anthropic', byInferenceType='ON_DEMAND')
                return [
                    {
                        "id": model["modelId"],
                        "name": model["modelName"],
                    }
                    for model in response["modelSummaries"]
                ]
            except Exception as e:
                print(f"Error: {e}")
                return [
                    {
                        "id": "error",
                        "name": "Could not fetch models from Bedrock, please update the Access/Secret Key in the valves.",
                    },
                ]
        else:
            return []

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        system_message, messages = pop_system_message(messages)

        logging.info(f"pop_system_message: {json.dumps(messages)}")

        try:
            processed_messages = []
            image_count = 0
            for message in messages:
                processed_content = []
                if isinstance(message.get("content"), list):
                    for item in message["content"]:
                        if item["type"] == "text":
                            processed_content.append({"text": item["text"]})
                        elif item["type"] == "image_url":
                            if image_count >= 20:
                                raise ValueError("Maximum of 20 images per API call exceeded")
                            processed_image = self.process_image(item["image_url"])
                            processed_content.append(processed_image)
                            image_count += 1
                else:
                    processed_content = [{"text": message.get("content", "")}]

                processed_messages.append({"role": message["role"], "content": processed_content})

            payload = {"modelId": model_id,
                       "messages": processed_messages,
                       "system": [{'text': system_message if system_message else 'you are an intelligent ai assistant'}],
                       "inferenceConfig": {"temperature": body.get("temperature", 0.5)},
                       "additionalModelRequestFields": {"top_k": body.get("top_k", 200), "top_p": body.get("top_p", 0.9)}
                       }
            if body.get("stream", False):
                return self.stream_response(model_id, payload)
            else:
                return self.get_completion(model_id, payload)
        except Exception as e:
            return f"Error: {e}"

    def process_image(self, image: str):
        img_stream = None
        if image["url"].startswith("data:image"):
            if ',' in image["url"]:
                base64_string = image["url"].split(',')[1]
            image_data = base64.b64decode(base64_string)

            img_stream = BytesIO(image_data)
        else:
            img_stream = requests.get(image["url"]).content
        return {
            "image": {"format": "png" if image["url"].endswith(".png") else "jpeg",
                      "source": {"bytes": img_stream.read()}}
        }

    def stream_response(self, model_id: str, payload: dict) -> Generator:
        if "system" in payload:
            del payload["system"]
        if "additionalModelRequestFields" in payload:
            del payload["additionalModelRequestFields"]
        streaming_response = self.bedrock_runtime.converse_stream(**payload)
        for chunk in streaming_response["stream"]:
            if "contentBlockDelta" in chunk:
                yield chunk["contentBlockDelta"]["delta"]["text"]

    def get_completion(self, model_id: str, payload: dict) -> str:
        response = self.bedrock_runtime.converse(**payload)
        return response['output']['message']['content'][0]['text']


```

---

```python
# examples/filters/conversation_turn_limit_filter.py
import os
from typing import List, Optional
from pydantic import BaseModel
from schemas import OpenAIChatMessage
import time


class Pipeline:
    class Valves(BaseModel):
        # List target pipeline ids (models) that this filter will be connected to.
        # If you want to connect this filter to all pipelines, you can set pipelines to ["*"]
        pipelines: List[str] = []

        # Assign a priority level to the filter pipeline.
        # The priority level determines the order in which the filter pipelines are executed.
        # The lower the number, the higher the priority.
        priority: int = 0

        # Valves for conversation turn limiting
        target_user_roles: List[str] = ["user"]
        max_turns: Optional[int] = None

    def __init__(self):
        # Pipeline filters are only compatible with Open WebUI
        # You can think of filter pipeline as a middleware that can be used to edit the form data before it is sent to the OpenAI API.
        self.type = "filter"

        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "conversation_turn_limit_filter_pipeline"
        self.name = "Conversation Turn Limit Filter"

        self.valves = self.Valves(
            **{
                "pipelines": os.getenv("CONVERSATION_TURN_PIPELINES", "*").split(","),
                "max_turns": 10,
            }
        )

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"pipe:{__name__}")
        print(body)
        print(user)

        if user.get("role", "admin") in self.valves.target_user_roles:
            messages = body.get("messages", [])
            if len(messages) > self.valves.max_turns:
                raise Exception(
                    f"Conversation turn limit exceeded. Max turns: {self.valves.max_turns}"
                )

        return body

```

---

```python
# examples/pipelines/rag/llamaindex_pipeline.py
"""
title: Llama Index Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library.
requirements: llama-index
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage


class Pipeline:
    def __init__(self):
        self.documents = None
        self.index = None

    async def on_startup(self):
        import os

        # Set the OpenAI API key
        os.environ["OPENAI_API_KEY"] = "your-api-key-here"

        from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

        self.documents = SimpleDirectoryReader("./data").load_data()
        self.index = VectorStoreIndex.from_documents(self.documents)
        # This function is called when the server is started.
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        print(messages)
        print(user_message)

        query_engine = self.index.as_query_engine(streaming=True)
        response = query_engine.query(user_message)

        return response.response_gen

```

---

```python
# examples/pipelines/rag/llamaindex_ollama_github_pipeline.py
"""
title: Llama Index Ollama Github Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library with Ollama embeddings from a GitHub repository.
requirements: llama-index, llama-index-llms-ollama, llama-index-embeddings-ollama, llama-index-readers-github
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import os
import asyncio


class Pipeline:
    def __init__(self):
        self.documents = None
        self.index = None

    async def on_startup(self):
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.llms.ollama import Ollama
        from llama_index.core import VectorStoreIndex, Settings
        from llama_index.readers.github import GithubRepositoryReader, GithubClient

        Settings.embed_model = OllamaEmbedding(
            model_name="nomic-embed-text",
            base_url="http://localhost:11434",
        )
        Settings.llm = Ollama(model="llama3")

        global index, documents

        github_token = os.environ.get("GITHUB_TOKEN")
        owner = "open-webui"
        repo = "plugin-server"
        branch = "main"

        github_client = GithubClient(github_token=github_token, verbose=True)

        reader = GithubRepositoryReader(
            github_client=github_client,
            owner=owner,
            repo=repo,
            use_parser=False,
            verbose=False,
            filter_file_extensions=(
                [
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".gif",
                    ".svg",
                    ".ico",
                    "json",
                    ".ipynb",
                ],
                GithubRepositoryReader.FilterType.EXCLUDE,
            ),
        )

        loop = asyncio.new_event_loop()

        reader._loop = loop

        try:
            # Load data from the branch
            self.documents = await asyncio.to_thread(reader.load_data, branch=branch)
            self.index = VectorStoreIndex.from_documents(self.documents)
        finally:
            loop.close()

        print(self.documents)
        print(self.index)

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        print(messages)
        print(user_message)

        query_engine = self.index.as_query_engine(streaming=True)
        response = query_engine.query(user_message)

        return response.response_gen

```

---

```python
# examples/pipelines/rag/text_to_sql_pipeline.py
"""
title: Llama Index DB Pipeline
author: 0xThresh
date: 2024-08-11
version: 1.1
license: MIT
description: A pipeline for using text-to-SQL for retrieving relevant information from a database using the Llama Index library.
requirements: llama_index, sqlalchemy, psycopg2-binary
"""

from typing import List, Union, Generator, Iterator
import os 
from pydantic import BaseModel
from llama_index.llms.ollama import Ollama
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core import SQLDatabase, PromptTemplate
from sqlalchemy import create_engine


class Pipeline:
    class Valves(BaseModel):
        DB_HOST: str
        DB_PORT: str
        DB_USER: str
        DB_PASSWORD: str        
        DB_DATABASE: str
        DB_TABLE: str
        OLLAMA_HOST: str
        TEXT_TO_SQL_MODEL: str 


    # Update valves/ environment variables based on your selected database 
    def __init__(self):
        self.name = "Database RAG Pipeline"
        self.engine = None
        self.nlsql_response = ""

        # Initialize
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],                                                           # Connect to all pipelines
                "DB_HOST": os.getenv("DB_HOST", "http://localhost"),                     # Database hostname
                "DB_PORT": os.getenv("DB_PORT", 5432),                                        # Database port 
                "DB_USER": os.getenv("DB_USER", "postgres"),                                  # User to connect to the database with
                "DB_PASSWORD": os.getenv("DB_PASSWORD", "password"),                          # Password to connect to the database with
                "DB_DATABASE": os.getenv("DB_DATABASE", "postgres"),                          # Database to select on the DB instance
                "DB_TABLE": os.getenv("DB_TABLE", "table_name"),                            # Table(s) to run queries against 
                "OLLAMA_HOST": os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434"), # Make sure to update with the URL of your Ollama host, such as http://localhost:11434 or remote server address
                "TEXT_TO_SQL_MODEL": os.getenv("TEXT_TO_SQL_MODEL", "llama3.1:latest")            # Model to use for text-to-SQL generation      
            }
        )

    def init_db_connection(self):
        # Update your DB connection string based on selected DB engine - current connection string is for Postgres
        self.engine = create_engine(f"postgresql+psycopg2://{self.valves.DB_USER}:{self.valves.DB_PASSWORD}@{self.valves.DB_HOST}:{self.valves.DB_PORT}/{self.valves.DB_DATABASE}")
        return self.engine

    async def on_startup(self):
        # This function is called when the server is started.
        self.init_db_connection()

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # Debug logging is required to see what SQL query is generated by the LlamaIndex library; enable on Pipelines server if needed

        # Create database reader for Postgres
        sql_database = SQLDatabase(self.engine, include_tables=[self.valves.DB_TABLE])

        # Set up LLM connection; uses phi3 model with 128k context limit since some queries have returned 20k+ tokens
        llm = Ollama(model=self.valves.TEXT_TO_SQL_MODEL, base_url=self.valves.OLLAMA_HOST, request_timeout=180.0, context_window=30000)

        # Set up the custom prompt used when generating SQL queries from text
        text_to_sql_prompt = """
        Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. 
        You can order the results by a relevant column to return the most interesting examples in the database.
        Unless the user specifies in the question a specific number of examples to obtain, query for at most 5 results using the LIMIT clause as per Postgres. You can order the results to return the most informative data in the database.
        Never query for all the columns from a specific table, only ask for a few relevant columns given the question.
        You should use DISTINCT statements and avoid returning duplicates wherever possible.
        Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Pay attention to which column is in which table. Also, qualify column names with the table name when needed. You are required to use the following format, each taking one line:

        Question: Question here
        SQLQuery: SQL Query to run
        SQLResult: Result of the SQLQuery
        Answer: Final answer here

        Only use tables listed below.
        {schema}

        Question: {query_str}
        SQLQuery: 
        """

        text_to_sql_template = PromptTemplate(text_to_sql_prompt)

        query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database, 
            tables=[self.valves.DB_TABLE],
            llm=llm, 
            embed_model="local", 
            text_to_sql_prompt=text_to_sql_template, 
            streaming=True
        )

        response = query_engine.query(user_message)

        return response.response_gen

```

---

```python
# examples/pipelines/integrations/applescript_pipeline.py
from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import requests


from subprocess import call


class Pipeline:
    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "applescript_pipeline"
        self.name = "AppleScript Pipeline"
        pass

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")

        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        OLLAMA_BASE_URL = "http://localhost:11434"
        MODEL = "llama3"

        if body.get("title", False):
            print("Title Generation")
            return "AppleScript Pipeline"
        else:
            if "user" in body:
                print("######################################")
                print(f'# User: {body["user"]["name"]} ({body["user"]["id"]})')
                print(f"# Message: {user_message}")
                print("######################################")

            commands = user_message.split(" ")

            if commands[0] == "volume":

                try:
                    commands[1] = int(commands[1])
                    if 0 <= commands[1] <= 100:
                        call(
                            [f"osascript -e 'set volume output volume {commands[1]}'"],
                            shell=True,
                        )
                except:
                    pass

            payload = {
                "model": MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are an agent of the AppleScript Pipeline. You have the power to control the volume of the system.",
                    },
                    {"role": "user", "content": user_message},
                ],
                "stream": body["stream"],
            }

            try:
                r = requests.post(
                    url=f"{OLLAMA_BASE_URL}/v1/chat/completions",
                    json=payload,
                    stream=True,
                )

                r.raise_for_status()

                if body["stream"]:
                    return r.iter_lines()
                else:
                    return r.json()
            except Exception as e:
                return f"Error: {e}"

```

---

```python
# examples/pipelines/integrations/wikipedia_pipeline.py
from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
from schemas import OpenAIChatMessage
import requests
import os


class Pipeline:
    class Valves(BaseModel):
        pass

    def __init__(self):
        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "wiki_pipeline"
        self.name = "Wikipedia Pipeline"

        # Initialize rate limits
        self.valves = self.Valves(**{"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "")})

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom pipelines like RAG.
        print(f"pipe:{__name__}")

        if body.get("title", False):
            print("Title Generation")
            return "Wikipedia Pipeline"
        else:
            titles = []
            for query in [user_message]:
                query = query.replace(" ", "_")

                r = requests.get(
                    f"https://en.wikipedia.org/w/api.php?action=opensearch&search={query}&limit=1&namespace=0&format=json"
                )

                response = r.json()
                titles = titles + response[1]
                print(titles)

            context = None
            if len(titles) > 0:
                r = requests.get(
                    f"https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro&explaintext&redirects=1&titles={'|'.join(titles)}"
                )
                response = r.json()
                # get extracts
                pages = response["query"]["pages"]
                for page in pages:
                    if context == None:
                        context = pages[page]["extract"] + "\n"
                    else:
                        context = context + pages[page]["extract"] + "\n"

            return context if context else "No information found"

```