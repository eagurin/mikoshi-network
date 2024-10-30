
<Info>
  **Prerequisite** You should have installed Node.js (version 18.10.0 or
  higher).
</Info>

Step 1. Install Mintlify on your OS:

<CodeGroup>

```bash npm
npm i -g mintlify
```

```bash yarn
yarn global add mintlify
```

</CodeGroup>

Step 2. Go to the docs are located (where you can find `mint.json`) and run the following command:

```bash
mintlify dev
```

The documentation website is now available at `http://localhost:3000`.

### Custom Ports

Mintlify uses port 3000 by default. You can use the `--port` flag to customize the port Mintlify runs on. For example, use this command to run in port 3333:

```bash
mintlify dev --port 3333
```

You will see an error like this if you try to run Mintlify in a port that's already taken:

```md
Error: listen EADDRINUSE: address already in use :::3000
```

## Mintlify Versions

Each CLI is linked to a specific version of Mintlify. Please update the CLI if your local website looks different than production.

<CodeGroup>

```bash npm
npm i -g mintlify@latest
```

```bash yarn
yarn global upgrade mintlify
```

</CodeGroup>

## Deployment

<Tip>
  Unlimited editors available under the [Startup
  Plan](https://mintlify.com/pricing)
</Tip>

You should see the following if the deploy successfully went through:

<Frame>
  <img src="/images/checks-passed.png" style={{ borderRadius: '0.5rem' }} />
</Frame>

## Troubleshooting

Here's how to solve some common problems when working with the CLI.

<AccordionGroup>
  <Accordion title="Mintlify is not loading">
    Update to Node v18. Run `mintlify install` and try again.
  </Accordion>
  <Accordion title="No such file or directory on Windows">
Go to the `C:/Users/Username/.mintlify/` directory and remove the `mint`
folder. Then Open the Git Bash in this location and run `git clone
https://github.com/mintlify/mint.git`.

Repeat step 3.

  </Accordion>
  <Accordion title="Getting an unknown error">
    Try navigating to the root of your device and delete the ~/.mintlify folder.
    Then run `mintlify dev` again.
  </Accordion>
</AccordionGroup>

---


## Introduction

To run R2R with default local LLM settings, execute `r2r serve --docker --config-name=local_llm`.

R2R supports RAG with local LLMs through the Ollama library. You may follow the instructions on their [official website](https://ollama.com/) to install Ollama outside of the R2R Docker.

## Preparing Local LLMs

Next, make sure that you have all the necessary LLMs installed:
```bash
# in a separate terminal
ollama pull llama3.1
ollama pull mxbai-embed-large
ollama serve
```

These commands will need to be replaced with models specific to your configuration when deploying R2R with a customized configuration.

## Configuration

R2R uses a TOML configuration file for managing settings, which you can [read about here](/documentation/configuration/introduction). For local setup, we'll use the default `local_llm` configuration. This can be customized to your needs by setting up a standalone project.

<AccordionGroup>


<Accordion icon="gear" title="Local Configuration Details">
The `local_llm` configuration file (`core/configs/local_llm.toml`) includes:

```toml
[completion]
provider = "litellm"
concurrent_request_limit = 1

  [completion.generation_config]
  model = "ollama/llama3.1"
  temperature = 0.1
  top_p = 1
  max_tokens_to_sample = 1_024
  stream = false
  add_generation_kwargs = { }

[database]
provider = "postgres"

[embedding]
provider = "ollama"
base_model = "mxbai-embed-large"
base_dimension = 1_024
batch_size = 32
add_title_as_prefix = true
concurrent_request_limit = 32

[ingestion]
excluded_parsers = [ "mp4" ]
```

This configuration uses `ollama` and the model `mxbai-embed-large` to run embeddings. We have excluded media file parsers as they are not yet supported locally.

<Note>
We are still working on adding local multimodal RAG features. Your feedback would be appreciated.
</Note>

</Accordion>

</AccordionGroup>

For more information on how to configure R2R, [visit here](/documentation/configuration/introduction).

## Summary

The above steps are all you need to get RAG up and running with local LLMs in R2R. For detailed setup and basic functionality, refer back to the [R2R Quickstart]((/documentation/quickstart/introduction). For more advanced usage and customization options, refer to the [basic configuration]((/documentation/configuration/introduction) or join the [R2R Discord community](https://discord.gg/p6KqD2kjtB).

---


## Introduction

R2R offers a flexible configuration system that allows you to customize your Retrieval-Augmented Generation (RAG) applications. This guide introduces the key concepts and methods for configuring R2R.

## Configuration Levels

R2R supports two main levels of configuration:

1. **Server Configuration**: Define default server-side settings.
2. **Runtime Configuration**: Dynamically override settings when making API calls.

## Server Configuration

The default settings for a `light` R2R installation are specified in the [`r2r.toml`](https://github.com/SciPhi-AI/R2R/blob/main/r2r.toml) file.

When doing a `full` installation the R2R CLI uses the [`full.toml`](https://github.com/SciPhi-AI/R2R/blob/main/py/core/configs/full.toml) to override some of the default light default settings with those of the added providers.

To create your own custom configuration:

1. Create a new file named `my_r2r.toml` in your project directory.
2. Add only the settings you wish to customize. For example:

```toml my_r2r.toml
[embedding]
provider = "litellm"
base_model = "text-embedding-3-small"
base_dimension = 1536

[completion]
    [completion.generation_config]
    model = "anthropic/claude-3-opus-20240229"
```

3. Launch R2R with the CLI using your custom configuration:

```bash
r2r serve --config-path=my_r2r.toml
```

R2R will use your specified settings, falling back to defaults for any unspecified options.

## Runtime Configuration

When calling endpoints, you can override server configurations on-the-fly. This allows for dynamic control over search settings, model selection, prompt customization, and more.

For example, using the Python SDK:

```python
client = R2RClient("http://localhost:7272")

response = client.rag(
    "Who was Aristotle?",
    rag_generation_config={
        "model": "anthropic/claude-3-haiku-20240307",
        "temperature": 0.7
    },
    vector_search_settings={
        "search_limit": 100,
        "use_hybrid_search": True
    }
)
```

## Next Steps

For more detailed information on configuring specific components of R2R, please refer to the following pages:

- [Postgres Configuration](/documentation/configuration/postgres)
- [LLM Configuration](/documentation/configuration/llm)
- [RAG Configuration](/documentation/configuration/rag)
- [Ingestion Configuration](/documentation/configuration/ingestion/overview)
- [Knowledge Graph Configuration](/documentation/configuration/knowledge-graph/overview)
- [Retrieval Configuration](/documentation/configuration/retrieval/overview)

---

# Mintlify Starter Kit

Click on `Use this template` to copy the Mintlify starter kit. The starter kit contains examples including

- Guide pages
- Navigation
- Customizations
- API Reference pages
- Use of popular components

### Development

Install the [Mintlify CLI](https://www.npmjs.com/package/mintlify) to preview the documentation changes locally. To install, use the following command

```
npm i -g mintlify
```

Run the following command at the root of your documentation (where mint.json is)

```
mintlify dev
```

### Publishing Changes

Install our Github App to auto propagate changes from your repo to your deployment. Changes will be deployed to production automatically after pushing to the default branch. Find the link to install on your dashboard.

#### Troubleshooting

- Mintlify dev isn't running - Run `mintlify install` it'll re-install dependencies.
- Page loads as a 404 - Make sure you are running in a folder with `mint.json`

---


import GithubButtons from '../components/GithubButtons';


![r2r](./images/r2r.png)

R2R (RAG to Riches), the Elasticsearch for RAG, bridges the gap between experimenting with and deploying state of the art Retrieval-Augmented Generation (RAG) applications. It's a complete platform that helps you quickly build and launch scalable RAG solutions. Built around a containerized [RESTful API](/api-reference/introduction), R2R offers multimodal ingestion support, hybrid search, GraphRAG, user & document management, and observability / analytics features.

## Key Features
- [**📁 Multimodal Ingestion**](/documentation/configuration/ingestion/overview): Parse `.txt`, `.pdf`, `.json`, `.png`, `.mp3`, and more.
- [**🔍 Hybrid Search**](/cookbooks/hybrid-search): Combine semantic and keyword search with reciprocal rank fusion for enhanced relevancy.
- [**🔗 Graph RAG**](/cookbooks/graphrag): Automatically extract relationships and build knowledge graphs.
- [**🗂️ App Management**](/cookbooks/user-auth): Efficiently manage documents and users with full authentication.
- [**🔭 Observability**](/cookbooks/observability): Observe and analyze your RAG engine performance.
- [**🧩 Configurable**](/documentation/configuration/introduction): Provision your application using intuitive configuration files.
- [**🖥️ Dashboard**](https://github.com/SciPhi-AI/R2R-Dashboard): An open-source React+Next.js app with optional authentication, to interact with R2R via GUI.

## Getting Started

- [Installation](/documentation/installation): Quick installation of R2R using Docker or pip
- [Quickstart](/documentation/quickstart): A quick introduction to R2R's core features
- [Setup](/documentation/configuration/introduction): Learn how to setup and configure R2R

## API & SDKs

- [SDK](/documentation/python-sdk): API reference and Python/JS SDKs for interacting with R2R
- [API](/api-reference/introduction): API reference and Python/JS SDKs for interacting with R2R
- [Configuration](/documentation/configuration): A guide on how to configure your R2R system
- [SciPhi Website](https://sciphi.ai/): Explore a managed AI solution powered by R2R.
- [Contact Us](mailto:founders@sciphi.ai): Get in touch with our team to discuss your specific needs.

## Cookbooks

- Advanced RAG Pipelines
  - [RAG Agent](/cookbooks/agent): R2R's powerful RAG agent
  - [Hybrid Search](/cookbooks/hybrid-search): Introduction to hybrid search
  - [Advanced RAG](/cookbooks/advanced-rag): Advanced RAG features

- Knowledge Graphs
  - [GraphRAG](/cookbooks/graphrag): Walkthrough of GraphRAG

- Orchestration
  - [GraphRAG](/cookbooks/orchestration): R2R event orchestration

- Auth & Admin Features
  - [Web Development](/cookbooks/web-dev): Building webapps using R2R
  - [User Auth](/cookbooks/user-auth): Authenticating users
  - [Collections](/cookbooks/collections): Document collections
  - [Analytics & Observability](/cookbooks/observability): End-to-end logging and analytics
  - [Web Application](/cookbooks/application): Connecting with the R2R Application

## Community

[Join our Discord server](https://discord.gg/p6KqD2kjtB) to get support and connect with both the R2R team and other developers in the community. Whether you're encountering issues, looking for advice on best practices, or just want to share your experiences, we're here to help.

---


## Key Terms

- **RAG (Retrieval-Augmented Generation)**: A technique that combines information retrieval and language model generation to produce more accurate and informative responses.
- **Ingestion**: The process of parsing and indexing documents into the R2R system for later retrieval and generation.
- **Fragment**: A chunk of text extracted from an ingested document, used for similarity search and context in RAG.
- **Hybrid Search**: A search method that combines semantic vector search with traditional keyword search for improved relevancy.
- **Knowledge Graph**: A structured representation of entities and their relationships, used for advanced querying and reasoning.
- **RAG Agent**: An interactive and intelligent query interface that can formulate its own questions, search for information, and provide informed responses based on retrieved context.
- **Collection**: A grouping of documents and users for efficient access control and organization.

## Environment Settings

- `OPENAI_API_KEY`: API key for OpenAI's language models and embeddings.
- `ANTHROPIC_API_KEY`: API key for Anthropic's language models and embeddings.
- `R2R_POSTGRES_USER`: Username for the Postgres database.
- `R2R_POSTGRES_PASSWORD`: Password for the Postgres database.
- `R2R_POSTGRES_HOST`: Hostname or IP address of the Postgres database server.
- `R2R_POSTGRES_PORT`: Port number for the Postgres database server.
- `R2R_POSTGRES_DBNAME`: Name of the Postgres database to use for R2R.
- `R2R_PROJECT_NAME`: Defines the tables within the Postgres database where the selected R2R project resides.
- `R2R_PORT`: Defines the port over which the R2R process is served.
- `R2R_HOST`: Defines the host address over which the R2R process is served.
- `HATCHET_CLIENT_TOKEN`: API token for Hatchet orchestration service (required for full R2R installation).
- `UNSTRUCTURED_API_KEY`: API key for Unstructured.io document parsing service (required for full R2R installation).

For more information on these terms and settings, please refer to the relevant sections of the documentation:

- [Installation](/documentation/installation)
- [Configuration](/documentation/configuration)
- [Ingestion](/documentation/configuration/ingestion/overview)
- [Search](/cookbooks/hybrid-search)
- [Knowledge Graph](/cookbooks/graphrag)
- [RAG Agent](/cookbooks/agent)
- [Collections](/cookbooks/collections)

---


R2R uses language models to generate responses based on retrieved context. You can configure R2R's server-side LLM generation settings with the [`r2r.toml`](https://github.com/SciPhi-AI/R2R/blob/main/py/r2r.toml):

```toml r2r.toml
[completion]
provider = "litellm"
concurrent_request_limit = 16

    [completion.generation_config]
    model = "openai/gpt-4o"
    temperature = 0.1
    top_p = 1
    max_tokens_to_sample = 1_024
    stream = false
    add_generation_kwargs = {}
```

Key generation configuration options:

- `provider`: The LLM provider (defaults to "LiteLLM" for maximum flexibility).
- `concurrent_request_limit`: Maximum number of concurrent LLM requests.
- `model`: The language model to use for generation.
- `temperature`: Controls the randomness of the output (0.0 to 1.0).
- `top_p`: Nucleus sampling parameter (0.0 to 1.0).
- `max_tokens_to_sample`: Maximum number of tokens to generate.
- `stream`: Enable/disable streaming of generated text.
- `api_base`: The base URL for remote communication, e.g. `https://api.openai.com/v1`

#### Serving select LLM providers


<Tabs>
    <Tab title="OpenAI">
        ```python
        export OPENAI_API_KEY=your_openai_key
        # .. set other environment variables

        # Optional - Update default model
        # Set '"model": "openai/gpt-4o-mini"' in `r2r.toml`
        # then call `r2r serve --config-path=r2r.toml`
        r2r serve
        ```
        Supported models include:
        - openai/gpt-4o
        - openai/gpt-4-turbo
        - openai/gpt-4
        - openai/gpt-4o-mini

        For a complete list of supported OpenAI models and detailed usage instructions, please refer to the [LiteLLM OpenAI documentation](https://docs.litellm.ai/docs/providers/openai).
    </Tab>
    <Tab title="Azure">
        ```python
        export AZURE_API_KEY=your_azure_api_key
        export AZURE_API_BASE=your_azure_api_base
        export AZURE_API_VERSION=your_azure_api_version
        # .. set other environment variables

        # Optional - Update default model
        # Set '"model": "azure/<your deployment name>"' in `r2r.toml`
        r2r serve --config-path=my_r2r.toml
        ```
        Supported models include:
        - azure/gpt-4o
        - azure/gpt-4-turbo
        - azure/gpt-4
        - azure/gpt-4o-mini
        - azure/gpt-4o-mini
        For a complete list of supported Azure models and detailed usage instructions, please refer to the [LiteLLM Azure documentation](https://docs.litellm.ai/docs/providers/azure).

    </Tab>

    <Tab title="Anthropic">

        ```python
        export ANTHROPIC_API_KEY=your_anthropic_key
        # export ANTHROPIC_API_BASE=your_anthropic_base_url
        # .. set other environment variables

        # Optional - Update default model
        # Set '"model": "anthropic/claude-3-opus-20240229"' in `r2r.toml`
        r2r serve --config-path=my_r2r.toml
        ```
        Supported models include:

        - anthropic/claude-3-5-sonnet-20240620
        - anthropic/claude-3-opus-20240229
        - anthropic/claude-3-sonnet-20240229
        - anthropic/claude-3-haiku-20240307
        - anthropic/claude-2.1

        For a complete list of supported Anthropic models and detailed usage instructions, please refer to the [LiteLLM Anthropic documentation](https://docs.litellm.ai/docs/providers/anthropic).
    </Tab>


  <Tab title="Vertex AI">
    ```python
    export GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json
    export VERTEX_PROJECT=your_project_id
    export VERTEX_LOCATION=your_project_location
    # .. set other environment variables

    # Optional - Update default model
    # Set '"model": "vertex_ai/gemini-pro"' in `r2r.toml`
    r2r serve --config-path=my_r2r.toml
    ```

    Supported models include:
    - vertex_ai/gemini-pro
    - vertex_ai/gemini-pro-vision
    - vertex_ai/claude-3-opus@20240229
    - vertex_ai/claude-3-sonnet@20240229
    - vertex_ai/mistral-large@2407

    For a complete list of supported Vertex AI models and detailed usage instructions, please refer to the [LiteLLM Vertex AI documentation](https://docs.litellm.ai/docs/providers/vertex).

    <Note> Vertex AI requires additional setup for authentication and project configuration. Refer to the documentation for detailed instructions on setting up service accounts and configuring your environment. </Note>
    </Tab>

    <Tab title="AWS Bedrock">
        ```python
        export AWS_ACCESS_KEY_ID=your_access_key
        export AWS_SECRET_ACCESS_KEY=your_secret_key
        export AWS_REGION_NAME=your_region_name
        # .. set other environment variables

        # Optional - Update default model
        # Set '"model": "bedrock/anthropic.claude-v2"' in `r2r.toml`
        r2r serve --config-path=my_r2r.toml
        ```

        Supported models include:
        - bedrock/anthropic.claude-3-sonnet-20240229-v1:0
        - bedrock/anthropic.claude-v2
        - bedrock/anthropic.claude-instant-v1
        - bedrock/amazon.titan-text-express-v1
        - bedrock/meta.llama2-70b-chat-v1
        - bedrock/mistral.mixtral-8x7b-instruct-v0:1

        For a complete list of supported AWS Bedrock models and detailed usage instructions, please refer to the [LiteLLM AWS Bedrock documentation](https://docs.litellm.ai/docs/providers/bedrock).

        <Note> AWS Bedrock requires boto3 to be installed (`pip install boto3>=1.28.57`). Make sure to set up your AWS credentials properly before using Bedrock models. </Note>

    </Tab>
  <Tab title="Groq">
    ```python
    export GROQ_API_KEY=your_groq_api_key
    # .. set other environment variables

    # Optional - Update default model
    # Set '"model": "groq/llama3-8b-8192"' in `r2r.toml`
    r2r serve --config-path=my_r2r.toml
    ```

    Supported models include:
    - llama-3.1-8b-instant
    - llama-3.1-70b-versatile
    - llama-3.1-405b-reasoning
    - llama3-8b-8192
    - llama3-70b-8192
    - mixtral-8x7b-32768
    - gemma-7b-it

    For a complete list of supported Groq models and detailed usage instructions, please refer to the [LiteLLM Groq documentation](https://docs.litellm.ai/docs/providers/groq).

    Note: Groq supports ALL models available on their platform. Use the prefix `groq/` when specifying the model name.

    Additional features:
    - Supports streaming responses
    - Function/Tool calling available for compatible models
    - Speech-to-Text capabilities with Whisper model
  </Tab>

  <Tab title="Ollama">
    ```python
    # Ensure your Ollama server is running
    # Default Ollama server address: http://localhost:11434
    # <-- OR -->
    # Use `r2r --config-name=local_llm serve --docker`
    # which bundles ollama with R2R in Docker by default!

    # Optional - Update default model
    # Copy `r2r/examples/configs/local_llm.toml` into `my_r2r_local_llm.toml`
    # Set '"model": "ollama/llama3.1"' in `my_r2r_local_llm.toml`
    # then call `r2r --config-path=my_r2r_local_llm.toml`
    ```

    Supported models include:
    - llama2
    - mistral
    - mistral-7B-Instruct-v0.1
    - mixtral-8x7B-Instruct-v0.1
    - codellama
    - llava (vision model)

    For a complete list of supported Ollama models and detailed usage instructions, please refer to the [LiteLLM Ollama documentation](https://docs.litellm.ai/docs/providers/ollama).

    <Note>Ollama supports local deployment of various open-source models. Ensure you have the desired model pulled and running on your Ollama server. [See here](/documentation/local-rag) for more detailed instructions on local RAG setup.</Note>

  </Tab>

    <Tab title="Cohere">
    ```python
    export COHERE_API_KEY=your_cohere_api_key
    # .. set other environment variables

    # Optional - Update default model
    # Set '"model": "command-r"' in `r2r.toml`
    r2r serve --config-path=my_r2r.toml
    ```

    Supported models include:
    - command-r
    - command-light
    - command-r-plus
    - command-medium

    For a complete list of supported Cohere models and detailed usage instructions, please refer to the [LiteLLM Cohere documentation](https://docs.litellm.ai/docs/providers/cohere).

    </Tab>

<Tab title="Anyscale">
    ```python
    export ANYSCALE_API_KEY=your_anyscale_api_key
    # .. set other environment variables

    # Optional - Update default model
    # Set '"model": "anyscale/mistralai/Mistral-7B-Instruct-v0.1"' in `r2r.toml`
    r2r serve --config-path=my_r2r.toml
    ```

    Supported models include:
    - anyscale/meta-llama/Llama-2-7b-chat-hf
    - anyscale/meta-llama/Llama-2-13b-chat-hf
    - anyscale/meta-llama/Llama-2-70b-chat-hf
    - anyscale/mistralai/Mistral-7B-Instruct-v0.1
    - anyscale/codellama/CodeLlama-34b-Instruct-hf

    For a complete list of supported Anyscale models and detailed usage instructions, please refer to the [Anyscale Endpoints documentation](https://app.endpoints.anyscale.com/).


    <Note>Anyscale supports a wide range of models, including Llama 2, Mistral, and CodeLlama variants. Check the Anyscale Endpoints documentation for the most up-to-date list of available models.</Note>
  </Tab>


</Tabs>



### Runtime Configuration of LLM Provider

R2R supports runtime configuration of the LLM provider, allowing you to dynamically change the model or provider for each request. This flexibility enables you to use different models or providers based on specific requirements or use cases.

### Combining Search and Generation

When performing a RAG query, you can dynamically set the LLM generation settings:

```python
response = client.rag(
    "What are the latest advancements in quantum computing?",
    rag_generation_config={
        "stream": False,
        "model": "openai/gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 150
    }
)
```

For more detailed information on configuring other search and RAG settings, please refer to the [RAG Configuration documentation](/documentation/configuration/rag).


## Next Steps

For more detailed information on configuring specific components of R2R, please refer to the following pages:

- [Postgres Configuration](/documentation/configuration/postgres)
- [RAG Configuration](/documentation/configuration/rag)
- [Ingestion Configuration](/documentation/configuration/ingestion/overview)
- [Knowledge Graph Configuration](/documentation/configuration/knowledge-graph/overview)
- [Retrieval Configuration](/documentation/configuration/retrieval/overview)

---


This basic quickstart shows how to:

1. Ingest files into your R2R system
2. Search over ingested files
3. Request or stream a RAG (Retrieval-Augmented Generation) response
4. Use the RAG Agent for more complex, interactive queries

Be sure to complete the [installation instructions](/documentation/installation) before continuing with this guide. If you prefer to dive straight into the API details, select a choice from below:

<CardGroup cols={3}>
  <Card title="API Reference" icon="message-code" href="/api-reference/introduction" />
  <Card title="Python SDK" icon="python" href="/documentation/python-sdk" />
  <Card title="Javascript SDK" icon="js" href="/documentation/js-sdk" />
</CardGroup>


## Getting started

Start by checking that you have correctly deployed your R2R instance locally:

```bash
curl http://localhost:7272/v2/health
# {"results":{"response":"ok"}}
```

<Note>
SciPhi offers managed enterprise solutions for R2R. If you're interested in a fully managed, scalable deployment of R2R for your organization, please contact their team at founders@sciphi.ai for more information on enterprise offerings.
</Note>


## Ingesting file(s) and directories

The remainder of this quickstart will proceed with CLI commands, but all of these commands are easily reproduced inside of the Javascript or Python SDK.

Ingest your selected files or directories:

```bash
r2r ingest-files --file-paths /path/to/your_file_1 /path/to/your_dir_1 ...
```

**For testing**: Use the sample file(s) included inside the R2R project:

```bash
r2r ingest-sample-file
# or r2r ingest-sample-files for multi-ingestion
```

Example output:
```plaintext
[{'message': 'Ingestion task queued successfully.', 'task_id': '2b16bb55-4f47-4e66-a6bd-da9e215b9793', 'document_id': '9fbe403b-c11c-5aae-8ade-ef22980c3ad1'}]
```

<Info>When no document ID(s) are provided to the ingest_files endpoint, a unique document ID is automatically generated for each ingested document from the input filepath and user id.</Info>

After successful ingestion, the documents overview endpoint will return output like so:
```bash
r2r documents-overview
```

Example output:
```plaintext
{
    'id': '9fbe403b-c11c-5aae-8ade-ef22980c3ad1',
    'title': 'aristotle.txt',
    'user_id': '2acb499e-8428-543b-bd85-0d9098718220',
    ...
    'ingestion_status': 'parsing',
    ...
}
... within 10s ...
{
    'id': '9fbe403b-c11c-5aae-8ade-ef22980c3ad1',
    ...
    'ingestion_status': 'success',
    ...
}
```

Ingestion is complete when all documents are in a `success` or `failed` state.
## Executing a search

Perform a search query:

```bash
r2r search --query="who was aristotle?"
```

The search query will use basic similarity search to find the most relevant documents. You can use advanced search methods like [hybrid search](/cookbooks/hybrid-search) or [knowledge graph search](/cookbooks/graphrag) depending on your use case.

Example output:
```plaintext
{'results':
    {'vector_search_results': [
        {
            'fragment_id': '34c32587-e2c9-529f-b0a7-884e9a3c3b2e',
            'extraction_id': '8edf5123-0a5c-568c-bf97-654b6adaf8dc',
            'document_id': '9fbe403b-c11c-5aae-8ade-ef22980c3ad1',
            'user_id': '2acb499e-8428-543b-bd85-0d9098718220',
            'collection_ids': [],
            'score': 0.780314067545999,
            'text': 'Aristotle[A] (Greek: Ἀριστοτέλης Aristotélēs, pronounced [aristotélɛːs]; 384–322 BC) was an Ancient Greek philosopher and polymath. His writings cover a broad range of subjects spanning the natural sciences, philosophy, linguistics, economics, politics, psychology, and the arts. As the founder of the Peripatetic school of philosophy in the Lyceum in Athens, he began the wider Aristotelian tradition that followed, which set the groundwork for the development of modern science.',
            'metadata': {
                'title': 'aristotle.txt',
                'version': 'v0',
                'chunk_order': 0,
                ...
```

## RAG Response

Generate a RAG response:

```bash
r2r rag --query="who was aristotle?" --use-hybrid-search
```

Example output:
```plaintext
Search Results:
{'vector_search_results': ... }
Completion:
{'results': [
    {
        'id': 'chatcmpl-9eXL6sKWlUkP3f6QBnXvEiKkWKBK4',
        'choices': [
            {
                'finish_reason': 'stop',
                'index': 0,
                'logprobs': None,
                'message': {
                    'content': "Aristotle (384–322 BC) was an Ancient Greek philosopher and polymath whose writings covered a broad range of subjects including the natural sciences,
                    ...
```

## Stream a RAG Response

Stream a RAG response:

```bash
r2r rag --query="who was aristotle?" --stream --use-hybrid-search
```

Example output (streamed):
```plaintext
<search>"{\"fragment_id\":\"34c32587-e2c9-52.....}"</search>
<completion>Aristotle (384–322 BC) was an Ancient Greek philosopher ... </completion>
```

## Using the RAG Agent

The RAG Agent provides a more interactive and intelligent way to query your knowledge base. It can formulate its own questions, search for information, and provide informed responses based on the retrieved context.

### Basic RAG Agent Usage

Here's how to use the RAG Agent for a simple query:

```python
from r2r import R2RClient

client = R2RClient("http://localhost:7272")
# when using auth, do client.login(...)

messages = [
    {"role": "user", "content": "What was Aristotle's main contribution to philosophy?"},
    {"role": "assistant", "content": "Aristotle made numerous significant contributions to philosophy, but one of his main contributions was in the field of logic and reasoning. He developed a system of formal logic, which is considered the first comprehensive system of its kind in Western philosophy. This system, often referred to as Aristotelian logic or term logic, provided a framework for deductive reasoning and laid the groundwork for scientific thinking."},
    {"role": "user", "content": "Can you elaborate on how this influenced later thinkers?"}
]

result = client.agent(
    messages=messages,
    vector_search_settings={"use_hybrid_search":True},
    rag_generation_config={"model": "openai/gpt-4o", "temperature": 0.7}
)
print(result)
```

## Additional Features

R2R offers additional features to enhance your document management and user experience:

### User Authentication

R2R provides a complete set of user authentication and management features, allowing you to implement secure and feature-rich authentication systems or integrate with your preferred authentication provider.

<CardGroup cols={2}>
  <Card title="User Auth Cookbook" icon="key" href="/cookbooks/user-auth">
    Learn how to implement user registration, login, email verification, and more using R2R's built-in authentication capabilities.
  </Card>
  <Card title="Auth Providers" icon="user-shield" href="/documentation/deep-dive/providers/auth">
    Explore the available authentication provider options in R2R and how to integrate with your preferred provider.
  </Card>
</CardGroup>

### Collections

Collections in R2R enable efficient access control and organization of users and documents. With collections, you can manage permissions and access at a group level.

<CardGroup cols={2}>
  <Card title="Collections Cookbook" icon="database" href="/cookbooks/collections">
    Discover how to create, manage, and utilize collections in R2R for granular access control and document organization.
  </Card>
  <Card title="Collection Permissions" icon="user-lock" href="/cookbooks/collections#security-considerations">
    Learn about best practices for implementing collection permissions and customizing access control in your R2R application.
  </Card>
</CardGroup>

## Next Steps

Now that you have a basic understanding of R2R's core features, you can explore more advanced topics:

- Dive deeper into [document ingestion](/documentation/python-sdk/ingestion) and customization options.
- Learn about [search and RAG](/documentation/python-sdk/retrieval) inside R2R.
- Implement [user authentication](/cookbooks/user-auth) to secure your application.
- Organize your documents using [collections](/cookbooks/collections) for granular access control.

If you have any questions or need further assistance, please refer to the [R2R documentation](/) or reach out to our support team.

---


## Prompt Management in R2R

R2R provides a flexible system for managing prompts, allowing you to create, update, retrieve, and delete prompts dynamically. This system is crucial for customizing the behavior of language models and ensuring consistent interactions across your application.

## Default Prompts

R2R comes with a set of default prompts that are loaded from YAML files located in the [`py/core/providers/prompts/defaults`](https://github.com/SciPhi-AI/R2R/tree/main/py/core/providers/prompts/defaults) directory. These default prompts provide a starting point for various tasks within the R2R system.

For example, the default RAG (Retrieval-Augmented Generation) prompt is defined as follows:

```yaml
default_rag:
  template: >
    ## Task:

    Answer the query given immediately below given the context which follows later. Use line item references to like [1], [2], ... refer to specifically numbered items in the provided context. Pay close attention to the title of each given source to ensure it is consistent with the query.


    ### Query:

    {query}


    ### Context:

    {context}


    ### Query:

    {query}


    REMINDER - Use line item references to like [1], [2], ... refer to specifically numbered items in the provided context.

    ## Response:
  input_types:
    query: str
    context: str
```

### Default Prompt Usage

Here's a table showing the purpose of some key default prompts:

Certainly! I'll create an expanded table that explains all the prompts you've listed, with links to their respective GitHub files. Here's the updated table:

| Prompt File | Purpose |
|-------------|---------|
| [`default_rag.yaml`](https://github.com/SciPhi-AI/R2R/blob/main/py/core/providers/database/prompts/default_rag.yaml) | Default prompt for Retrieval-Augmented Generation (RAG) tasks. It instructs the model to answer queries based on provided context, using line item references. |
| [`graphrag_community_reports.yaml`](https://github.com/SciPhi-AI/R2R/blob/main/py/core/providers/database/prompts/graphrag_community_reports.yaml) | Used in GraphRAG to generate reports about communities or clusters in the knowledge graph. |
| [`graphrag_entity_description.yaml.yaml`](https://github.com/SciPhi-AI/R2R/blob/main/py/core/providers/database/prompts/graphrag_entity_description.yaml) | System prompt for the "map" phase in GraphRAG, used to process individual nodes or edges. |
| [`graphrag_map_system.yaml`](https://github.com/SciPhi-AI/R2R/blob/main/py/core/providers/database/prompts/graphrag_map_system.yaml) | System prompt for the "map" phase in GraphRAG, used to process individual nodes or edges. |
| [`graphrag_reduce_system.yaml`](https://github.com/SciPhi-AI/R2R/blob/main/py/core/providers/database/prompts/graphrag_reduce_system.yaml) | System prompt for the "reduce" phase in GraphRAG, used to combine or summarize information from multiple sources. |
| [`graphrag_triples_extraction_few_shot.yaml`](https://github.com/SciPhi-AI/R2R/blob/main/py/core/providers/database/prompts/graphrag_triples_extraction_few_shot.yaml) | Few-shot prompt for extracting subject-predicate-object triplets in GraphRAG, with examples. |
| [`hyde.yaml`](https://github.com/SciPhi-AI/R2R/blob/main/py/core/providers/database/prompts/hyde.yaml) | Related to Hypothetical Document Embeddings (HyDE) for improving retrieval performance. |
| [`kg_search.yaml`](https://github.com/SciPhi-AI/R2R/blob/main/py/core/providers/database/prompts/kg_search.yaml) | Used for searching the knowledge graph, possibly to formulate queries or interpret results. |
| [`kg_search_with_spec.yaml`](https://github.com/SciPhi-AI/R2R/blob/main/py/core/providers/database/prompts/kg_search_with_spec.yaml) | Similar to `kg_search.yaml`, but with a specific schema or specification for the search process. |
| [`rag_agent.yaml`](https://github.com/SciPhi-AI/R2R/blob/main/py/core/providers/database/prompts/rag_agent.yaml) | Defines the behavior and instructions for the RAG agent, which coordinates the retrieval and generation process. |
| [`rag_context.yaml`](https://github.com/SciPhi-AI/R2R/blob/main/py/core/providers/database/prompts/rag_context.yaml) | Used to process or format the context retrieved for RAG tasks. |
| [`rag_fusion.yaml`](https://github.com/SciPhi-AI/R2R/blob/main/py/core/providers/database/prompts/rag_fusion.yaml) | Used in RAG fusion techniques, possibly for combining information from multiple retrieved passages. |
| [`system.yaml`](https://github.com/SciPhi-AI/R2R/blob/main/py/core/providers/database/prompts/system.yaml) | Contains general system-level prompts or instructions for the R2R system. |


You can find the full list of default prompts and their contents in the [defaults directory](https://github.com/SciPhi-AI/R2R/tree/main/py/core/providers/prompts/defaults).

## Prompt Provider

R2R uses a postgres class to manage prompts. This allows for storage, retrieval, and manipulation of prompts, leveraging both a PostgreSQL database and YAML files for flexibility and persistence.

Key features of prompts inside R2R:

1. **Database Storage**: Prompts are stored in a PostgreSQL table, allowing for efficient querying and updates.
2. **YAML File Support**: Prompts can be loaded from YAML files, providing an easy way to version control and distribute default prompts.
3. **In-Memory Cache**: Prompts are kept in memory for fast access during runtime.

## Prompt Structure

Each prompt in R2R consists of:

- **Name**: A unique identifier for the prompt.
- **Template**: The actual text of the prompt, which may include placeholders for dynamic content.
- **Input Types**: A dictionary specifying the expected types for any dynamic inputs to the prompt.

## Managing Prompts

R2R provides several endpoints and SDK methods for managing prompts:

### Adding a Prompt

To add a new prompt:

```python
from r2r import R2RClient

client = R2RClient()

response = client.add_prompt(
    name="my_new_prompt",
    template="Hello, {name}! Welcome to {service}.",
    input_types={"name": "str", "service": "str"}
)
```

### Updating a Prompt

To update an existing prompt:

```python
response = client.update_prompt(
    name="my_existing_prompt",
    template="Updated template: {variable}",
    input_types={"variable": "str"}
)
```

### Retrieving a Prompt

To get a specific prompt:

```python
response = client.get_prompt(
    prompt_name="my_prompt",
    inputs={"variable": "example"},
    prompt_override="Optional override text"
)
```

### Listing All Prompts

To retrieve all prompts:

```python
response = client.get_all_prompts()
```

### Deleting a Prompt

To delete a prompt:

```python
response = client.delete_prompt("prompt_to_delete")
```

## Security Considerations

Access to prompt management functions is restricted to superusers to prevent unauthorized modifications to system prompts. Ensure that only trusted administrators have superuser access to your R2R deployment.

## Best Practices

1. **Version Control**: Store your prompts in version-controlled YAML files for easy tracking of changes and rollbacks.
2. **Consistent Naming**: Use a consistent naming convention for your prompts to make them easy to identify and manage.
3. **Input Validation**: Always specify input types for your prompts to ensure that they receive the correct data types.
4. **Regular Audits**: Periodically review and update your prompts to ensure they remain relevant and effective.
5. **Testing**: Test prompts thoroughly before deploying them to production, especially if they involve complex logic or multiple input variables.

## Advanced Usage

### Dynamic Prompt Loading

R2R's prompt system allows for dynamic loading of prompts from both the database and YAML files. This enables you to:

1. Deploy default prompts with your application.
2. Override or extend these prompts at runtime.
3. Easily update prompts without redeploying your entire application.

### Prompt Templating

The prompt template system in R2R supports complex string formatting. You can include conditional logic, loops, and other Python expressions within your prompts using a templating engine.

Example of a more complex prompt template:

```python
complex_template = """
Given the following information:
{% for item in data %}
- {{ item.name }}: {{ item.value }}
{% endfor %}

Please provide a summary that {% if include_analysis %}includes an analysis of the data{% else %}lists the key points{% endif %}.
"""

client.add_prompt(
    name="complex_summary",
    template=complex_template,
    input_types={"data": "list", "include_analysis": "bool"}
)
```

This flexibility allows you to create highly dynamic and context-aware prompts that can adapt to various scenarios in your application.

## Conclusion

R2R's prompt management system provides a powerful and flexible way to control the behavior of language models in your application. By leveraging this system effectively, you can create more dynamic, context-aware, and maintainable AI-powered features.

For more detailed information on other aspects of R2R configuration, please refer to the following pages:

- [LLM Configuration](/documentation/configuration/llm)
- [RAG Configuration](/documentation/configuration/rag)
- [Postgres Configuration](/documentation/configuration/postgres)
- [Ingestion Configuration](/documentation/configuration/ingestion/overview)

---


## RAG Customization

RAG (Retrieval-Augmented Generation) in R2R can be extensively customized to suit various use cases. The main components for customization are:

1. **Generation Configuration**: Control the language model's behavior.
2. **Search Settings**: Fine-tune the retrieval process.
3. **Task Prompt Override**: Customize the system prompt for specific tasks.


### LLM Provider Configuration

Refer to the LLM configuration [page here](/documentation/configuration/llm).


### Retrieval Configuration

Refer to the retrieval configuration [page here](/documentation/configuration/retrieval/overview).


### Combining LLM and Retrieval Configuration for RAG



The `rag_generation_config` parameter allows you to customize the language model's behavior. Default settings are set on the server-side using the `r2r.toml`, as described in in previous configuraiton guides. These settings can be overridden at runtime as shown below:

```python
# Configure vector search
vector_search_settings = {
    "use_vector_search": True,
    "search_limit": 20,
    "use_hybrid_search": True,
    "selected_collection_ids": ["c3291abf-8a4e-5d9d-80fd-232ef6fd8526"]
}

# Configure graphRAG search
kg_search_settings = {
    "use_kg_search": True,
    "kg_search_type": "local",
    "kg_search_level": None,
    "generation_config": {
        "model": "gpt-4",
        "temperature": 0.1
    },
    "entity_types": ["Person", "Organization"],
    "relationships": ["worksFor", "foundedBy"],
    "max_community_description_length": 65536,
    "max_llm_queries_for_global_search": 250,
    "local_search_limits": {"__Entity__": 20, "__Relationship__": 20, "__Community__": 20}
}

# Configure LLM generation
rag_generation_config = {
    "model": "anthropic/claude-3-opus-20240229",
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens_to_sample": 1500,
    "stream": True,
    "functions": None,  # For function calling, if supported
    "tools": None,  # For tool use, if supported
    "add_generation_kwargs": {},  # Additional provider-specific parameters
    "api_base": None  # Custom API endpoint, if needed
}
```

When performing a RAG query you can combine these vector search, knowledge graph search, and generation settings at runtime:

```python
from r2r import R2RClient

client = R2RClient()

response = client.rag(
    "What are the latest advancements in quantum computing?",
    rag_generation_config=rag_generation_config
    vector_search_settings=vector_search_settings,
    kg_search_settings=kg_search_settings,
)
```

R2R defaults to the specified server-side settings when no runtime overrides are specified.
### RAG Prompt Override

For specialized tasks, you can override the default RAG task prompt at runtime:

```python
task_prompt_override = """You are an AI assistant specializing in quantum computing.
Your task is to provide a concise summary of the latest advancements in the field,
focusing on practical applications and breakthroughs from the past year."""

response = client.rag(
    "What are the latest advancements in quantum computing?",
    rag_generation_config=rag_generation_config,
    task_prompt_override=task_prompt_override
)
```

This prompt can also be set statically on as part of the server configuration process.

## Agent-based Interaction

R2R supports multi-turn conversations and complex query processing through its agent endpoint:

```python
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "What are the key differences between quantum and classical computing?"}
]

response = client.agent(
    messages=messages,
    vector_search_settings=vector_search_settings,
    kg_search_settings=kg_search_settings,
    rag_generation_config=rag_generation_config,
)
```

The agent can break down complex queries into sub-tasks, leveraging both retrieval and generation capabilities to provide comprehensive responses. The settings specified in the example above will propagate to the agent and it's tools.

By leveraging these configuration options, you can fine-tune R2R's retrieval and generation process to best suit your specific use case and requirements.



## Next Steps

For more detailed information on configuring specific components of R2R, please refer to the following pages:

- [Postgres Configuration](/documentation/configuration/postgres)
- [LLM Configuration](/documentation/configuration/llm)
- [Ingestion Configuration](/documentation/configuration/ingestion/overview)
- [Knowledge Graph Configuration](/documentation/configuration/knowledge-graph/overview)
- [Retrieval Configuration](/documentation/configuration/retrieval/overview)

---

One of the core principles of software development is DRY (Don't Repeat
Yourself). This is a principle that apply to documentation as
well. If you find yourself repeating the same content in multiple places, you
should consider creating a custom snippet to keep your content in sync.

---


It is often effective to restructure data after ingestion to improve retrieval performance and accuracy. R2R supports knowledge graphs for data restructuring. You can find out more about creating knowledge graphs in the [Knowledge Graphs Guide](/cookbooks/graphrag).

You can configure knowledge graph enrichment in the R2R configuration file. To do this, just set the `kg.kg_enrichment_settings` section in the configuration file. Following is the sample format from the example configuration file `r2r.toml`.

```toml
[database]
provider = "postgres"
batch_size = 256

  [database.kg_creation_settings]
    kg_triples_extraction_prompt = "graphrag_triples_extraction_few_shot"
    entity_types = [] # if empty, all entities are extracted
    relation_types = [] # if empty, all relations are extracted
    fragment_merge_count = 4 # number of fragments to merge into a single extraction
    max_knowledge_triples = 100 # max number of triples to extract for each document chunk
    generation_config = { model = "openai/gpt-4o-mini" } # and other generation params

  [database.kg_enrichment_settings]
    max_description_input_length = 65536 # increase if you want more comprehensive descriptions
    max_summary_input_length = 65536
    generation_config = { model = "openai/gpt-4o-mini" } # and other generation params
    leiden_params = {} # more params in graspologic/partition/leiden.py

  [database.kg_search_settings]
    generation_config = { model = "openai/gpt-4o-mini" }
```

Next you can do GraphRAG with the knowledge graph. Find out more about GraphRAG in the [GraphRAG Guide](/cookbooks/graphrag).

---


## Embedding Provider

By default, R2R uses the LiteLLM framework to communicate with various cloud embedding providers. To customize the embedding settings:

```toml r2r.toml
[embedding]
provider = "litellm"
base_model = "openai/text-embedding-3-small"
base_dimension = 512
batch_size = 128
add_title_as_prefix = false
rerank_model = "None"
concurrent_request_limit = 256
```

Let's break down the embedding configuration options:

- `provider`: Choose from `ollama`, `litellm` and `openai`. R2R defaults to using the LiteLLM framework for maximum embedding provider flexibility.
- `base_model`: Specifies the embedding model to use. Format is typically "provider/model-name" (e.g., `"openai/text-embedding-3-small"`).
- `base_dimension`: Sets the dimension of the embedding vectors. Should match the output dimension of the chosen model.
- `batch_size`: Determines the number of texts to embed in a single API call. Larger values can improve throughput but may increase latency.
- `add_title_as_prefix`: When true, prepends the document title to the text before embedding, providing additional context.
- `rerank_model`: Specifies a model for reranking results. Set to "None" to disable reranking (note: not supported by LiteLLMEmbeddingProvider).
- `concurrent_request_limit`: Sets the maximum number of concurrent embedding requests to manage load and avoid rate limiting.

<Note> Embedding providers for an R2R system cannot be configured at runtime and are instead configured server side. </Note>


### Supported  LiteLLM Providers

Support for any of the embedding providers listed below is provided through LiteLLM.
<Tabs>
  <Tab title="OpenAI">
    Example configuration:
    ```toml example r2r.toml
    provider = "litellm"
    base_model = "openai/text-embedding-3-small"
    base_dimension = 512
    ```

    ```bash
    export OPENAI_API_KEY=your_openai_key
    # .. set other environment variables

    r2r serve --config-path=r2r.toml
    ```
    Supported models include:
    - text-embedding-3-small
    - text-embedding-3-large
    - text-embedding-ada-002

    For detailed usage instructions, refer to the [LiteLLM OpenAI Embedding documentation](https://docs.litellm.ai/docs/embedding/supported_embedding#openai-embedding-models).
  </Tab>

  <Tab title="Azure">
    Example configuration:
    ```toml example r2r.toml
    provider = "litellm"
    base_model = "azure/<your deployment name>"
    base_dimension = XXX
    ```

    ```bash
    export AZURE_API_KEY=your_azure_api_key
    export AZURE_API_BASE=your_azure_api_base
    export AZURE_API_VERSION=your_azure_api_version
    # .. set other environment variables

    r2r serve --config-path=r2r.toml
    ```
    Supported models include:
    - text-embedding-ada-002

    For detailed usage instructions, refer to the [LiteLLM Azure Embedding documentation](https://docs.litellm.ai/docs/embedding/supported_embedding#azure-openai-embedding-models).
  </Tab>

  <Tab title="Anthropic">
    Anthropic does not currently offer embedding models. Consider using OpenAI or another provider for embeddings.
  </Tab>

  <Tab title="Cohere">
    Example configuration:
    ```toml example r2r.toml
    provider = "litellm"
    base_model = "cohere/embed-english-v3.0"
    base_dimension = 1_024
    ```

    ```bash
    export COHERE_API_KEY=your_cohere_api_key
    # .. set other environment variables

    r2r serve --config-path=r2r.toml
    ```

    Supported models include:
    - embed-english-v3.0
    - embed-english-light-v3.0
    - embed-multilingual-v3.0
    - embed-multilingual-light-v3.0
    - embed-english-v2.0
    - embed-english-light-v2.0
    - embed-multilingual-v2.0

    For detailed usage instructions, refer to the [LiteLLM Cohere Embedding documentation](https://docs.litellm.ai/docs/embedding/supported_embedding#cohere-embedding-models).
  </Tab>

  <Tab title="Ollama">


    When running with Ollama, additional changes are recommended for the to the `r2r.toml` file. In addition to using the `ollama` provider directly, we recommend restricting the `concurrent_request_limit` in order to avoid exceeding the throughput of your Ollama server.
    ```toml example r2r.toml
    [embedding]
    provider = "ollama"
    base_model = "ollama/mxbai-embed-large"
    base_dimension = 1_024
    batch_size = 32
    add_title_as_prefix = true
    ```


    ```bash
    # Ensure your Ollama server is running
    # Default Ollama server address: http://localhost:11434
    # <-- OR -->
    # Use `r2r --config-name=local_llm serve --docker`
    # which bundles ollama with R2R in Docker by default!

    r2r serve --config-path=r2r.toml
    ```

    Then, deploy your R2R server with `r2r serve --config-path=r2r.toml `.
  </Tab>

  <Tab title="HuggingFace">
    Example configuration:

    ```toml example r2r.toml
    [embedding]
    provider = "litellm"
    base_model = "huggingface/microsoft/codebert-base"
    base_dimension = 768
    ```

    ```python
    export HUGGINGFACE_API_KEY=your_huggingface_api_key

    r2r serve --config-path=r2r.toml
    ```
    LiteLLM supports all Feature-Extraction Embedding models on HuggingFace.

    For detailed usage instructions, refer to the [LiteLLM HuggingFace Embedding documentation](https://docs.litellm.ai/docs/embedding/supported_embedding#huggingface-embedding-models).
  </Tab>

  <Tab title="Bedrock">
    Example configuration:

    ```toml example r2r.toml
    provider = "litellm"
    base_model = "bedrock/amazon.titan-embed-text-v1"
    base_dimension = 1_024
    ```

    ```bash
    export AWS_ACCESS_KEY_ID=your_access_key
    export AWS_SECRET_ACCESS_KEY=your_secret_key
    export AWS_REGION_NAME=your_region_name
    # .. set other environment variables

    r2r serve --config-path=r2r.toml
    ```
    Supported models include:
    - amazon.titan-embed-text-v1
    - cohere.embed-english-v3
    - cohere.embed-multilingual-v3

    For detailed usage instructions, refer to the [LiteLLM Bedrock Embedding documentation](https://docs.litellm.ai/docs/embedding/supported_embedding#bedrock-embedding).
  </Tab>

  <Tab title="Vertex AI">

    Example configuration:
    ```toml example r2r.toml
    provider = "litellm"
    base_model = "vertex_ai/textembedding-gecko"
    base_dimension = 768
    ```

    ```bash
    export GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json
    export VERTEX_PROJECT=your_project_id
    export VERTEX_LOCATION=your_project_location
    # .. set other environment variables

    r2r serve --config-path=r2r.toml
    ```
    Supported models include:
    - textembedding-gecko
    - textembedding-gecko-multilingual
    - textembedding-gecko@001
    - textembedding-gecko@003
    - text-embedding-preview-0409
    - text-multilingual-embedding-preview-0409

    For detailed usage instructions, refer to the [LiteLLM Vertex AI Embedding documentation](https://docs.litellm.ai/docs/embedding/supported_embedding#vertex-ai-embedding-models).
  </Tab>

  <Tab title="Voyage AI">
  Example Configuration
    ```toml example r2r.toml
    provider = "litellm"
    base_model = "voyage/voyage-01"
    base_dimension = 1_024
    ```

    ```bash
    export VOYAGE_API_KEY=your_voyage_api_key
    # .. set other environment variables

    r2r serve --config-path=r2r.toml
    ```
    Supported models include:
    - voyage-01
    - voyage-lite-01
    - voyage-lite-01-instruct

    For detailed usage instructions, refer to the [LiteLLM Voyage AI Embedding documentation](https://docs.litellm.ai/docs/embedding/supported_embedding#voyage-ai-embedding-models).
  </Tab>
</Tabs>

---


## Postgres Database

R2R uses PostgreSQL as the sole provider for relational and vector search queries. This means that Postgres is involved in handling authentication, document management, and search across R2R. For robust search capabilities, R2R leverages the `pgvector` extension and `ts_rank` to implement [customizable hybrid search](/cookbooks/hybrid-search).

<Note>
  R2R chose Postgres as its core technology for several reasons:

  - **Versatility**: Postgres is a robust, advanced database that can handle both relational data and vector embeddings.
  - **Simplicity**: By using Postgres for both traditional data and vector search, R2R eliminates the need for complex syncing between separate databases.
  - **Familiarity**: Many developers are already comfortable with Postgres, making it easier to integrate R2R into existing workflows.
  - **Extensibility**: Postgres's rich ecosystem of extensions allows R2R to leverage advanced features and optimizations.

  Read more about [Postgres here](https://www.postgresql.org/).
</Note>

## Postgres Configuration

To customize the database settings, you can modify the `database` section in your `r2r.toml` file and set corresponding environment variables or provide the settings directly in the configuration file.

1. Edit the `database` section in your `r2r.toml` file:

```toml r2r.toml
[database]
provider = "postgres" # currently only `postgres` is supported

# optional parameters which are typically set in the environment instead:
user = "your_postgres_user"
password = "your_postgres_password"
host = "your_postgres_host"
port = "your_postgres_port"
db_name = "your_database_name"
your_project_name = "your_project_name"
```

2. Alternatively, you can set the following environment variables:

```bash
export R2R_POSTGRES_USER=your_postgres_user
export R2R_POSTGRES_PASSWORD=your_postgres_password
export R2R_POSTGRES_HOST=your_postgres_host
export R2R_POSTGRES_PORT=your_postgres_port
export R2R_POSTGRES_DBNAME=your_database_name
export R2R_PROJECT_NAME=your_project_name
```

## Advanced Postgres Features in R2R

R2R leverages several advanced PostgreSQL features to provide powerful search and retrieval capabilities:

### pgvector Extension

R2R uses the `pgvector` extension to enable efficient vector similarity search. This is crucial for semantic search operations. The `collection.py` file defines a custom `Vector` type that interfaces with `pgvector`:

```python
class Vector(UserDefinedType):
    # ... (implementation details)

    class comparator_factory(UserDefinedType.Comparator):
        def l2_distance(self, other):
            return self.op("<->", return_type=Float)(other)

        def max_inner_product(self, other):
            return self.op("<#>", return_type=Float)(other)

        def cosine_distance(self, other):
            return self.op("<=>", return_type=Float)(other)
```

This allows R2R to perform vector similarity searches using different distance measures.

### Hybrid Search

R2R implements a sophisticated hybrid search that combines full-text search and vector similarity search. This approach provides more accurate and contextually relevant results. Key components of the hybrid search include:

1. **Full-Text Search**: Utilizes PostgreSQL's built-in full-text search capabilities with `ts_rank` and `websearch_to_tsquery`.
2. **Semantic Search**: Performs vector similarity search using `pgvector`.
3. **Reciprocal Rank Fusion (RRF)**: Merges results from full-text and semantic searches.

The `collection.py` file includes methods for building complex SQL queries that implement this hybrid search approach.

### GIN Indexing

R2R uses GIN (Generalized Inverted Index) indexing to optimize full-text searches:

```python
Index(f"idx_{name}_fts", "fts", postgresql_using="gin"),
```

This indexing strategy allows for efficient full-text search and trigram similarity matching.

### JSON Support

R2R leverages PostgreSQL's JSONB type for flexible metadata storage:

```python
Column(
    "metadata",
    postgresql.JSONB,
    server_default=text("'{}'::jsonb"),
    nullable=False,
)
```

This allows for efficient storage and querying of structured metadata alongside vector embeddings.

## Performance Considerations

When setting up PostgreSQL for R2R, consider the following performance optimizations:

1. **Indexing**: Ensure proper indexing for both full-text and vector searches. R2R automatically creates necessary indexes, but you may need to optimize them based on your specific usage patterns.

2. **Hardware**: For large-scale deployments, consider using dedicated PostgreSQL instances with sufficient CPU and RAM to handle vector operations efficiently.

3. **Vacuuming**: Regular vacuuming helps maintain database performance, especially for tables with frequent updates or deletions.

4. **Partitioning**: For very large datasets, consider table partitioning to improve query performance.

By leveraging these advanced PostgreSQL features and optimizations, R2R provides a powerful and flexible foundation for building sophisticated retrieval and search systems.

---

Knowledge graph search settings can be configured both server-side and at runtime. Runtime settings are passed as a dictionary to the search and RAG endpoints. You may refer to the [search API documentation here](/api-reference/endpoint/search) for additional materials.


```python
kg_search_settings = {
    "use_kg_search": True,
    "kg_search_type": "local",
    "kg_search_level": None,
    "generation_config": {
        "model": "gpt-4",
        "temperature": 0.1
    },
    "entity_types": ["Person", "Organization"],
    "relationships": ["worksFor", "foundedBy"],
    "max_community_description_length": 65536,
    "max_llm_queries_for_global_search": 250,
    "local_search_limits": {"__Entity__": 20, "__Relationship__": 20, "__Community__": 20}
}

response = client.search("query", kg_search_settings=kg_search_settings)
```

**KGSearchSettings**

1. `use_kg_search` (bool): Whether to use knowledge graph search
2. `kg_search_type` (str): Type of knowledge graph search ('global' or 'local')
3. `kg_search_level` (Optional[str]): Level of knowledge graph search
4. `generation_config` (Optional[GenerationConfig]): Configuration for knowledge graph search generation
5. `entity_types` (list): Types of entities to search for
6. `relationships` (list): Types of relationships to search for
7. `max_community_description_length` (int): Maximum length of community descriptions (default: 65536)
8. `max_llm_queries_for_global_search` (int): Maximum number of LLM queries for global search (default: 250)
9. `local_search_limits` (dict[str, int]): Limits for local search results by type

These settings provide fine-grained control over the search process in R2R, including vector search, hybrid search, and knowledge graph search configurations.

---

## Parsing & Chunking

R2R supports different parsing and chunking providers to extract text from various document formats and break it down into manageable pieces for efficient processing and retrieval.

To configure the parsing and chunking settings, update the `[ingestion]` section in your `r2r.toml` file:

```toml
[ingestion]
provider = "r2r" # or "unstructured_local" or "unstructured_api"
# ... provider-specific settings ...
```

### Runtime Configuration

In addition to configuring parsing and chunking settings in the `r2r.toml` file, you can also customize these settings at runtime when ingesting files using the Python SDK. This allows for more flexibility and control over the ingestion process on a per-file or per-request basis.

Some of the configurable options include:

- Chunking strategy (e.g., "recursive", "by_title", "basic")
- Chunk size and overlap
- Excluded parsers
- Provider-specific settings (e.g., max characters, overlap, languages)


An exhaustive list of runtime ingestion inputs to the `ingest-files` endpoint is shown below:

<ParamField path="file_paths" type="list[str]" required>
  A list of file paths or directory paths to ingest. If a directory path is provided, all files within the directory and its subdirectories will be ingested.
</ParamField>

<ParamField path="metadatas" type="Optional[list[dict]]">
  An optional list of metadata dictionaries corresponding to each file. If provided, the length should match the number of files being ingested.
</ParamField>

<ParamField path="document_ids" type="Optional[list[Union[UUID, str]]]">
  An optional list of document IDs to assign to the ingested files. If provided, the length should match the number of files being ingested.
</ParamField>

<ParamField path="versions" type="Optional[list[str]]">
  An optional list of version strings for the ingested files. If provided, the length should match the number of files being ingested.
</ParamField>

<ParamField path="ingestion_config" type="Optional[Union[dict, IngestionConfig]]">
  The ingestion config override parameter enables developers to customize their R2R chunking strategy at runtime. Learn more about [configuration here](/documentation/configuration/ingestion/parsing_and_chunking).
  <Expandable title="Other Provider Options">
    <ParamField path="provider" type="str" default="r2r">
      Which R2R ingestion provider to use. Options are "r2r".
    </ParamField>
    <ParamField path="chunking_strategy" type="str" default="recursive">
      Only `recursive` is currently supported.
    </ParamField>
    <ParamField path="chunk_size" type="number" default="1_024">
      The target size for output chunks.
    </ParamField>
    <ParamField path="chunk_overlap" type="number" default="512">
      The target overlap fraction for output chunks
    </ParamField>
    <ParamField path="excluded_parsers" type="list[str]" default="['mp4']">
      Which parsers to exclude from inside R2R.
    </ParamField>
  </Expandable>

  <Expandable title="Unstructured Provider Options">
    <ParamField path="provider" type="str" default="unstructured_local">
      Which unstructured ingestion provider to use. Options are "unstructured_local", or "unstructured_api".
    </ParamField>

    <ParamField path="max_chunk_size" type="Optional[int]" default="None">
      Sets a maximum size on output chunks.
    </ParamField>

    <ParamField path="combine_under_n_chars" type="Optional[int]">
      Combine chunks smaller than this number of characters.
    </ParamField>

    <ParamField path="max_characters" type="Optional[int]">
      Maximum number of characters per chunk.
    </ParamField>

    <ParamField path="coordinates" type="bool" default="False">
      Whether to include coordinates in the output.
    </ParamField>

    <ParamField path="encoding" type="Optional[str]">
      Encoding to use for text files.
    </ParamField>

    <ParamField path="extract_image_block_types" type="Optional[list[str]]">
      Types of image blocks to extract.
    </ParamField>

    <ParamField path="gz_uncompressed_content_type" type="Optional[str]">
      Content type for uncompressed gzip files.
    </ParamField>

    <ParamField path="hi_res_model_name" type="Optional[str]">
      Name of the high-resolution model to use.
    </ParamField>

    <ParamField path="include_orig_elements" type="Optional[bool]" default="False">
      Whether to include original elements in the output.
    </ParamField>

    <ParamField path="include_page_breaks" type="bool">
      Whether to include page breaks in the output.
    </ParamField>

    <ParamField path="languages" type="Optional[list[str]]">
      List of languages to consider for text processing.
    </ParamField>

    <ParamField path="multipage_sections" type="bool" default="True">
      Whether to allow sections to span multiple pages.
    </ParamField>

    <ParamField path="new_after_n_chars" type="Optional[int]">
      Start a new chunk after this many characters.
    </ParamField>

    <ParamField path="ocr_languages" type="Optional[list[str]]">
      Languages to use for OCR.
    </ParamField>

    <ParamField path="output_format" type="str" default="application/json">
      Format of the output.
    </ParamField>

    <ParamField path="overlap" type="int" default="0">
      Number of characters to overlap between chunks.
    </ParamField>

    <ParamField path="overlap_all" type="bool" default="False">
      Whether to overlap all chunks.
    </ParamField>

    <ParamField path="pdf_infer_table_structure" type="bool" default="True">
      Whether to infer table structure in PDFs.
    </ParamField>

    <ParamField path="similarity_threshold" type="Optional[float]">
      Threshold for considering chunks similar.
    </ParamField>

    <ParamField path="skip_infer_table_types" type="Optional[list[str]]">
      Types of tables to skip inferring.
    </ParamField>

    <ParamField path="split_pdf_concurrency_level" type="int" default="5">
      Concurrency level for splitting PDFs.
    </ParamField>

    <ParamField path="split_pdf_page" type="bool" default="True">
      Whether to split PDFs by page.
    </ParamField>

    <ParamField path="starting_page_number" type="Optional[int]">
      Page number to start processing from.
    </ParamField>

    <ParamField path="strategy" type="str" default="auto">
      Strategy for processing. Options are "auto", "fast", or "hi_res".
    </ParamField>

    <ParamField path="chunking_strategy" type="Optional[str]" default="by_title">
      Strategy for chunking. Options are "by_title" or "basic".
    </ParamField>

    <ParamField path="unique_element_ids" type="bool" default="False">
      Whether to generate unique IDs for elements.
    </ParamField>

    <ParamField path="xml_keep_tags" type="bool" default="False">
      Whether to keep XML tags in the output.
    </ParamField>
  </Expandable>
</ParamField>


For a comprehensive list of available runtime configuration options and examples of how to use them, refer to the [Python SDK Ingestion Documentation](/documentation/python-sdk/ingestion).


### Supported Providers

R2R offers two main parsing and chunking providers:

1. **R2R (default for 'light' installation)**:
   - Uses R2R's built-in parsing and chunking logic.
   - Supports a wide range of file types, including TXT, JSON, HTML, PDF, DOCX, PPTX, XLSX, CSV, Markdown, images, audio, and video.
   - Configuration options:
     ```toml
     [ingestion]
     provider = "r2r"
     chunking_strategy = "recursive"
     chunk_size = 1_024
     chunk_overlap = 512
     excluded_parsers = ["mp4"]
     ```
   - `chunking_strategy`: The chunking method ("recursive").
   - `chunk_size`: The target size for each chunk.
   - `chunk_overlap`: The number of characters to overlap between chunks.
   - `excluded_parsers`: List of parsers to exclude (e.g., ["mp4"]).

2. **Unstructured (default for 'full' installation)**:
   - Leverages Unstructured's open-source ingestion platform.
   - Provides more advanced parsing capabilities.
   - Configuration options:
     ```toml
     [ingestion]
     provider = "unstructured_local"
     strategy = "auto"
     chunking_strategy = "by_title"
     new_after_n_chars = 512
     max_characters = 1_024
     combine_under_n_chars = 128
     overlap = 20
     ```
   - `strategy`: The overall chunking strategy ("auto", "fast", or "hi_res").
   - `chunking_strategy`: The specific chunking method ("by_title" or "basic").
   - `new_after_n_chars`: Soft maximum size for a chunk.
   - `max_characters`: Hard maximum size for a chunk.
   - `combine_under_n_chars`: Minimum size for combining small sections.
   - `overlap`: Number of characters to overlap between chunks.

### Supported File Types

Both R2R and Unstructured providers support parsing a wide range of file types, including:

- TXT, JSON, HTML, PDF, DOCX, PPTX, XLSX, CSV, Markdown, images (BMP, GIF, HEIC, JPEG, JPG, PNG, SVG, TIFF), audio (MP3), video (MP4), and more.

Refer to the [Unstructured documentation](https://docs.unstructured.io/welcome) for more details on their ingestion capabilities and limitations.

### Configuring Parsing & Chunking

To configure parsing and chunking settings, update the `[ingestion]` section in your `r2r.toml` file with the desired provider and its specific settings.

For example, to use the R2R provider with custom chunk size and overlap:

```toml
[ingestion]
provider = "r2r"
chunking_strategy = "recursive"
chunk_size = 2_048
chunk_overlap = 256
excluded_parsers = ["mp4"]
```

Or, to use the Unstructured provider with a specific chunking strategy and character limits:

```toml
[ingestion]
provider = "unstructured_local"
strategy = "hi_res"
chunking_strategy = "basic"
new_after_n_chars = 1_000
max_characters = 2_000
combine_under_n_chars = 256
overlap = 50
```

Adjust the settings based on your specific requirements and the characteristics of your input documents.

### Next Steps

- Learn more about [Embedding Configuration](/documentation/configuration/ingestion/embedding).
- Explore [Knowledge Graph Configuration](/documentation/configuration/knowledge-graph/overview).
- Check out [Retrieval Configuration](/documentation/configuration/retrieval/overview).

---

## Introduction
R2R's ingestion pipeline efficiently processes various document formats, transforming them into searchable content. It seamlessly integrates with vector databases and knowledge graphs for optimal retrieval and analysis.

### Implementation Options
R2R offers two main implementations for ingestion:
- **Light**: Uses R2R's **built-in** ingestion logic, supporting a wide range of file types including TXT, JSON, HTML, PDF, DOCX, PPTX, XLSX, CSV, Markdown, images, audio, and video. For high-quality PDF parsing, it is recommended to use the zerox parser.
- **Full**: Leverages **Unstructured's** open-source [ingestion platform](https://docs.unstructured.io/open-source/introduction/overview) to handle supported file types. This is the default for the 'full' installation and provides more advanced parsing capabilities.

## Core Concepts

### Document Processing Pipeline
Inside R2R, ingestion refers to the complete pipeline for processing input data:
- Parsing files into text
- Chunking text into semantic units
- Generating embeddings
- Storing data for retrieval

Ingested files are stored with an associated document identifier and user identifier to enable comprehensive management.

### Multimodal Support
R2R has recently expanded its capabilities to include multimodal foundation models. In addition to using such models by default for images, R2R can now use them on PDFs by configuring the parser override:

```json
"ingestion_config": {
  "parser_overrides": {
    "pdf": "zerox"
  }
}
```

## Configuration

### Key Configuration Areas
Many settings are managed by the `r2r.toml` configuration file:

```toml
[database]
provider = "postgres"

[ingestion]
provider = "r2r"
chunking_strategy = "recursive"
chunk_size = 1_024
chunk_overlap = 512
excluded_parsers = ["mp4"]

[embedding]
provider = "litellm"
base_model = "openai/text-embedding-3-small"
base_dimension = 512
batch_size = 128
add_title_as_prefix = false
rerank_model = "None"
concurrent_request_limit = 256
```

### Configuration Impact
These settings directly influence how R2R performs ingestion:

1. **Database Configuration**
   - Configures Postgres database for semantic search and document management
   - Used during retrieval to find relevant document chunks via vector similarity

2. **Ingestion Settings**
   - Determines file type processing and text conversion methods
   - Controls text chunking protocols and granularity
   - Affects information storage and retrieval precision

3. **Embedding Configuration**
   - Defines model and parameters for text-to-vector conversion
   - Used during retrieval to embed user queries
   - Enables vector comparison against stored document embeddings

## Document Management

### Document Ingestion
The system provides several methods for ingesting documents:

1. **File Ingestion**
```python
file_paths = ['path/to/file1.txt', 'path/to/file2.txt']
metadatas = [{'key1': 'value1'}, {'key2': 'value2'}]

ingest_response = client.ingest_files(
    file_paths=file_paths,
    metadatas=metadatas,
    ingestion_config={
        "provider": "unstructured_local",
        "strategy": "auto",
        "chunking_strategy": "by_title",
        "new_after_n_chars": 256,
        "max_characters": 512
    }
)
```

2. **Direct Chunk Ingestion**
```python
chunks = [
  {
    "text": "Sample text chunk 1...",
  },
  {
    "text": "Sample text chunk 2...",
  }
]

ingest_response = client.ingest_chunks(
  chunks=chunks,
  metadata={"title": "Sample", "source": "example"}
)
```

### Document Updates
Update existing documents while maintaining version history:

```python
update_response = client.update_files(
    file_paths=file_paths,
    document_ids=document_ids,
    metadatas=[{"status": "reviewed"}]
)
```

### Vector Index Management

#### Creating Indices
Vector indices improve search performance for large collections:

```python
create_response = client.create_vector_index(
    table_name="vectors",
    index_method="hnsw",
    index_measure="cosine_distance",
    index_arguments={"m": 16, "ef_construction": 64},
    concurrently=True
)
```

Important considerations for index creation:
- Resource intensive process
- Requires pre-warming for optimal performance
- Parameters affect build time and search quality
- Monitor system resources during creation

#### Managing Indices
List and delete indices as needed:

```python
# List indices
indices = client.list_vector_indices(table_name="vectors")

# Delete index
delete_response = client.delete_vector_index(
    index_name="index_name",
    table_name="vectors",
    concurrently=True
)
```

## Troubleshooting

### Common Issues and Solutions

1. **Ingestion Failures**
   - Verify file permissions and paths
   - Check supported file formats
   - Ensure metadata matches file_paths
   - Monitor memory usage

2. **Chunking Issues**
   - Large chunks may impact retrieval quality
   - Small chunks may lose context
   - Adjust overlap for better context preservation

3. **Vector Index Performance**
   - Monitor creation time
   - Check memory usage
   - Verify warm-up queries
   - Consider rebuilding if quality degrades

## Pipeline Architecture
The ingestion pipeline consists of several key components:

```mermaid
graph TD
    A[Input Documents] --> B[Parsing Pipe]
    B --> C[Embedding Pipeline]
    B --> D[Knowledge Graph Pipeline]
    C --> E[Vector Database]
    D --> F[Knowledge Graph Database]
```

This modular design allows for customization and extension of individual components while maintaining robust document processing capabilities.

## Next Steps

For more detailed information on configuring specific components of the ingestion pipeline, please refer to the following pages:

- [Parsing & Chunking Configuration](/documentation/configuration/ingestion/parsing_and_chunking)
- [Embedding Configuration](/documentation/configuration/ingestion/embedding)
- [Knowledge Graph Configuration](/documentation/configuration/knowledge-graph/overview)
- [Retrieval Configuration](/documentation/configuration/retrieval/overview)

---


<Note>
Occasionally this SDK documentation falls out of date, cross-check with the automatcially generated <a href="/api-reference/introduction"> API Reference documentation </a> for the latest parameters.
</Note>


A collection in R2R is a logical grouping of users and documents that allows for efficient access control and organization. Collections enable you to manage permissions and access to documents at a collection level, rather than individually.

R2R provides a comprehensive set of collection features, allowing you to implement efficient access control and organization of users and documents in your applications.


<Note>
Collection permissioning in R2R is still under development and as a result the API will likely evolve.
</Note>


## Collection creation and Management

### Create a Collection

Create a new collection with a name and optional description:

```python
create_collection_response = client.create_collection("Marketing Team", "Collection for marketing department")
collection_id = create_collection_response["results"]["collection_id] # '123e4567-e89b-12d3-a456-426614174000'
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      ```python
      {
        'results': {
          'collection_id': '123e4567-e89b-12d3-a456-426614174000',
          'name': 'Marketing Team',
          'description': 'Collection for marketing department',
          'created_at': '2024-07-16T22:53:47.524794Z',
          'updated_at': '2024-07-16T22:53:47.524794Z'
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Get Collection details

Retrieve details about a specific collection:

```python
collection_details = client.get_collection(collection_id)
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      ```python
      {
        'results': {
          'collection_id': '123e4567-e89b-12d3-a456-426614174000',
          'name': 'Marketing Team',
          'description': 'Collection for marketing department',
          'created_at': '2024-07-16T22:53:47.524794Z',
          'updated_at': '2024-07-16T22:53:47.524794Z'
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Update a Collection

Update a collection's name or description:

```python
update_result = client.update_collection(
  collection_id,
  name="Updated Marketing Team",
  description="New description for marketing team"
)
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      ```python
      {
        'results': {
          'collection_id': '123e4567-e89b-12d3-a456-426614174000',
          'name': 'Updated Marketing Team',
          'description': 'New description for marketing team',
          'created_at': '2024-07-16T22:53:47.524794Z',
          'updated_at': '2024-07-16T23:15:30.123456Z'
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### List Collections

Get a list of all collections:

```python
collections_list = client.list_collections()
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      ```python
      {
        'results': [
          {
            'collection_id': '123e4567-e89b-12d3-a456-426614174000',
            'name': 'Updated Marketing Team',
            'description': 'New description for marketing team',
            'created_at': '2024-07-16T22:53:47.524794Z',
            'updated_at': '2024-07-16T23:15:30.123456Z'
          },
          # ... other collections ...
        ]
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

## User Management in Collections

### Add User to Collection

Add a user to a collection:

```python

user_id = '456e789f-g01h-34i5-j678-901234567890'  # This should be a valid user ID
add_user_result = client.add_user_to_collection(user_id, collection_id)
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      ```python
      {
        'results': {
          'message': 'User successfully added to the collection'
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Remove User from Collection

Remove a user from a collection:

```python
remove_user_result = client.remove_user_from_collection(user_id, collection_id)
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      ```python
      {
        'results': {
          'message': 'User successfully removed from the collection'
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### List Users in Collection

Get a list of all users in a specific collection:

```python
users_in_collection = client.get_users_in_collection(collection_id)
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      ```python
      {
        'results': [
          {
            'user_id': '456e789f-g01h-34i5-j678-901234567890',
            'email': 'user@example.com',
            'name': 'John Doe',
            # ... other user details ...
          },
          # ... other users ...
        ]
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Get User's Collections

Get all collections that a user is a member of:

```python
user_collections = client.user_collections(user_id)
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      ```python
      {
        'results': [
          {
            'collection_id': '123e4567-e89b-12d3-a456-426614174000',
            'name': 'Updated Marketing Team',
            # ... other Collection details ...
          },
          # ... other collections ...
        ]
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

## Document Management in Collections

### Assign Document to Collection

Assign a document to a collection:

```python
document_id = '789g012j-k34l-56m7-n890-123456789012' # must be a valid document id
assign_doc_result = client.assign_document_to_collection(document_id, collection_id)
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      ```python
      {
        'results': {
          'message': 'Document successfully assigned to the collection'
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Remove Document from Collection

Remove a document from a collection:

```python
remove_doc_result = client.remove_document_from_collection(document_id, collection_id)
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      ```python
      {
        'results': {
          'message': 'Document successfully removed from the collection'
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### List Documents in Collection

Get a list of all documents in a specific collection:

```python
docs_in_collection = client.documents_in_collection(collection_id)
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      ```python
      {
        'results': [
          {
            'document_id': '789g012j-k34l-56m7-n890-123456789012',
            'title': 'Marketing Strategy 2024',
            # ... other document details ...
          },
          # ... other documents ...
        ]
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Get Document's Collections

Get all collections that a document is assigned to:

```python
document_collections = client.document_collections(document_id)
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      ```python
      {
        'results': [
          {
            'collection_id': '123e4567-e89b-12d3-a456-426614174000',
            'name': 'Updated Marketing Team',
            # ... other Collection details ...
          },
          # ... other collections ...
        ]
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

## Advanced Collection Management

### Collection Overview

Get an overview of collections, including user and document counts:

```python
collections_overview = client.collections_overview()
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      ```python
      {
        'results': [
          {
            'collection_id': '123e4567-e89b-12d3-a456-426614174000',
            'name': 'Updated Marketing Team',
            'description': 'New description for marketing team',
            'user_count': 5,
            'document_count': 10,
            'created_at': '2024-07-16T22:53:47.524794Z',
            'updated_at': '2024-07-16T23:15:30.123456Z'
          },
          # ... other collections ...
        ]
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Delete a Collection

Delete a collection:

```python
delete_result = client.delete_collection(collection_id)
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      ```python
      {
        'results': {
          'message': 'Collection successfully deleted'
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

## Pagination and Filtering

Many collection-related methods support pagination and filtering:

```python
# List collections with pagination
paginated_collection = client.list_collections(offset=10, limit=20)

# Get users in a collection with pagination
paginated_users = client.get_users_in_collection(collection_id, offset=5, limit=10)

# Get documents in a collection with pagination
paginated_docs = client.documents_in_collection(collection_id, offset=0, limit=50)

# Get collections overview with specific collection IDs
specific_collections_overview = client.collections_overview(collection_ids=['id1', 'id2', 'id3'])
```

## Security Considerations

When implementing collection permissions, consider the following security best practices:

1. Always use HTTPS in production to encrypt data in transit.
2. Implement the principle of least privilege by assigning the minimum necessary permissions to users and collections.
3. Regularly audit collection memberships and document assignments.
4. Ensure that only authorized users (e.g., admins) can perform collection management operations.
5. Implement comprehensive logging for all collection-related actions.
6. Consider implementing additional access controls or custom roles within your application logic for more fine-grained permissions.

For more advanced use cases or custom implementations, refer to the R2R documentation or reach out to the community for support.

---

## Knowledge Graph Provider

R2R supports knowledge graph functionality to enhance document understanding and retrieval. By default, R2R creates the graph by clustering with `graspologic` and saving the output triples and relationships into Postgres. We are actively working to integrate with [Memgraph](https://memgraph.com/docs). You can find out more about creating knowledge graphs in the [GraphRAG Cookbook](/cookbooks/graphrag).


To configure the knowledge graph settings for your project, edit the `kg` section in your `r2r.toml` file:

```toml r2r.toml
[database]
provider = "postgres"
batch_size = 256
kg_triples_extraction_prompt = "graphrag_triples_extraction_few_shot"

  [database.kg_creation_settings]
    entity_types = [] # if empty, all entities are extracted
    relation_types = [] # if empty, all relations are extracted
    generation_config = { model = "openai/gpt-4o-mini" }
    max_knowledge_triples = 100 # max number of triples to extract for each document chunk
    fragment_merge_count = 4 # number of fragments to merge into a single extraction

  [database.kg_enrichment_settings]
    max_description_input_length = 65536 # increase if you want more comprehensive descriptions
    max_summary_input_length = 65536
    generation_config = { model = "openai/gpt-4o-mini" } # and other generation params below
    leiden_params = {}

  [database.kg_search_settings]
    generation_config = { model = "openai/gpt-4o-mini" }
```


Let's break down the knowledge graph configuration options:

- `provider`: Specifies the knowledge graph provider. Currently, "postgres" is supported.
- `batch_size`: Determines the number of entities or relationships to process in a single batch during import operations.
- `kg_triples_extraction_prompt`: Specifies the prompt template to use for extracting knowledge graph information from text.
- `kg_creation_settings`: Configuration for the model used in knowledge graph creation.
  - `max_knowledge_triples`: The maximum number of knowledge triples to extract for each document chunk.
  - `fragment_merge_count`: The number of fragments to merge into a single extraction.
  - `generation_config`: Configuration for the model used in knowledge graph creation.
- `kg_enrichment_settings`: Similar configuration for the model used in knowledge graph enrichment.
  - `generation_config`: Configuration for the model used in knowledge graph enrichment.
  - `leiden_params`: Parameters for the Leiden algorithm.
- `kg_search_settings`: Similar configuration for the model used in knowledge graph search operations.



<Note>
Setting configuration values in the `r2r.toml` will override environment variables by default.
</Note>


### Knowledge Graph Operations

1. **Entity Management**: Add, update, and retrieve entities in the knowledge graph.
2. **Relationship Management**: Create and query relationships between entities.
3. **Batch Import**: Efficiently import large amounts of data using batched operations.
4. **Vector Search**: Perform similarity searches on entity embeddings.
5. **Community Detection**: Identify and manage communities within the graph.

### Customization

You can customize the knowledge graph extraction and search processes by modifying the `kg_triples_extraction_prompt` and adjusting the model configurations in `kg_extraction_settings` and `kg_search_settings`. Moreover, you can customize the LLM models used in various parts of the knowledge graph creation process. All of these options can be selected at runtime, with the only exception being the specified database provider. For more details, refer to the knowledge graph settings in the [search API](/api-reference/endpoint/search).

By leveraging the knowledge graph capabilities, you can enhance R2R's understanding of document relationships and improve the quality of search and retrieval operations.



## Next Steps

For more detailed information on configuring specific components of the ingestion pipeline, please refer to the following pages:

- [Ingestion Configuration](/documentation/configuration/ingestion/overview)
- [Enrichment Configuration](/documentation/configuration/knowledge-graph/enrichment)
- [Retrieval Configuration](/documentation/configuration/retrieval/overview)

---


## Introduction

R2R's `LLMProvider` supports multiple third-party Language Model (LLM) providers, offering flexibility in choosing and switching between different models based on your specific requirements. This guide provides an in-depth look at configuring and using various LLM providers within the R2R framework.

## Architecture Overview

R2R's LLM system is built on a flexible provider model:

1. **LLM Provider**: An abstract base class that defines the common interface for all LLM providers.
2. **Specific LLM Providers**: Concrete implementations for different LLM services (e.g., OpenAI, LiteLLM).

These providers work in tandem to ensure flexible and efficient language model integration.

## Providers

### LiteLLM Provider (Default)

The default `LiteLLMProvider` offers a unified interface for multiple LLM services.

Key features:
- Support for OpenAI, Anthropic, Vertex AI, HuggingFace, Azure OpenAI, Ollama, Together AI, and Openrouter
- Consistent API across different LLM providers
- Easy switching between models

### OpenAI Provider

The `OpenAILLM` class provides direct integration with OpenAI's models.

Key features:
- Direct access to OpenAI's API
- Support for the latest OpenAI models
- Fine-grained control over model parameters

### Local Models

Support for running models locally using Ollama or other local inference engines, through LiteLLM.

Key features:
- Privacy-preserving local inference
- Customizable model selection
- Reduced latency for certain use cases

## Configuration

### LLM Configuration

Update the `completions` section in your `r2r.toml` file:
```
[completions]
provider = "litellm"

[completions.generation_config]
model = "gpt-4"
temperature = 0.7
max_tokens = 150
```
<Note>
The provided `generation_config` is used to establish the default generation parameters for your deployment. These settings can be overridden at runtime, offering flexibility in your application. You can adjust parameters:

1. At the application level, by modifying the R2R configuration
2. For individual requests, by passing custom parameters to the `rag` or `get_completion` methods
3. Through API calls, by including specific parameters in your request payload

This allows you to fine-tune the behavior of your language model interactions on a per-use basis while maintaining a consistent baseline configuration.
</Note>


## Security Best Practices

1. **API Key Management**: Use environment variables or secure key management solutions for API keys.
2. **Rate Limiting**: Implement rate limiting to prevent abuse of LLM endpoints.
3. **Input Validation**: Sanitize and validate all inputs before passing them to LLMs.
4. **Output Filtering**: Implement content filtering for LLM outputs to prevent inappropriate content.
5. **Monitoring**: Regularly monitor LLM usage and outputs for anomalies or misuse.

## Custom LLM Providers in R2R

### LLM Provider Structure

The LLM system in R2R is built on two main components:

1. `LLMConfig`: A configuration class for LLM providers.
2. `LLMProvider`: An abstract base class that defines the interface for all LLM providers.

### LLMConfig

The `LLMConfig` class is used to configure LLM providers:

```python
from r2r.base import ProviderConfig
from r2r.base.abstractions.llm import GenerationConfig
from typing import Optional

class LLMConfig(ProviderConfig):
    provider: Optional[str] = None
    generation_config: Optional[GenerationConfig] = None

    def validate(self) -> None:
        if not self.provider:
            raise ValueError("Provider must be set.")
        if self.provider and self.provider not in self.supported_providers:
            raise ValueError(f"Provider '{self.provider}' is not supported.")

    @property
    def supported_providers(self) -> list[str]:
        return ["litellm", "openai"]
```

### LLMProvider

The `LLMProvider` is an abstract base class that defines the common interface for all LLM providers:

```python
from abc import abstractmethod
from r2r.base import Provider
from r2r.base.abstractions.llm import GenerationConfig, LLMChatCompletion, LLMChatCompletionChunk

class LLMProvider(Provider):
    def __init__(self, config: LLMConfig) -> None:
        if not isinstance(config, LLMConfig):
            raise ValueError("LLMProvider must be initialized with a `LLMConfig`.")
        super().__init__(config)

    @abstractmethod
    def get_completion(
        self,
        messages: list[dict],
        generation_config: GenerationConfig,
        **kwargs,
    ) -> LLMChatCompletion:
        pass

    @abstractmethod
    def get_completion_stream(
        self,
        messages: list[dict],
        generation_config: GenerationConfig,
        **kwargs,
    ) -> LLMChatCompletionChunk:
        pass
```

### Creating a Custom LLM Provider

To create a custom LLM provider, follow these steps:

1. Create a new class that inherits from `LLMProvider`.
2. Implement the required methods: `get_completion` and `get_completion_stream`.
3. (Optional) Add any additional methods or attributes specific to your provider.

Here's an example of a custom LLM provider:

```python
import logging
from typing import Generator
from r2r.base import LLMProvider, LLMConfig, LLMChatCompletion, LLMChatCompletionChunk
from r2r.base.abstractions.llm import GenerationConfig

logger = logging.getLogger(__name__)

class CustomLLMProvider(LLMProvider):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        # Initialize any custom attributes or connections here
        self.custom_client = self._initialize_custom_client()

    def _initialize_custom_client(self):
        # Initialize your custom LLM client here
        pass

    def get_completion(
        self,
        messages: list[dict],
        generation_config: GenerationConfig,
        **kwargs,
    ) -> LLMChatCompletion:
        # Implement the logic to get a completion from your custom LLM
        response = self.custom_client.generate(messages, **generation_config.dict(), **kwargs)

        # Convert the response to LLMChatCompletion format
        return LLMChatCompletion(
            id=response.id,
            choices=[
                {
                    "message": {
                        "role": "assistant",
                        "content": response.text
                    },
                    "finish_reason": response.finish_reason
                }
            ],
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        )

    def get_completion_stream(
        self,
        messages: list[dict],
        generation_config: GenerationConfig,
        **kwargs,
    ) -> Generator[LLMChatCompletionChunk, None, None]:
        # Implement the logic to get a streaming completion from your custom LLM
        stream = self.custom_client.generate_stream(messages, **generation_config.dict(), **kwargs)

        for chunk in stream:
            yield LLMChatCompletionChunk(
                id=chunk.id,
                choices=[
                    {
                        "delta": {
                            "role": "assistant",
                            "content": chunk.text
                        },
                        "finish_reason": chunk.finish_reason
                    }
                ]
            )

    # Add any additional methods specific to your custom provider
    def custom_method(self, *args, **kwargs):
        # Implement custom functionality
        pass
```

### Registering and Using the Custom Provider

To use your custom LLM provider in R2R:

1. Update the `LLMConfig` class to include your custom provider:

```python
class LLMConfig(ProviderConfig):
    # ...existing code...

    @property
    def supported_providers(self) -> list[str]:
        return ["litellm", "openai", "custom"]  # Add your custom provider here
```

2. Update your R2R configuration to use the custom provider:

```toml
[completions]
provider = "custom"

[completions.generation_config]
model = "your-custom-model"
temperature = 0.7
max_tokens = 150
```

3. In your R2R application, register the custom provider:

```python
from r2r import R2R
from r2r.base import LLMConfig
from your_module import CustomLLMProvider

def get_llm_provider(config: LLMConfig):
    if config.provider == "custom":
        return CustomLLMProvider(config)
    # ... handle other providers ...

r2r = R2R(llm_provider_factory=get_llm_provider)
```

Now you can use your custom LLM provider seamlessly within your R2R application:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
]

response = r2r.get_completion(messages)
print(response.choices[0].message.content)
```

By following this structure, you can integrate any LLM or service into R2R, maintaining consistency with the existing system while adding custom functionality as needed.
## Prompt Engineering

R2R supports advanced prompt engineering techniques:

1. **Template Management**: Create and manage reusable prompt templates.
2. **Dynamic Prompts**: Generate prompts dynamically based on context or user input.
3. **Few-shot Learning**: Incorporate examples in your prompts for better results.

## Troubleshooting

Common issues and solutions:

1. **API Key Errors**: Ensure your API keys are correctly set and have the necessary permissions.
2. **Rate Limiting**: Implement exponential backoff for retries on rate limit errors.
3. **Context Length Errors**: Be mindful of the maximum context length for your chosen model.
4. **Model Availability**: Ensure the requested model is available and properly configured.

## Performance Considerations

1. **Batching**: Use batching for multiple, similar requests to improve throughput.
2. **Streaming**: Utilize streaming for long-form content generation to improve perceived latency.
3. **Model Selection**: Balance between model capability and inference speed based on your use case.

## Server Configuration

The `R2RConfig` class handles the configuration of various components, including LLMs. Here's a simplified version:

```python
class R2RConfig:
    REQUIRED_KEYS: dict[str, list] = {
        # ... other keys ...
        "completions": ["provider"],
        # ... other keys ...
    }

    def __init__(self, config_data: dict[str, Any]):
        # Load and validate configuration
        # ...

        # Set LLM configuration
        self.completions = LLMConfig.create(**self.completions)

        # Override GenerationConfig defaults
        GenerationConfig.set_default(**self.completions.get("generation_config", {}))

        # ... other initialization ...
```

This configuration system allows for flexible setup of LLM providers and their default parameters.

## Conclusion

R2R's LLM system provides a flexible and powerful foundation for integrating various language models into your applications. By understanding the available providers, configuration options, and best practices, you can effectively leverage LLMs to enhance your R2R-based projects.

For further customization and advanced use cases, refer to the [R2R API Documentation](/api-reference) and [configuration guide](/documentation/configuration).

---


Vector search settings can be configured both server-side and at runtime. Runtime settings are passed as a dictionary to the search and RAG endpoints. You may refer to the [search API documentation here](/api-reference/endpoint/search) for additional materials.


Example using the Python SDK:

```python
vector_search_settings = {
    "use_vector_search": True,
    "search_filters": {"document_type": {"$eq": "article"}},
    "search_limit": 20,
    "use_hybrid_search": True,
    "selected_collection_ids": ["c3291abf-8a4e-5d9d-80fd-232ef6fd8526"]
}

response = client.search("query", vector_search_settings=vector_search_settings)
```

#### Configurable Parameters

**VectorSearchSettings**

1. `use_vector_search` (bool): Whether to use vector search
2. `use_hybrid_search` (bool): Whether to perform a hybrid search (combining vector and keyword search)
3. `filters` (dict): Alias for filters
3. `search_filters` (dict): Filters to apply to the vector search
4. `search_limit` (int): Maximum number of results to return (1-1000)
5. `selected_collection_ids` (list[UUID]): Collection Ids to search for
6. `index_measure` (IndexMeasure): The distance measure to use for indexing (cosine_distance, l2_distance, or max_inner_product)
7. `include_values` (bool): Whether to include search score values in the search results
8. `include_metadatas` (bool): Whether to include element metadata in the search results
9. `probes` (Optional[int]): Number of ivfflat index lists to query (default: 10)
10. `ef_search` (Optional[int]): Size of the dynamic candidate list for HNSW index search (default: 40)
11. `hybrid_search_settings` (Optional[HybridSearchSettings]): Settings for hybrid search

**HybridSearchSettings**

1. `full_text_weight` (float): Weight to apply to full text search (default: 1.0)
2. `semantic_weight` (float): Weight to apply to semantic search (default: 5.0)
3. `full_text_limit` (int): Maximum number of results to return from full text search (default: 200)
4. `rrf_k` (int): K-value for RRF (Rank Reciprocal Fusion) (default: 50)

#### Advanced Filtering

R2R supports complex filtering using PostgreSQL-based queries. Allowed operators include:
- `eq`, `neq`: Equality and inequality
- `gt`, `gte`, `lt`, `lte`: Greater than, greater than or equal, less than, less than or equal
- `like`, `ilike`: Pattern matching (case-sensitive and case-insensitive)
- `in`, `nin`: Inclusion and exclusion in a list of values

Example of advanced filtering:

```python
filters = {
    "$and": [
        {"publication_date": {"$gte": "2023-01-01"}},
        {"author": {"$in": ["John Doe", "Jane Smith"]}},
        {"category": {"$ilike": "%technology%"}}
    ]
}
vector_search_settings["filters"] = filters
```

---


## Introduction

Retrieval in R2R is a sophisticated system that leverages ingested data to provide powerful search and Retrieval-Augmented Generation (RAG) capabilities. It combines vector-based semantic search, knowledge graph querying, and language model generation to deliver accurate and contextually relevant results.

## Key Configuration Areas

To configure the retrieval system in R2R, you'll need to focus on several areas in your `r2r.toml` file:

```toml
[database]
provider = "postgres"

[embedding]
provider = "litellm"
base_model = "openai/text-embedding-3-small"
base_dimension = 512
batch_size = 128
add_title_as_prefix = false
rerank_model = "None"
concurrent_request_limit = 256

[database]
provider = "postgres"
batch_size = 256

[completion]
provider = "litellm"
concurrent_request_limit = 16

[completion.generation_config]
model = "openai/gpt-4"
temperature = 0.1
top_p = 1
max_tokens_to_sample = 1_024
stream = false
```

These settings directly impact how R2R performs retrieval operations:

- The `[database]` section configures the vector database used for semantic search and document management.
- The `[embedding]` section defines the model and parameters for converting text into vector embeddings.
- The `[database]` section, when configured, enables knowledge graph-based retrieval.
- The `[completion]` section sets up the language model used for generating responses in the RAG pipeline.

## Customization and Advanced Features

R2R's retrieval system is highly customizable, allowing you to:

- Implement hybrid search combining vector-based and knowledge graph queries
- Customize search filters, limits, and query generation
- Add custom pipes to the search and RAG pipelines
- Implement reranking for improved result relevance

### Structured Outputs

R2R supports structured outputs for RAG responses, allowing you to define specific response formats using Pydantic models. This ensures consistent, type-safe responses that can be easily validated and processed programmatically.

<Info>Some models may require the word 'JSON' to appear in their prompt for structured outputs to work. Be sure to update your prompt to reflect this, if necessary.</Info>

Here's a simple example of using structured outputs with Pydantic models:

```python
from r2r import R2RClient, GenerationConfig
from pydantic import BaseModel

# Initialize the client
client = R2RClient()

# Define your response structure
class ResponseModel(BaseModel):
    answer: str
    sources: list[str]

# Make a RAG query with structured output
response = client.rag(
    query="…",
    rag_generation_config=GenerationConfig(
        response_format=ResponseModel
    )
)
```

## Pipeline Architecture

Retrieval in R2R is implemented as a pipeline and consists of the main components shown below:

```mermaid
graph TD
    A[User Query] --> B[RAG Pipeline]
    B --> C[Search Pipeline]
    B --> D[RAG Generation Pipeline]
    C --> E[Vector Search]
    C --> F[Knowledge Graph Search]
    E --> G[Search Results]
    F --> G
    G --> D
    D --> H[Generated Response]
```


## Next Steps

For more detailed information on configuring specific components of the ingestion pipeline, please refer to the following pages:

- [Ingestion Configuration](/documentation/configuration/ingestion/overview)
- [Vector Search Configuration](/documentation/configuration/retrieval/vector-search)
- [Knowledge Graph Search Configuration](/documentation/configuration/retrieval/knowledge-graph)
- [Retrieval Configuration](/documentation/configuration/retrieval/overview)

---


## Analytics and Observability
R2R provides various tools for analytics and observability to help you monitor and improve the performance of your RAG system.


<ParamField path="message_id" type="uuid.UUID" default="None" required>
The unique identifier of the completion message to be scored. Found in logs for a message.
</ParamField>
<ParamField path="score" type="float" default="None" required>
The score to assign to the completion, ranging from -1 to 1.
</ParamField>

---


<Note>
This SDK documentation is periodically updated. For the latest parameter details, please cross-reference with the <a href="/api-reference/introduction">API Reference documentation</a>.
</Note>

Inside R2R, `ingestion` refers to the complete pipeline for processing input data:
- Parsing files into text
- Chunking text into semantic units
- Generating embeddings
- Storing data for retrieval

Ingested files are stored with an associated document identifier as well as a user identifier to enable comprehensive management.

## Document Ingestion and Management

<Note>

R2R has recently expanded the available options for ingesting files using multimodal foundation models. In addition to using such models by default for images, R2R can now use them on PDFs, [like it is shown here](https://github.com/getomni-ai/zerox), by passing the following in your ingestion configuration:

```json
"ingestion_config": {
  ...,
  "parser_overrides": {
    "pdf": "zerox"
  }
}
```

We recommend this method for achieving the highest quality ingestion results.

</Note>



### Ingest Files



Ingest files or directories into your R2R system:

```python
file_paths = ['path/to/file1.txt', 'path/to/file2.txt']
metadatas = [{'key1': 'value1'}, {'key2': 'value2'}]

# Ingestion configuration for `R2R Full`
ingest_response = client.ingest_files(
    file_paths=file_paths,
    metadatas=metadatas,
    # Runtime chunking configuration
    ingestion_config={
        "provider": "unstructured_local",  # Local processing
        "strategy": "auto",                # Automatic processing strategy
        "chunking_strategy": "by_title",   # Split on title boundaries
        "new_after_n_chars": 256,          # Start new chunk (soft limit)
        "max_characters": 512,             # Maximum chunk size (hard limit)
        "combine_under_n_chars": 64,       # Minimum chunk size
        "overlap": 100,                    # Character overlap between chunks
        "chunk_enrichment_settings": {     # Document enrichment settings
            "enable_chunk_enrichment": False,
        }
    }
)
```

An `ingested` file is parsed, chunked, embedded and stored inside your R2R system. The stored information includes a document identifier, a corresponding user identifier, and other metadata. Knowledge graph creation is done separately and at the `collection` level. Refer to the [ingestion configuration](/documentation/configuration/ingestion/parsing_and_chunking) section for comprehensive details on available options.

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      The response from the R2R system after ingesting the files.
      ```bash
      [{'message': 'Ingestion task queued successfully.', 'task_id': '6e27dfca-606d-422d-b73f-2d9e138661b4', 'document_id': 'c3291abf-8a4e-5d9d-80fd-232ef6fd8526'}, ...]
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="file_paths" type="list[str]" required>
  A list of file paths or directory paths to ingest. If a directory path is provided, all files within the directory and its subdirectories will be ingested.
</ParamField>

<ParamField path="metadatas" type="Optional[list[dict]]">
  An optional list of metadata dictionaries corresponding to each file. If provided, the length should match the number of files being ingested.
</ParamField>

<ParamField path="document_ids" type="Optional[list[Union[UUID, str]]]">
  An optional list of document IDs to assign to the ingested files. If provided, the length should match the number of files being ingested.
</ParamField>

<ParamField path="versions" type="Optional[list[str]]">
  An optional list of version strings for the ingested files. If provided, the length should match the number of files being ingested.
</ParamField>


<ParamField path="ingestion_config" type="Optional[Union[dict, IngestionConfig]]">
  The ingestion config override parameter enables developers to customize their R2R chunking strategy at runtime. Learn more about [configuration here](/documentation/configuration/ingestion/parsing_and_chunking).
  <Expandable title="Other Provider Options">
    <ParamField path="provider" type="str" default="r2r">
      Which R2R ingestion provider to use. Options are "r2r".
    </ParamField>
    <ParamField path="chunking_strategy" type="str" default="recursive">
      Only `recursive` is currently supported.
    </ParamField>
    <ParamField path="chunk_size" type="number" default="1_024">
      The target size for output chunks.
    </ParamField>
    <ParamField path="chunk_overlap" type="number" default="512">
      The target overlap fraction for output chunks
    </ParamField>
    <ParamField path="parser_overrides" type="dict[str, str]" >
      Dictionary of filetypes and selected override parsers. Currently only `{"pdf": "zerox"}` is supported.
    </ParamField>
  </Expandable>

  <Expandable title="Unstructured Provider Options">
    <ParamField path="provider" type="str" default="unstructured_local">
      Which unstructured ingestion provider to use. Options are "unstructured_local", or "unstructured_api".
    </ParamField>

    <ParamField path="parser_overrides" type="dict[str, str]" >
      Dictionary of filetypes and selected override parsers. Currently only `{"pdf": "zerox"}` is supported.
    </ParamField>

    <ParamField path="max_chunk_size" type="Optional[int]" default="None">
      Sets a maximum size on output chunks.
    </ParamField>

    <ParamField path="combine_under_n_chars" type="Optional[int]">
      Combine chunks smaller than this number of characters.
    </ParamField>

    <ParamField path="max_characters" type="Optional[int]">
      Maximum number of characters per chunk.
    </ParamField>

    <ParamField path="coordinates" type="bool" default="False">
      Whether to include coordinates in the output.
    </ParamField>

    <ParamField path="encoding" type="Optional[str]">
      Encoding to use for text files.
    </ParamField>

    <ParamField path="extract_image_block_types" type="Optional[list[str]]">
      Types of image blocks to extract.
    </ParamField>

    <ParamField path="gz_uncompressed_content_type" type="Optional[str]">
      Content type for uncompressed gzip files.
    </ParamField>

    <ParamField path="hi_res_model_name" type="Optional[str]">
      Name of the high-resolution model to use.
    </ParamField>

    <ParamField path="include_orig_elements" type="Optional[bool]" default="False">
      Whether to include original elements in the output.
    </ParamField>

    <ParamField path="include_page_breaks" type="bool">
      Whether to include page breaks in the output.
    </ParamField>

    <ParamField path="languages" type="Optional[list[str]]">
      List of languages to consider for text processing.
    </ParamField>

    <ParamField path="multipage_sections" type="bool" default="True">
      Whether to allow sections to span multiple pages.
    </ParamField>

    <ParamField path="new_after_n_chars" type="Optional[int]">
      Start a new chunk after this many characters.
    </ParamField>

    <ParamField path="ocr_languages" type="Optional[list[str]]">
      Languages to use for OCR.
    </ParamField>

    <ParamField path="output_format" type="str" default="application/json">
      Format of the output.
    </ParamField>

    <ParamField path="overlap" type="int" default="0">
      Number of characters to overlap between chunks.
    </ParamField>

    <ParamField path="overlap_all" type="bool" default="False">
      Whether to overlap all chunks.
    </ParamField>

    <ParamField path="pdf_infer_table_structure" type="bool" default="True">
      Whether to infer table structure in PDFs.
    </ParamField>

    <ParamField path="similarity_threshold" type="Optional[float]">
      Threshold for considering chunks similar.
    </ParamField>

    <ParamField path="skip_infer_table_types" type="Optional[list[str]]">
      Types of tables to skip inferring.
    </ParamField>

    <ParamField path="split_pdf_concurrency_level" type="int" default="5">
      Concurrency level for splitting PDFs.
    </ParamField>

    <ParamField path="split_pdf_page" type="bool" default="True">
      Whether to split PDFs by page.
    </ParamField>

    <ParamField path="starting_page_number" type="Optional[int]">
      Page number to start processing from.
    </ParamField>

    <ParamField path="strategy" type="str" default="auto">
      Strategy for processing. Options are "auto", "fast", or "hi_res".
    </ParamField>

    <ParamField path="chunking_strategy" type="Optional[str]" default="by_title">
      Strategy for chunking. Options are "by_title" or "basic".
    </ParamField>

    <ParamField path="unique_element_ids" type="bool" default="False">
      Whether to generate unique IDs for elements.
    </ParamField>

    <ParamField path="xml_keep_tags" type="bool" default="False">
      Whether to keep XML tags in the output.
    </ParamField>
  </Expandable>

  <ParamField path="run_with_orchestration" type="Optional[bool]">
    Whether or not ingestion runs with [orchestration](/cookbooks/orchestration), default is `True`. When set to `False`, the ingestion process will run synchronous and directly return the result.
  </ParamField>


</ParamField>

### Understanding Ingestion Status

After calling `ingest_files`, the response includes important status information:

```python
# Successful ingestion
{
    'message': 'Ingestion task queued successfully.',
    'task_id': '6e27dfca-606d-422d-b73f-2d9e138661b4',
    'document_id': 'c3291abf-8a4e-5d9d-80fd-232ef6fd8526'
}

# Check document status later
doc_status = client.documents_overview(
    document_ids=['c3291abf-8a4e-5d9d-80fd-232ef6fd8526']
)
# ingestion_status will be one of: 'pending', 'processing', 'success', 'failed'
```

<Note>

We have added support for contextual chunk enrichment! You can learn more about it [here](/cookbooks/contextual-enrichment).

Currently, you need to enable it in your ingestion config:

```toml
[ingestion.chunk_enrichment_settings]
    enable_chunk_enrichment = true # disabled by default
    strategies = ["semantic", "neighborhood"]
    forward_chunks = 3            # Look ahead 3 chunks
    backward_chunks = 3           # Look behind 3 chunks
    semantic_neighbors = 10       # Find 10 semantically similar chunks
    semantic_similarity_threshold = 0.7  # Minimum similarity score
    generation_config = { model = "openai/gpt-4o-mini" }
```

</Note>

### Ingest Chunks

The `ingest_chunks` method allows direct ingestion of pre-processed text, bypassing the standard parsing pipeline. This is useful for:
- Custom preprocessing pipelines
- Streaming data ingestion
- Working with non-file data sources

```python
chunks = [
  {
    "text": "Aristotle was a Greek philosopher...",
  },
  ...,
  {
    "text": "He was born in 384 BC in Stagira...",
  }
]

ingest_response = client.ingest_chunks(
  chunks=chunks,
  metadata={"title": "Aristotle", "source": "wikipedia"}
)
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      The response from the R2R system after ingesting the chunks.
      ```bash
      {'message': 'Ingest chunks task queued successfully.', 'task_id': '8f27dfca-606d-422d-b73f-2d9e138661c3', 'document_id': 'd4391abf-8a4e-5d9d-80fd-232ef6fd8527'}
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="chunks" type="list[dict]" required>
  A list of chunk dictionaries to ingest. Each dictionary should contain at least a "text" key with the chunk text. An optional "metadata" key can contain a dictionary of metadata for the chunk.
</ParamField>

<ParamField path="document_id" type="Optional[UUID]">
  An optional document ID to assign to the ingested chunks. If not provided, a new document ID will be generated.
</ParamField>

<ParamField path="metadata" type="Optional[dict]">
  An optional metadata dictionary for the document.
</ParamField>

<ParamField path="run_with_orchestration" type="Optional[bool]">
  Whether or not ingestion runs with [orchestration](/cookbooks/orchestration), default is `True`. When set to `False`, the ingestion process will run synchronous and directly return the result.
</ParamField>


### Update Files

Update existing documents while maintaining version history:

```python
# Basic update with new metadata
update_response = client.update_files(
    file_paths=file_paths,
    document_ids=document_ids,
    metadatas=[{
        "status": "reviewed"
    }]
)

# Update with custom chunking
update_response = client.update_files(
    file_paths=file_paths,
    document_ids=document_ids,
    ingestion_config={
        "chunking_strategy": "by_title",
        "max_characters": 1024  # Larger chunks for this version
    }
)
```

The ingestion configuration can be customized analogously to the ingest files endpoint above.
<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      The response from the R2R system after updating the files.
      ```bash
      [{'message': 'Update files task queued successfully.', 'task_id': '6e27dfca-606d-422d-b73f-2d9e138661b4', 'document_id': '9f375ce9-efe9-5b57-8bf2-a63dee5f3621'}, ...]
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="file_paths" type="list[str]" required>
  A list of file paths to update.
</ParamField>

<ParamField path="document_ids" type="Optional[list[Union[UUID, str]]]" required>
  A list of document IDs corresponding to the files being updated. When not provided, an attempt is made to generate the correct document id from the given user id and file path.
</ParamField>

<ParamField path="metadatas" type="Optional[list[dict]]">
  An optional list of metadata dictionaries for the updated files.
</ParamField>


<ParamField path="ingestion_config" type="Optional[Union[dict, IngestionConfig]]">
  The ingestion config override parameter enables developers to customize their R2R chunking strategy at runtime. Learn more about [configuration here](/documentation/configuration/ingestion/parsing_and_chunking).
  <Expandable title="Other Provider Options">
    <ParamField path="provider" type="str" default="r2r">
      Which R2R ingestion provider to use. Options are "r2r".
    </ParamField>
    <ParamField path="chunking_strategy" type="str" default="recursive">
      Only `recursive` is currently supported.
    </ParamField>
    <ParamField path="chunk_size" type="number" default="1_024">
      The target size for output chunks.
    </ParamField>
    <ParamField path="chunk_overlap" type="number" default="512">
      The target overlap fraction for output chunks
    </ParamField>
    <ParamField path="parser_overrides" type="dict[str, str]" >
      Dictionary of filetypes and selected override parsers. Currently only `{"pdf": "zerox"}` is supported.
    </ParamField>
  </Expandable>

  <Expandable title="Unstructured Provider Options">
    <ParamField path="provider" type="str" default="unstructured_local">
      Which unstructured ingestion provider to use. Options are "unstructured_local", or "unstructured_api".
    </ParamField>

    <ParamField path="max_chunk_size" type="Optional[int]" default="None">
      Sets a maximum size on output chunks.
    </ParamField>

    <ParamField path="combine_under_n_chars" type="Optional[int]">
      Combine chunks smaller than this number of characters.
    </ParamField>

    <ParamField path="max_characters" type="Optional[int]">
      Maximum number of characters per chunk.
    </ParamField>

    <ParamField path="coordinates" type="bool" default="False">
      Whether to include coordinates in the output.
    </ParamField>

    <ParamField path="encoding" type="Optional[str]">
      Encoding to use for text files.
    </ParamField>

    <ParamField path="extract_image_block_types" type="Optional[list[str]]">
      Types of image blocks to extract.
    </ParamField>

    <ParamField path="gz_uncompressed_content_type" type="Optional[str]">
      Content type for uncompressed gzip files.
    </ParamField>

    <ParamField path="hi_res_model_name" type="Optional[str]">
      Name of the high-resolution model to use.
    </ParamField>

    <ParamField path="include_orig_elements" type="Optional[bool]" default="False">
      Whether to include original elements in the output.
    </ParamField>

    <ParamField path="include_page_breaks" type="bool">
      Whether to include page breaks in the output.
    </ParamField>

    <ParamField path="languages" type="Optional[list[str]]">
      List of languages to consider for text processing.
    </ParamField>

    <ParamField path="multipage_sections" type="bool" default="True">
      Whether to allow sections to span multiple pages.
    </ParamField>

    <ParamField path="new_after_n_chars" type="Optional[int]">
      Start a new chunk after this many characters.
    </ParamField>

    <ParamField path="ocr_languages" type="Optional[list[str]]">
      Languages to use for OCR.
    </ParamField>

    <ParamField path="output_format" type="str" default="application/json">
      Format of the output.
    </ParamField>

    <ParamField path="overlap" type="int" default="0">
      Number of characters to overlap between chunks.
    </ParamField>

    <ParamField path="overlap_all" type="bool" default="False">
      Whether to overlap all chunks.
    </ParamField>

    <ParamField path="pdf_infer_table_structure" type="bool" default="True">
      Whether to infer table structure in PDFs.
    </ParamField>

    <ParamField path="similarity_threshold" type="Optional[float]">
      Threshold for considering chunks similar.
    </ParamField>

    <ParamField path="skip_infer_table_types" type="Optional[list[str]]">
      Types of tables to skip inferring.
    </ParamField>

    <ParamField path="split_pdf_concurrency_level" type="int" default="5">
      Concurrency level for splitting PDFs.
    </ParamField>

    <ParamField path="split_pdf_page" type="bool" default="True">
      Whether to split PDFs by page.
    </ParamField>

    <ParamField path="starting_page_number" type="Optional[int]">
      Page number to start processing from.
    </ParamField>

    <ParamField path="strategy" type="str" default="auto">
      Strategy for processing. Options are "auto", "fast", or "hi_res".
    </ParamField>

    <ParamField path="chunking_strategy" type="Optional[str]" default="by_title">
      Strategy for chunking. Options are "by_title" or "basic".
    </ParamField>

    <ParamField path="unique_element_ids" type="bool" default="False">
      Whether to generate unique IDs for elements.
    </ParamField>

    <ParamField path="xml_keep_tags" type="bool" default="False">
      Whether to keep XML tags in the output.
    </ParamField>
  </Expandable>

  <ParamField path="run_with_orchestration" type="Optional[bool]">
    Whether or not ingestion runs with orchestration, default is `True`. When set to `False`, the ingestion process will run synchronous and directly return the result.
  </ParamField>
</ParamField>


### Update Chunks

Update the content of an existing chunk in your R2R system:

```python
document_id = "9fbe403b-c11c-5aae-8ade-ef22980c3ad1"
extraction_id = "aeba6400-1bd0-5ee9-8925-04732d675434"

update_response = client.update_chunks(
    document_id=document_id,
    extraction_id=extraction_id,
    text="Updated chunk content...",
    metadata={"source": "manual_edit", "edited_at": "2024-10-24"}
)
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      The response from the R2R system after updating the chunk.
      ```bash
      {
        'message': 'Update chunk task queued successfully.',
        'task_id': '7e27dfca-606d-422d-b73f-2d9e138661b4',
        'document_id': '9fbe403b-c11c-5aae-8ade-ef22980c3ad1'
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="document_id" type="UUID" required>
  The ID of the document containing the chunk to update.
</ParamField>

<ParamField path="extraction_id" type="UUID" required>
  The ID of the specific chunk to update.
</ParamField>

<ParamField path="text" type="str" required>
  The new text content to replace the existing chunk text.
</ParamField>

<ParamField path="metadata" type="Optional[dict]">
  An optional metadata dictionary for the updated chunk. If provided, this will replace the existing chunk metadata.
</ParamField>

<ParamField path="run_with_orchestration" type="Optional[bool]">
  Whether or not the update runs with orchestration, default is `True`. When set to `False`, the update process will run synchronous and directly return the result.
</ParamField>


### Documents Overview

Retrieve high-level document information. Results are restricted to the current user's files, unless the request is made by a superuser, in which case results from all users are returned:

```python
documents_overview = client.documents_overview()
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="list[dict]">
      A list of dictionaries containing document information.
      ```bash
      [
        {
          'document_id': '9fbe403b-c11c-5aae-8ade-ef22980c3ad1',
          'version': 'v0',
          'collection_ids': [],
          'ingestion_status': 'success',
          'restructuring_status': 'pending',
          'user_id': '2acb499e-8428-543b-bd85-0d9098718220',
          'title': 'aristotle.txt',
          'created_at': '2024-07-21T20:09:14.218741Z',
          'updated_at': '2024-07-21T20:09:14.218741Z',
          'metadata': {'title': 'aristotle.txt', 'version': 'v0', 'x': 'y'}
        },
        ...
      ]
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="document_ids" type="Optional[list[Union[UUID, str]]]">
  An optional list of document IDs to filter the overview.
</ParamField>
<ParamField path="offset" type="Optional[int]">
  An optional value to offset the starting point of fetched results, defaults to `0`.
</ParamField>
<ParamField path="limit" type="Optional[int]">
  An optional value to limit the fetched results, defaults to `100`.
</ParamField>


### Document Chunks

Fetch and examine chunks for a particular document. Chunks represent the atomic units of text after processing:

```python
document_id = "9fbe403b-c11c-5aae-8ade-ef22980c3ad1"
chunks = client.document_chunks(document_id)
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="list[dict]">
      A list of dictionaries containing chunk information.
      ```bash
      [
        {
          'text': 'Aristotle[A] (Greek: Ἀριστοτέλης Aristotélēs, pronounced [aristotélɛːs]; 384–322 BC) was an Ancient Greek philosopher and polymath...',
          'user_id': '2acb499e-8428-543b-bd85-0d9098718220',
          'document_id': '9fbe403b-c11c-5aae-8ade-ef22980c3ad1',
          'extraction_id': 'aeba6400-1bd0-5ee9-8925-04732d675434',
          'fragment_id': 'f48bcdad-4155-52a4-8c9d-8ba06e996ba3'
          'metadata': {'title': 'aristotle.txt', 'version': 'v0', 'chunk_order': 0, 'document_type': 'txt', 'unstructured_filetype': 'text/plain', 'unstructured_languages': ['eng'], 'unstructured_parent_id': '971399f6ba2ec9768d2b5b92ab9d17d6', 'partitioned_by_unstructured': True}
        },
        ...
      ]
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>


<ParamField path="document_id" type="str" required>
  The ID of the document to retrieve chunks for.
</ParamField>
<ParamField path="offset" type="Optional[int]">
  An optional value to offset the starting point of fetched results, defaults to `0`.
</ParamField>
<ParamField path="limit" type="Optional[int]">
  An optional value to limit the fetched results, defaults to `100`.
</ParamField>
<ParamField path="include_vectors" type="Optional[bool]">
  An optional value to return the vectors associated with each chunk, defaults to `False`.
</ParamField>


### Delete Documents

Delete a document by its ID:

```python
delete_response = client.delete(
  {
    "document_id":
      {"$eq": "9fbe403b-c11c-5aae-8ade-ef22980c3ad1"}
  }
)
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      The response from the R2R system after successfully deleting the documents.
      ```bash
      {'results': {}}
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="filters" type="list[dict]" required>
  A list of logical filters to perform over input documents fields which identifies the unique set of documents to delete (e.g., `{"document_id": {"$eq": "9fbe403b-c11c-5aae-8ade-ef22980c3ad1"}}`). Logical operations might include variables such as `"user_id"` or  `"title"` and filters like `neq`, `gte`, etc.
</ParamField>


## Vector Index Management

### Create Vector Index

<Note>
Vector indices significantly improve search performance for large collections but add overhead for smaller datasets. Only create indices when working with hundreds of thousands of documents or when search latency is critical.
</Note>

Create a vector index for similarity search:


```python
create_response = client.create_vector_index(
    table_name="vectors",
    index_method="hnsw",
    index_measure="cosine_distance",
    index_arguments={"m": 16, "ef_construction": 64},
    concurrently=True
)
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      The response from the R2R system after creating the vector index.
      ```bash
      {
        'message': 'Vector index creation task queued successfully.',
        'task_id': '7d38dfca-606d-422d-b73f-2d9e138661b5'
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="table_name" type="str">
  The table to create the index on. Options: vectors, entities_document, entities_collection, communities. Default: vectors
</ParamField>

<ParamField path="index_method" type="str">
  The indexing method to use. Options: hnsw, ivfflat, auto. Default: hnsw
</ParamField>

<ParamField path="index_measure" type="str">
  Distance measure for vector comparisons. Options: cosine_distance, l2_distance, max_inner_product. Default: cosine_distance
</ParamField>

<ParamField path="index_arguments" type="Optional[dict]">
  Configuration parameters for the chosen index method.
  <Expandable title="Configuration Options">
    <ParamField path="HNSW Parameters">
      <ul>
        <li>m (int): Number of connections per element</li>
        <li>ef_construction (int): Size of the dynamic candidate list for construction</li>
      </ul>
    </ParamField>
    <ParamField path="IVFFlat Parameters">
      <ul>
        <li>n_lists (int): Number of clusters/inverted lists</li>
      </ul>
    </ParamField>
  </Expandable>
</ParamField>

<ParamField path="index_name" type="Optional[str]">
  Custom name for the index. If not provided, one will be auto-generated
</ParamField>

<ParamField path="concurrently" type="bool">
  Whether to create the index concurrently. Default: True
</ParamField>

#### Important Considerations

Vector index creation requires careful planning and consideration of your data and performance requirements. Keep in mind:

**Resource Intensive Process**
- Index creation can be CPU and memory intensive, especially for large datasets
- For HNSW indexes, memory usage scales with both dataset size and `m` parameter
- Consider creating indexes during off-peak hours for production systems

**Performance Tuning**
1. **HNSW Parameters:**
   - `m`: Higher values (16-64) improve search quality but increase memory usage and build time
   - `ef_construction`: Higher values increase build time and quality but have diminishing returns past 100
   - Recommended starting point: `m=16`, `ef_construction=64`

```python
# Example balanced configuration
client.create_vector_index(
    table_name="vectors",
    index_method="hnsw",
    index_measure="cosine_distance",
    index_arguments={
        "m": 16,                # Moderate connectivity
        "ef_construction": 64   # Balanced build time/quality
    },
    concurrently=True
)
```
**Pre-warming Required**
- **Important:** Newly created indexes require pre-warming to achieve optimal performance
- Initial queries may be slower until the index is loaded into memory
- The first several queries will automatically warm the index
- For production systems, consider implementing explicit pre-warming by running representative queries after index creation
- Without pre-warming, you may not see the expected performance improvements

**Best Practices**
1. Always use `concurrently=True` in production to avoid blocking other operations
2. Monitor system resources during index creation
3. Test index performance with representative queries before deploying
4. Consider creating indexes on smaller test datasets first to validate parameters

**Distance Measures**
Choose the appropriate measure based on your use case:
- `cosine_distance`: Best for normalized vectors (most common)
- `l2_distance`: Better for absolute distances
- `max_inner_product`: Optimized for dot product similarity

### List Vector Indices

List existing vector indices for a table:

```python
indices = client.list_vector_indices(table_name="vectors")
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      The response containing the list of indices.
      ```bash
      {
        'indices': [
          {
            'name': 'ix_vector_cosine_ops_hnsw__20241021211541',
            'table': 'vectors',
            'method': 'hnsw',
            'measure': 'cosine_distance'
          },
          ...
        ]
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="table_name" type="str">
  The table to list indices from. Options: vectors, entities_document, entities_collection, communities. Default: vectors
</ParamField>

### Delete Vector Index

Delete a vector index from a table:

```python
delete_response = client.delete_vector_index(
    index_name="ix_vector_cosine_ops_hnsw__20241021211541",
    table_name="vectors",
    concurrently=True
)
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      The response from the R2R system after deleting the vector index.
      ```bash
      {
        'message': 'Vector index deletion task queued successfully.',
        'task_id': '8e49efca-606d-422d-b73f-2d9e138661b6'
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="index_name" type="str" required>
  Name of the index to delete
</ParamField>

<ParamField path="table_name" type="str">
  The table containing the index. Options: vectors, entities_document, entities_collection, communities. Default: vectors
</ParamField>

<ParamField path="concurrently" type="bool">
  Whether to delete the index concurrently. Default: True
</ParamField>


### Troubleshooting Common Issues

#### Ingestion Failures
- Check file permissions and paths
- Verify file formats are supported
- Ensure metadata length matches file_paths length
- Monitor memory usage for large files

#### Chunking Issues
- Large chunks may impact retrieval quality
- Small chunks may lose context
- Adjust overlap for better context preservation

#### Vector Index Performance
- Monitor index creation time
- Check memory usage during creation
- Verify warm-up queries are representative
- Consider index rebuild if quality degrades

---


<Warning>
This feature is currently in beta. Functionality may change, and we value your feedback around these features.
</Warning>

<Note>
Occasionally this SDK documentation falls out of date, cross-check with the automatically generated <a href="/api-reference/introduction"> API Reference documentation </a> for the latest parameters.
</Note>

## Conversation Management

### Get Conversations Overview

Retrieve an overview of existing conversations:

```python
conversation_ids = ['123e4567-e89b-12d3-a456-426614174000', '987f6543-e21b-12d3-a456-426614174000']
offset = 0
limit = 10
overview_response = await client.conversations_overview(conversation_ids, offset, limit)
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      The response containing an overview of conversations.
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="conversation_ids" type="list[str]">
  Optional list of conversation UUIDs to retrieve.
</ParamField>

<ParamField path="offset" type="int">
  The offset to start listing conversations from.
</ParamField>

<ParamField path="limit" type="int">
  The maximum number of conversations to return.
</ParamField>

### Get Conversation

Fetch a specific conversation by its UUID:

```python
conversation_id = '123e4567-e89b-12d3-a456-426614174000'
branch_id = 'optional-branch-id'
conversation = await client.get_conversation(conversation_id, branch_id)
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      The response containing the requested conversation details.
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="conversation_id" type="str" required>
  The UUID of the conversation to retrieve.
</ParamField>

<ParamField path="branch_id" type="str">
  Optional ID of a specific branch to retrieve.
</ParamField>

### Create Conversation

Create a new conversation:

```python
new_conversation = await client.create_conversation()
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      The response containing details of the newly created conversation.
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Add Message

Add a message to an existing conversation:

```python
conversation_id = '123e4567-e89b-12d3-a456-426614174000'
message = Message(text='Hello, world!')
parent_id = '98765432-e21b-12d3-a456-426614174000'
metadata = {'key': 'value'}

add_message_response = await client.add_message(conversation_id, message, parent_id, metadata)
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      The response after adding the message to the conversation.
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="conversation_id" type="str" required>
  The UUID of the conversation to add the message to.
</ParamField>

<ParamField path="message" type="Message" required>
  The message object to add to the conversation.
</ParamField>

<ParamField path="parent_id" type="str">
  An optional UUID of the parent message.
</ParamField>

<ParamField path="metadata" type="dict">
  An optional metadata dictionary for the message.
</ParamField>

### Update Message

Update an existing message in a conversation:

```python
message_id = '98765432-e21b-12d3-a456-426614174000'
updated_message = Message(text='Updated message content')

update_message_response = await client.update_message(message_id, updated_message)
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      The response after updating the message.
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="message_id" type="str" required>
  The UUID of the message to update.
</ParamField>

<ParamField path="message" type="Message" required>
  The updated message object.
</ParamField>

### Get Branches Overview

Retrieve an overview of branches in a conversation:

```python
conversation_id = '123e4567-e89b-12d3-a456-426614174000'
branches_overview = await client.branches_overview(conversation_id)
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      The response containing an overview of branches in the conversation.
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="conversation_id" type="str" required>
  The UUID of the conversation to get branches for.
</ParamField>

### Delete Conversation

Delete a conversation by its UUID:

```python
conversation_id = '123e4567-e89b-12d3-a456-426614174000'
delete_response = await client.delete_conversation(conversation_id)
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      The response after deleting the conversation.
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="conversation_id" type="str" required>
  The UUID of the conversation to delete.
</ParamField>

---


## Create a graph

Creating a graph on your documents.


```python
client.create_graph(
    collection_id='122fdf6a-e116-546b-a8f6-e4cb2e2c0a09', # optional
    run_type="run", # estimate or run
    kg_creation_settings={
        "force_kg_creation": True,
        "kg_triples_extraction_prompt": "graphrag_triples_extraction_few_shot",
        "entity_types": [],
        "relation_types": [],
        "extraction_merge_count": 4,
        "max_knowledge_triples": 100,
        "max_description_input_length": 65536,
        "generation_config": {
            "model": "openai/gpt-4o-mini",
            # other generation config params
        }
    }
)
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      The response from the R2R system after creating the graph.
      ```bash
      {'message': 'Graph creation task queued successfully.', 'task_id': '6e27dfca-606d-422d-b73f-2d9e138661b4'}
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="collection_id" type="Optional[Union[UUID, str]]">
  The ID of the collection to create the graph for. If not provided, the graph will be created for the default collection.
</ParamField>

<ParamField path="run_type" type="Optional[Union[str, KGRunType]]">
  The type of run to perform. Options are "estimate" or "run". Estimate will return an estimate of the creation cost, and run will create the graph.
</ParamField>


<ParamField path="kg_creation_settings" type="Optional[Union[dict, KGCreationSettings]]">
  The settings for the graph creation process.
  <Expandable title="KGCreationSettings">
    <ParamField path="kg_triples_extraction_prompt" type="str">
      The prompt to use for triples extraction.
    </ParamField>
    <ParamField path="entity_types" type="list[str]">
      The entity types to extract. If not provided, all entity types will be extracted.
    </ParamField>
    <ParamField path="relation_types" type="list[str]">
      The relation types to extract. If not provided, all relation types will be extracted.
    </ParamField>
    <ParamField path="extraction_merge_count" type="int">
      The number of chunks to merge into a single extraction.
    </ParamField>
    <ParamField path="max_knowledge_triples" type="int">
      The maximum number of triples to extract from each chunk.
    </ParamField>
    <ParamField path="max_description_input_length" type="int">
      The maximum length of the description for a node in the graph in characters (and not tokens).

      Used so that we don't hit the input context window of the LLM while generating descriptions.
    </ParamField>
    <ParamField path="generation_config" type="GenerationConfig">
      The configuration for text generation during graph enrichment.
    </ParamField>
  </Expandable>
</ParamField>

## Enrich a graph

```python
client.enrich_graph(
    collection_id='122fdf6a-e116-546b-a8f6-e4cb2e2c0a09',
    run_type="run",
    kg_enrichment_settings={
        "community_reports_prompt": "graphrag_community_reports",
        "max_summary_input_length": 65536,
        "generation_config": {
            "model": "openai/gpt-4o-mini",
            "temperature": 0.12,
            # other generation config params
        },
        "leiden_params": {
            # leiden algorithm params, all are optional, default values are shown
            "max_cluster_size": 1000,
            "starting_communities": None,
            "extra_forced_iterations": 0,
            "resolution": 1.0,
            "randomness": 0.001,
            "use_modularity": True,
            "random_seed": 7272, # If not set, defaults to 7272
            "weight_attribute": "weight",
            "is_weighted": None,
            "weight_default": 1.0,
            "check_directed": True,
        }
    }
)
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      The response from the R2R system after enriching the graph.
      ```bash
      {'message': 'Graph enrichment task queued successfully.', 'task_id': '6e27dfca-606d-422d-b73f-2d9e138661b4'}
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>


<ParamField path="collection_id" type="Optional[Union[UUID, str]]">
  The ID of the collection to enrich the graph for. If not provided, the graph will be enriched for the default collection.
</ParamField>

<ParamField path="run_type" type="Optional[Union[str, KGRunType]]">
  The type of run to perform. Options are "estimate" or "run". Estimate will return an estimate of the enrichment cost, and run will create the enriched graph.
</ParamField>

<ParamField path="kg_enrichment_settings" type="Optional[Union[dict, KGEnrichmentSettings]]">
  The settings for the graph enrichment process.
  <Expandable title="KGEnrichmentSettings">
    <ParamField path="community_reports_prompt" type="str">
      The prompt to use for community reports.
    </ParamField>
    <ParamField path="max_summary_input_length" type="int">
      The maximum length of the summary input in characters (and not tokens).

      Used so that we don't hit the input context window of the LLM while generating community summaries.
    </ParamField>
    <ParamField path="generation_config" type="GenerationConfig">
      The configuration for text generation during graph enrichment.
    </ParamField>
    <ParamField path="leiden_params" type="dict">
      The parameters for the Leiden algorithm.
      <Expandable title="LeidenParams">
        <ParamField path="max_cluster_size" type="int">
          Default is ``1000``. Any partition or cluster with
          membership >= ``max_cluster_size`` will be isolated into a subnetwork. This
          subnetwork will be used for a new leiden global partition mapping, which will
          then be remapped back into the global space after completion. Once all
          clusters with membership >= ``max_cluster_size`` have been completed, the level
          increases and the partition scheme is scanned again for any new clusters with
          membership >= ``max_cluster_size`` and the process continues until every
          cluster's membership is < ``max_cluster_size`` or if they cannot be broken into
          more than one new community.

        </ParamField>
        <ParamField path="starting_communities" type="Optional[Dict[Any, int]]">
          Default is ``None``. An optional community mapping dictionary that contains a node
          id mapping to the community it belongs to. Please see the Notes section regarding
          node ids used.

          If no community map is provided, the default behavior is to create a node
          community identity map, where every node is in their own community.
        </ParamField>
        <ParamField path="extra_forced_iterations" type="int">
          Default is ``0``. Leiden will run until a maximum quality score has been found
          for the node clustering and no nodes are moved to a new cluster in another
          iteration. As there is an element of randomness to the Leiden algorithm, it is
          sometimes useful to set ``extra_forced_iterations`` to a number larger than 0
          where the entire process is forced to attempt further refinement.
        </ParamField>
        <ParamField path="resolution" type="Union[int, float]">
                Default is ``1.0``. Higher resolution values lead to more communities and lower
        resolution values leads to fewer communities. Must be greater than 0.

        </ParamField>
        <ParamField path="randomness" type="Union[int, float]">
        Default is ``0.001``. The larger the randomness value, the more exploration of
        the partition space is possible. This is a major difference from the Louvain
        algorithm, which is purely greedy in the partition exploration.
        </ParamField>
        <ParamField path="use_modularity" type="bool">
        Default is ``True``. If ``False``, will use a Constant Potts Model (CPM).
        </ParamField>
        <ParamField path="random_seed" type="Optional[int]">
        Default is ``7272``. Can provide an optional seed to the PRNG used in Leiden
        for deterministic output.
        </ParamField>
        <ParamField path="weight_attribute" type="str">
        Default is ``weight``. Only used when creating a weighed edge list of tuples
        when the source graph is a networkx graph. This attribute corresponds to the
        edge data dict key.
        </ParamField>
        <ParamField path="is_weighted" type="Optional[bool]">
        Default is ``None``. Only used when creating a weighted edge list of tuples
        when the source graph is an adjacency matrix. The
        :func:`graspologic.utils.is_unweighted` function will scan these
        matrices and attempt to determine whether it is weighted or not. This flag can
        short circuit this test and the values in the adjacency matrix will be treated
        as weights.
        </ParamField>
        <ParamField path="weight_default" type="Union[int, float]">
        Default is ``1.0``. If the graph is a networkx graph and the graph does not
        have a fully weighted sequence of edges, this default will be used. If the
        adjacency matrix is found or specified to be unweighted, this weight_default
        will be used for every edge.
        </ParamField>
        <ParamField path="check_directed" type="bool">
        Default is ``True``. If the graph is an adjacency matrix, we will attempt to
        ascertain whether it is directed or undirected. As our leiden implementation is
        only known to work with an undirected graph, this function will raise an error
        if it is found to be a directed graph. If you know it is undirected and wish to
        avoid this scan, you can set this value to ``False`` and only the lower triangle
        of the adjacency matrix will be used to generate the weighted edge list.
        </ParamField>
      </Expandable>
    </ParamField>
  </Expandable>
</ParamField>

## Get entities

```python
client.get_entities(
  collection_id='122fdf6a-e116-546b-a8f6-e4cb2e2c0a09',
  offset=0,
  limit=1000,
  entity_ids=None
)
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      The response from the R2R system containing the list of entities in the graph. Total entries is the total number of entities in the graph for the collection (not the number of entities returned).
      ```bash
      {
        "results": {
          "entities": [
            {
              "name": "PLATO",
              "id": 2,
              "category": null,
              "description": "Plato was a prominent philosopher in ancient Greece, known for his foundational role in Western philosophy and for being the teacher of Aristotle at the Academy in Athens. His teachings significantly influenced Aristotle, who later established the Lyceum, reflecting the enduring impact of Plato's philosophical ideas on subsequent educational institutions.",
              "description_embedding": null,
              "community_numbers": null,
              "extraction_ids": [
                "91370d27-31a4-5f6e-8a0c-2c943680fd78",
                "695fa5b3-e416-5608-ba52-3b8b0a66bf3a",
                "85e39f16-26b8-5fd3-89be-e9592864bb46",
                "42c9e012-08e4-54d5-8df9-a8cb9fe277c9"
              ],
              "collection_id": null,
              "document_id": "32b6a70f-a995-5c51-85d2-834f06283a1e",
              "attributes": null
            },
            ... # more entities
          ],
          "total_entries": 2
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="collection_id" type="Optional[Union[UUID, str]]">
  The ID of the collection to get the entities from. If not provided, the entities will be retrieved from the default collection.
</ParamField>

<ParamField path="offset" type="int">
  The offset for pagination.
</ParamField>


<ParamField path="limit" type="int">
  The limit for pagination.
</ParamField>

<ParamField path="entity_ids" type="Optional[list[str]]">
  The list of entity IDs to filter by.
</ParamField>

## Get triples

```python
client.get_triples(
  collection_id='122fdf6a-e116-546b-a8f6-e4cb2e2c0a09',
  offset=0,
  limit=100,
  entity_names=[],
  triple_ids=None
)
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      The response from the R2R system containing the list of triples in the graph. Total entries is the total number of triples in the graph for the collection (not the number of triples returned).

      ```bash
      {
        'results': {
          'total_entries': 28,
          'triples': [
            {
              'id': 1,
              'subject': 'ARISTOTLE',
              'predicate': 'Father-Son',
              'object': 'NICOMACHUS',
              'weight': 1.0,
              'description': 'Nicomachus was the father of Aristotle, who died when Aristotle was a child. ',
              'predicate_embedding': None,
              'extraction_ids': [],
              'document_id': None,
              'attributes': {}
            },
            ... # more triples
          ]
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>


<ParamField path="collection_id" type="Optional[Union[UUID, str]]">
  The ID of the collection to get the triples from. If not provided, the triples will be retrieved from the default collection.
</ParamField>

<ParamField path="offset" type="Optional[int]">
  The offset for pagination. Defaults to 0.
</ParamField>


<ParamField path="limit" type="Optional[int]">
  The limit for pagination. Defaults to 100.
</ParamField>

<ParamField path="entity_names" type="Optional[list[str]]">
  The list of entity names to filter by. Entities are in all caps. eg. ['ARISTOTLE', 'PLATO']
</ParamField>

<ParamField path="triple_ids" type="Optional[list[str]]">
  The list of triple IDs to filter by.
</ParamField>


## Get Communities

```python
client.get_communities(
  collection_id='122fdf6a-e116-546b-a8f6-e4cb2e2c0a09',
  offset=0,
  limit=100,
  levels=[],
  community_numbers=[],
)
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      The response from the R2R system containing the list of communities in the graph.

      ```bash
      {
        'results':
          {
            'communities': [
            {
              'community_number': 0,
              'level': 0,
              'collection_id': '122fdf6a-e116-546b-a8f6-e4cb2e2c0a09',
              'name': "Aristotle and Plato's Legacy",
              'summary': "This community encompasses key historical figures and institutions such as Aristotle, Plato, and their respective schools, Plato's Academy and the Lyceum. The interconnected relationships highlight the profound influence these entities have had on Western philosophy, education, and intellectual traditions.",
              'findings':
                [
                  "Aristotle, a polymath and philosopher, significantly contributed to various fields including natural sciences and ethics, and his teachings have had a lasting impact on medieval scholarship and modern philosophy. His role as a tutor to Alexander the Great further emphasizes his influence on historical leadership and governance. [Data: Entities (11), Relationships (3, 8, 22, +more)]",
                  "Plato's Academy served as a critical educational institution where Aristotle studied for 20 years, shaping his intellectual development and laying the groundwork for his later contributions to philosophy. The Academy's legacy is marked by its role in fostering critical thought and dialogue in ancient philosophy. [Data: Entities (5), Relationships (2, 17, 26, +more)]",
                  "The Lyceum, founded by Aristotle, became a vital source of knowledge and a hub for philosophical discourse in ancient Athens, influencing both medieval scholarship and the development of Western educational practices. Its establishment marked a significant evolution in the teaching of philosophy. [Data: Entities (7), Relationships (4, 11, 25, +more)]",
                  "Stagira, the birthplace of Aristotle, holds historical significance as a cultural landmark that contributed to the intellectual legacy of Aristotle and, by extension, Western philosophy. This connection underscores the importance of geographical context in the development of philosophical thought. [Data: Entities (3), Relationships (6, +more)]",
                  "The influence of Plato on Aristotle's teachings is evident in the establishment of the Lyceum, which reflects the enduring impact of Plato's philosophical ideas on subsequent educational institutions and the evolution of Western philosophy. [Data: Entities (2), Relationships (13, 26, +more)]"
                ],
                'rating': 9.5,
                'rating_explanation': 'The impact severity rating is exceptionally high due to the foundational role these philosophers and their teachings have played in shaping Western thought and educational systems.',
                'embedding': None,
                'attributes': None
            },
            ... # more communities
          ],
          'total_entries': 3
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="collection_id" type="Optional[Union[UUID, str]]">
  The ID of the collection to get the communities from. If not provided, the communities will be retrieved from the default collection.
</ParamField>

<ParamField path="offset" type="Optional[int]">
  The offset for pagination. Defaults to 0.
</ParamField>


<ParamField path="limit" type="Optional[int]">
  The limit for pagination. Defaults to 100.
</ParamField>


<ParamField path="levels" type="Optional[list[int]]">
  The list of levels to filter by. As output of hierarchical clustering, each community is assigned a level.
</ParamField>


<ParamField path="community_numbers" type="Optional[list[int]]">
  The list of community numbers to filter by.
</ParamField>

## Delete Graph

Delete the graph for a collection using the `delete_graph_for_collection` method.

```python
client.delete_graph_for_collection(
  collection_id='122fdf6a-e116-546b-a8f6-e4cb2e2c0a09',
  cascade=False
)
```

<ParamField path="collection_id" type="Union[UUID, str]">
  The ID of the collection to delete the graph for.
</ParamField>

<ParamField path="cascade" type="bool">
  Whether to cascade the deletion.

  NOTE: Setting this flag to true will delete entities and triples for documents that are shared across multiple collections. Do not set this flag unless you are absolutely sure that you want to delete the entities and triples for all documents in the collection.
</ParamField>

## Get Tuned Prompt

```python
client.get_tuned_prompt(
    prompt_name="graphrag_entity_description",
    collection_id='122fdf6a-e116-546b-a8f6-e4cb2e2c0a09',
    documents_offset=0,
    documents_limit=100,
    chunk_offset=0,
    chunk_limit=100
)
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      The response containing the tuned prompt for GraphRAG.
      ```bash
      {
        "results": {
          "tuned_prompt": "string"
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="prompt_name" type="str">
  The name of the prompt to tune. Valid values include "graphrag_entity_description", "graphrag_triples_extraction_few_shot", and "graphrag_community_reports".
</ParamField>

<ParamField path="collection_id" type="Optional[Union[UUID, str]]">
  The ID of the collection to tune the prompt for. If not provided, the default collection will be used.
</ParamField>

<ParamField path="documents_offset" type="Optional[int]">
  The offset for pagination of documents. Defaults to 0.
</ParamField>

<ParamField path="documents_limit" type="Optional[int]">
  The limit for pagination of documents. Defaults to 100. Controls how many documents are used for tuning.
</ParamField>

<ParamField path="chunk_offset" type="Optional[int]">
  The offset for pagination of chunks within each document. Defaults to 0.
</ParamField>

<ParamField path="chunk_limit" type="Optional[int]">
  The limit for pagination of chunks within each document. Defaults to 100. Controls how many chunks per document are used for tuning.
</ParamField>

The tuning process provides an LLM with chunks from each document in the collection. The relative sample size can therefore be controlled by adjusting the document and chunk limits.

## Deduplicate Entities

```python
client.deduplicate_entities(
    collection_id='122fdf6a-e116-546b-a8f6-e4cb2e2c0a09',
    entity_deduplication_settings=entity_deduplication_settings
)
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      The response from the R2R system after deduplicating the entities.
      ```bash
      {
        "message": "Entity deduplication task queued successfully.",
        "task_id": "6e27dfca-606d-422d-b73f-2d9e138661b4"
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>


<ParamField path="collection_id" type="Union[UUID, str]">
  The ID of the collection to deduplicate entities for.
</ParamField>

<ParamField path="entity_deduplication_settings" type="EntityDeduplicationSettings">
  The settings for the entity deduplication process.
  <Expandable title="EntityDeduplicationSettings">
    <ParamField path="kg_entity_deduplication_type" type="str">
      The type of deduplication to perform. Valid values are "by_name". More deduplication types will be added in the future.
    </ParamField>
    <ParamField path="kg_entity_deduplication_prompt" type="str">
      The prompt to use for entity deduplication.
    </ParamField>
    <ParamField path="generation_config" type="GenerationConfig">
      The configuration for text generation during entity deduplication.
    </ParamField>
    <ParamField path="max_description_input_length" type="int">
      The maximum length of the description for a node in the graph in characters (and not tokens).
      Used so that we don't hit the input context window of the LLM while generating descriptions.
    </ParamField>
  </Expandable>
</ParamField>

## Search and RAG

Please see the [Search and RAG](/documentation/python-sdk/retrieval) documentation for more information on how to perform search and RAG using Knowledge Graphs.

## API Reference

Please see the [API documentation](/api-reference/endpoint/create_graph) for more information on the capabilities of the R2R Graph creation and enrichment API.

---


## Introduction

R2R's `KGProvider` handles the creation, management, and querying of knowledge graphs in your applications. This guide offers an in-depth look at the system's architecture, configuration options, and best practices for implementation.

For a practical, step-by-step guide on implementing knowledge graphs in R2R, including code examples and common use cases, see our [GraphRAG Cookbook](/cookbooks/graphrag).


## Configuration

### Knowledge Graph Configuration

These are located in the `r2r.toml` file, under the `[database]` section.

```toml
[database]
provider = "postgres"
batch_size = 256

  [database.kg_creation_settings]
    kg_triples_extraction_prompt = "graphrag_triples_extraction_few_shot"
    entity_types = ["Person", "Organization", "Location"] # if empty, all entities are extracted
    relation_types = ["works at", "founded by", "invested in"] # if empty, all relations are extracted
    max_knowledge_triples = 100
    fragment_merge_count = 4 # number of fragments to merge into a single extraction
    generation_config = { model = "openai/gpt-4o-mini" } # and other params, model used for triplet extraction

  [database.kg_enrichment_settings]
    max_description_input_length = 65536 # increase if you want more comprehensive descriptions
    max_summary_input_length = 65536 # increase if you want more comprehensive summaries
    generation_config = { model = "openai/gpt-4o-mini" } # and other params, model used for node description and graph clustering
    leiden_params = {}
```


Environment variables take precedence over the config settings in case of conflicts. The R2R Docker includes configuration options that facilitate integration with a combined Postgres+pgvector database setup.


## Implementation Guide

### File Ingestion and Graph Construction

```python
from r2r import R2RClient

client = R2RClient("http://localhost:7272")

result = client.ingest_files(["path/to/your/file.txt"])

# following will create a graph on all ingested files
document_ids = [] # add document ids that you want to create a graph on
creation_result = client.create_graph(document_ids)
print(f"Creation Result: {creation_result}")
# wait for the creation to complete

enrichment_result = client.enrich_graph() # enrichment will run on all nodes in the graph
print(f"Enrichment Result: {enrichment_result}")
# wait for the enrichment to complete
```

### Graph-based Search

There are two types of graph-based search: `local` and `global`.

- `local` search is faster and more accurate, but it is not as comprehensive as `global` search.
- `global` search is slower and more comprehensive, but it will give you the most relevant results. Note that global search may perform a large number of LLM calls.

```python
search_result = client.search(
    query="Find founders who worked at Google",
    kg_search_settings={"use_kg_search":True, "kg_search_type": "local"}
)
print(f"Search Result: {search_result}")
```

### Retrieval-Augmented Generation

```python
rag_result = client.rag(
    query="Summarize the achievements of founders who worked at Google",
    kg_search_settings={"use_kg_search":True, "kg_search_type": "local"}
)
print(f"RAG Result: {rag_result}")
```

## Best Practices

1. **Optimize Chunk Size**: Adjust the `chunk_size` based on your data and model capabilities.
2. **Use Domain-Specific Entity Types and Relations**: Customize these for more accurate graph construction.
3. **Balance Batch Size**: Adjust `batch_size` for optimal performance and resource usage.
4. **Implement Caching**: Cache frequently accessed graph data for improved performance.
5. **Regular Graph Maintenance**: Periodically clean and optimize your knowledge graph.

## Advanced Topics

### Custom Knowledge Graph Providers

Extend the `KGProvider` class to implement custom knowledge graph providers:

```python
from r2r.base import KGProvider, KGConfig

class CustomKGProvider(KGProvider):
    def __init__(self, config: KGConfig):
        super().__init__(config)
        # Custom initialization...

    def ingest_files(self, file_paths: List[str]):
        # Custom implementation...

    def search(self, query: str, use_kg_search: bool = True):
        # Custom implementation...

    # Implement other required methods...
```

### Integrating External Graph Databases

To integrate with external graph databases:

1. Implement a custom `KGProvider`.
2. Handle data synchronization between R2R and the external database.
3. Implement custom querying methods to leverage the external database's features.

### Scaling Knowledge Graphs

For large-scale applications:

1. Implement graph partitioning for distributed storage and processing.
2. Use graph-specific indexing techniques for faster querying.
3. Consider using a graph computing framework for complex analytics.

## Troubleshooting

Common issues and solutions:

1. **Ingestion Errors**: Check file formats and encoding.
2. **Query Performance**: Optimize graph structure and use appropriate indexes.
3. **Memory Issues**: Adjust batch sizes and implement pagination for large graphs.

## Conclusion

R2R's Knowledge Graph system provides a powerful foundation for building applications that require structured data representation and complex querying capabilities. By understanding its components, following best practices, and leveraging its flexibility, you can create sophisticated information retrieval and analysis systems tailored to your specific needs.

For further customization and advanced use cases, refer to the [R2R API Documentation](/api-reference) and the [GraphRAG Cookbook](/cookbooks/graphrag).

---


## Introduction

R2R's `CryptoProvider` and `AuthProvider` combine to handle user authentication and cryptographic operations in your applications. This guide offers an in-depth look at the system's architecture, configuration options, and best practices for implementation.

For a practical, step-by-step guide on implementing authentication in R2R, including code examples and common use cases, see our [User Auth Cookbook](/cookbooks/user-auth).

<Note>
When authentication is not required (require_authentication is set to false, which is the default in `r2r.toml`), unauthenticated requests will default to using the credentials of the default admin user.

This behavior ensures that operations can proceed smoothly in development or testing environments where authentication may not be enforced, but it should be used with caution in production settings.
</Note>

## Architecture Overview

R2R's Crypto & Auth system is built on two primary components:

1. **Authentication Provider**: Handles user registration, login, token management, and related operations.
2. **Cryptography Provider**: Manages password hashing, verification, and generation of secure tokens.

These providers work in tandem to ensure secure user management and data protection.

## Providers

### Authentication Provider

The default `R2RAuthProvider` offers a complete authentication solution.

Key features:
- JWT-based access and refresh tokens
- User registration and login
- Email verification (optional)
- Password reset functionality
- Superuser capabilities

### Cryptography Provider

The default `BCryptProvider` handles cryptographic operations.

Key features:
- Secure password hashing using bcrypt
- Password verification
- Generation of cryptographically secure verification codes

## Configuration

### Authentication Configuration

```toml
[auth]
provider = "r2r"
access_token_minutes_lifetime = 60
access_token_days_lifetime = 7
require_authentication = true
require_email_verification = false
default_admin_email = "admin@example.com"
default_admin_password = "change_me_immediately"
```

### Cryptography Configuration

```toml
[crypto]
provider = "bcrypt"
salt_rounds = 12
```

## Secret Key Management

R2R uses a secret key for JWT signing. Generate a secure key using:

```bash
r2r generate-private-key
```

Set the key as an environment variable:

```bash
export R2R_SECRET_KEY=your_generated_key
```

<Warning>
Never commit your secret key to version control. Use environment variables or secure key management solutions in production.
</Warning>

## Auth Service Endpoints

The AuthProvider is responsible for providing functionality to support these core endpoints in R2R:

1. `register`: User registration
2. `login`: User authentication
3. `refresh_access_token`: Token refresh
4. `logout`: Session termination
5. `user`: Retrieve user data
6. `change_password`: Update user password
7. `request_password_reset`: Initiate password reset
8. `confirm_password_reset`: Complete password reset
9. `verify_email`: Email verification
10. `get_user_profile`: Fetch user profile
11. `update_user`: Modify user profile
12. `delete_user_account`: Account deletion

## Implementation Guide

### User Registration

```python
from r2r import R2RClient, UserCreate

client = R2RClient("http://localhost:7272")

result = client.register(user@example.com, secure_password123)
print(f"Registration Result: {result}")
```

### User Authentication

```python
login_result = client.login("user@example.com", "secure_password123")
client.access_token = login_result['results']['access_token']['token']
client.refresh_token = login_result['results']['refresh_token']['token']
```

### Making Authenticated Requests

```python
user = client.user()
print(f"Authenticated User Info: {user}")
```

### Token Refresh

```python
refresh_result = client.refresh_access_token()
client.access_token = refresh_result['results']['access_token']['token']
```

### Logout

```python
logout_result = client.logout()
print(f"Logout Result: {logout_result}")
```

## Security Best Practices

1. **HTTPS**: Always use HTTPS in production.
2. **Authentication Requirement**: Set `require_authentication` to `true` in production.
3. **Email Verification**: Enable `require_email_verification` for enhanced security.
4. **Password Policy**: Implement and enforce strong password policies.
5. **Rate Limiting**: Implement rate limiting on authentication endpoints.
6. **Token Management**: Implement secure token storage and transmission.
7. **Regular Audits**: Conduct regular security audits of your authentication system.


## Custom Authentication Flows and External Identity Providers in R2R

### Custom Authentication Flows

To implement custom authentication flows in R2R, you can extend the `AuthProvider` abstract base class. This allows you to create tailored authentication methods while maintaining compatibility with the R2R ecosystem.

Here's an example of how to create a custom authentication provider:

```python
from r2r.base import AuthProvider, AuthConfig
from r2r.abstractions.user import User, UserCreate, Token, TokenData
from typing import Dict

class CustomAuthProvider(AuthProvider):
    def __init__(self, config: AuthConfig):
        super().__init__(config)
        # Initialize any custom attributes or connections here

    def create_access_token(self, data: dict) -> str:
        # Implement custom access token creation logic
        pass

    def create_refresh_token(self, data: dict) -> str:
        # Implement custom refresh token creation logic
        pass

    def decode_token(self, token: str) -> TokenData:
        # Implement custom token decoding logic
        pass

    def user(self, token: str) -> User:
        # Implement custom user info retrieval logic
        pass

    def get_current_active_user(self, current_user: User) -> User:
        # Implement custom active user validation logic
        pass

    def register(self, user: UserCreate) -> Dict[str, str]:
        # Implement custom user registration logic
        pass

    def verify_email(self, verification_code: str) -> Dict[str, str]:
        # Implement custom email verification logic
        pass

    def login(self, email: str, password: str) -> Dict[str, Token]:
        # Implement custom login logic
        pass

    def refresh_access_token(self, user_email: str, refresh_access_token: str) -> Dict[str, str]:
        # Implement custom token refresh logic
        pass

    async def auth_wrapper(self, auth: Optional[HTTPAuthorizationCredentials] = Security(security)) -> User:
        # You can override this method if you need custom authentication wrapper logic
        return await super().auth_wrapper(auth)

    # Add any additional custom methods as needed
    async def custom_auth_method(self, ...):
        # Implement custom authentication logic
        pass
```

### Integrating External Identity Providers

To integrate external identity providers (e.g., OAuth, SAML) with R2R, you can create a custom `AuthProvider` that interfaces with these external services. Here's an outline of how you might approach this:

1. Create a new class that extends `AuthProvider`:

```python
from r2r.base import AuthProvider, AuthConfig
from r2r.abstractions.user import User, UserCreate, Token, TokenData
import some_oauth_library  # Replace with actual OAuth library

class OAuthAuthProvider(AuthProvider):
    def __init__(self, config: AuthConfig):
        super().__init__(config)
        self.oauth_client = some_oauth_library.Client(
            client_id=config.oauth_client_id,
            client_secret=config.oauth_client_secret
        )

    async def login(self, email: str, password: str) -> Dict[str, Token]:
        # Instead of password-based login, initiate OAuth flow
        auth_url = self.oauth_client.get_authorization_url()
        # Return auth_url or handle redirect as appropriate for your app
        pass

    async def oauth_callback(self, code: str) -> Dict[str, Token]:
        # Handle OAuth callback
        token = await self.oauth_client.get_access_token(code)
        user_data = await self.oauth_client.get_user_info(token)

        # Map external user data to R2R's user model
        r2r_user = self._map_oauth_user_to_r2r_user(user_data)

        # Create and return R2R tokens
        access_token = self.create_access_token({"sub": r2r_user.email})
        refresh_token = self.create_refresh_token({"sub": r2r_user.email})
        return {
            "access_token": Token(token=access_token, token_type="access"),
            "refresh_token": Token(token=refresh_token, token_type="refresh"),
        }

    def _map_oauth_user_to_r2r_user(self, oauth_user_data: dict) -> User:
        # Map OAuth user data to R2R User model
        return User(
            email=oauth_user_data["email"],
            # ... map other fields as needed
        )

    # Implement other required methods...
```

2. Update your R2R configuration to use the new provider:

```python
from r2r import R2R
from r2r.base import AuthConfig
from .custom_auth import OAuthAuthProvider

auth_config = AuthConfig(
    provider="custom_oauth",
    oauth_client_id="your_client_id",
    oauth_client_secret="your_client_secret",
    # ... other config options
)

r2r = R2R(
    auth_provider=OAuthAuthProvider(auth_config),
    # ... other R2R configuration
)
```

3. Implement necessary routes in your application to handle OAuth flow:

```python
from fastapi import APIRouter, Depends
from r2r import R2R

router = APIRouter()

@router.get("/login")
async def login():
    return await r2r.auth_provider.login(None, None)  # Initiate OAuth flow

@router.get("/oauth_callback")
async def oauth_callback(code: str):
    return await r2r.auth_provider.oauth_callback(code)
```

Remember to handle error cases, token storage, and user session management according to your application's needs and the specifics of the external identity provider you're integrating with.

This approach allows you to leverage R2R's authentication abstractions while integrating with external identity providers, giving you flexibility in how you manage user authentication in your application.

### Integrating External Identity Providers

To integrate with external identity providers (e.g., OAuth, SAML):

1. Implement a custom `AuthProvider`.
2. Handle token exchange and user profile retrieval.
3. Map external user data to R2R's user model.

### Scaling Authentication

For high-traffic applications:

1. Implement token caching (e.g., Redis).
2. Consider microservices architecture for auth services.
3. Use database replication for read-heavy operations.

## Troubleshooting

Common issues and solutions:

1. **Token Expiration**: Ensure proper token refresh logic.
2. **CORS Issues**: Configure CORS settings for cross-origin requests.
3. **Password Reset Failures**: Check email configuration and token expiration settings.

## Performance Considerations

1. **BCrypt Rounds**: Balance security and performance when setting `salt_rounds`.
2. **Database Indexing**: Ensure proper indexing on frequently queried user fields.
3. **Caching**: Implement caching for frequently accessed user data.

## Conclusion

R2R's Crypto & Auth system provides a solid foundation for building secure, scalable applications. By understanding its components, following best practices, and leveraging its flexibility, you can create robust authentication systems tailored to your specific needs.

For further customization and advanced use cases, refer to the [R2R API Documentation](/api-reference) and [configuration guide](/documentation/deep-dive/main/config).

---

# Embeddings in R2R

## Introduction

R2R supports multiple Embedding providers, offering flexibility in choosing and switching between different models based on your specific requirements. This guide provides an in-depth look at configuring and using various Embedding providers within the R2R framework.

For a quick start on basic configuration, including embedding setup, please refer to our [configuration guide](/documentation/configuration).

## Providers

R2R currently supports the following cloud embedding providers:
- OpenAI
- Azure
- Cohere
- HuggingFace
- Bedrock (Amazon)
- Vertex AI (Google)
- Voyage AI

And for local inference:
- Ollama
- SentenceTransformers

## Configuration

Update the `embedding` section in your `r2r.toml` file to configure your embedding provider. Here are some example configurations:

<AccordionGroup>

<Accordion icon="openai" title="LiteLLM (Default)">
The default R2R configuration uses LiteLLM to communicate with OpenAI:

```toml
[embedding]
provider = "litellm"
base_model = "text-embedding-3-small"
base_dimension = 512
batch_size = 128
add_title_as_prefix = false
rerank_model = "None"
concurrent_request_limit = 256

[embedding.text_splitter]
type = "recursive_character"
chunk_size = 512
chunk_overlap = 20
```
</Accordion>

<Accordion icon="openai" title="OpenAI">
Here is how to configure R2R to communicate with OpenAI via their client:

```toml
[embedding]
provider = "openai"
base_model = "text-embedding-3-small"
base_dimension = 512
batch_size = 128
add_title_as_prefix = false
rerank_model = "None"
concurrent_request_limit = 256

[embedding.text_splitter]
type = "recursive_character"
chunk_size = 512
chunk_overlap = 20
```
</Accordion>


<Accordion icon="llama" title="Ollama (Local)">
For local embedding generation using Ollama:

```toml
[embedding]
provider = "ollama"
base_model = "mxbai-embed-large"
base_dimension = 1024
batch_size = 32
concurrent_request_limit = 32

[embedding.text_splitter]
type = "recursive_character"
chunk_size = 512
chunk_overlap = 20
```
</Accordion>


</AccordionGroup>

## Selecting Different Embedding Providers

R2R supports a wide range of embedding providers through LiteLLM. Here's how to configure and use them:

<Tabs>
  <Tab title="OpenAI">
    ```python
    export OPENAI_API_KEY=your_openai_key
    # Update r2r.toml:
    # "provider": "litellm" | "openai"
    # "base_model": "text-embedding-3-small"
    # "base_dimension": 512
    r2r serve --config-path=r2r.toml
    ```
    Supported models include:
    - text-embedding-3-small
    - text-embedding-3-small
    - text-embedding-ada-002
  </Tab>

  <Tab title="Azure">
    ```python
    export AZURE_API_KEY=your_azure_api_key
    export AZURE_API_BASE=your_azure_api_base
    export AZURE_API_VERSION=your_azure_api_version
    # Update r2r.toml:
    # "provider": "litellm"
    # "base_model": "azure/<your deployment name>"
    # "base_dimension": XXX
    r2r serve --config-path=r2r.toml
    ```
    Supported models include:
    - text-embedding-ada-002
  </Tab>

  <Tab title="Cohere">
    ```python
    export COHERE_API_KEY=your_cohere_api_key
    # Update r2r.toml:
    # "provider": "litellm"
    # "base_model": "embed-english-v3.0"
    # "base_dimension": 1024
    r2r serve --config-path=r2r.toml
    ```
    Supported models include:
    - embed-english-v3.0
    - embed-english-light-v3.0
    - embed-multilingual-v3.0
    - embed-multilingual-light-v3.0
  </Tab>

  <Tab title="Ollama">
    ```toml r2r.toml
    [embedding]
    provider = "ollama"
    base_model = "mxbai-embed-large"
    base_dimension = 1024
    batch_size = 32
    concurrent_request_limit = 32
    ```
    Deploy your R2R server with:
    ```
    r2r serve --config-path=r2r.toml
    ```
  </Tab>

  <Tab title="HuggingFace">
    ```python
    export HUGGINGFACE_API_KEY=your_huggingface_api_key
    # Update r2r.toml:
    # "provider": "litellm"
    # "base_model": "huggingface/microsoft/codebert-base"
    # "base_dimension": 768
    r2r serve --config-path=r2r.toml
    ```
    LiteLLM supports all Feature-Extraction Embedding models on HuggingFace.
  </Tab>

  <Tab title="Bedrock">
    ```python
    export AWS_ACCESS_KEY_ID=your_access_key
    export AWS_SECRET_ACCESS_KEY=your_secret_key
    export AWS_REGION_NAME=your_region_name
    # Update r2r.toml:
    # "provider": "litellm"
    # "base_model": "amazon.titan-embed-text-v1"
    # "base_dimension": 1024
    r2r serve --config-path=r2r.toml
    ```
    Supported models include:
    - amazon.titan-embed-text-v1
    - cohere.embed-english-v3
    - cohere.embed-multilingual-v3
  </Tab>

  <Tab title="Vertex AI">
    ```python
    export GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json
    export VERTEX_PROJECT=your_project_id
    export VERTEX_LOCATION=your_project_location
    # Update r2r.toml:
    # "provider": "litellm"
    # "base_model": "vertex_ai/textembedding-gecko"
    # "base_dimension": 768
    r2r serve --config-path=r2r.toml
    ```
    Supported models include:
    - textembedding-gecko
    - textembedding-gecko-multilingual
  </Tab>

  <Tab title="Voyage AI">
    ```python
    export VOYAGE_API_KEY=your_voyage_api_key
    # Update r2r.toml:
    # "provider": "litellm"
    # "base_model": "voyage/voyage-01"
    # "base_dimension": 1024
    r2r serve --config-path=r2r.toml
    ```
    Supported models include:
    - voyage-01
    - voyage-lite-01
    - voyage-lite-01-instruct
  </Tab>
</Tabs>

## Embedding Service Endpoints

The EmbeddingProvider is responsible for core functionalities in these R2R endpoints:

1. `update_files`: When updating existing files in the system
2. `ingest_files`: During the ingestion of new files
3. `search`: For embedding search queries
4. `rag`: As part of the Retrieval-Augmented Generation process

Here's how you can use these endpoints with embeddings:

### File Ingestion

```python
from r2r import R2R

app = R2R()

# Ingest a file, which will use the configured embedding model
response = app.ingest_files(["path/to/your/file.txt"])
print(f"Ingestion response: {response}")
```

### Search

```python
# Perform a search, which will embed the query using the configured model
search_results = app.search("Your search query here")
print(f"Search results: {search_results}")
```

### RAG (Retrieval-Augmented Generation)

```python
# Use RAG, which involves embedding for retrieval
rag_response = app.rag("Your question or prompt here")
print(f"RAG response: {rag_response}")
```

### Updating Files

```python
# Update existing files, which may involve re-embedding
update_response = app.update_files(["path/to/updated/file.txt"])
print(f"Update response: {update_response}")
```

<Note>
Remember that you don't directly call the embedding methods in your application code. R2R handles the embedding process internally based on your configuration.
</Note>

## Security Best Practices

1. **API Key Management**: Use environment variables or secure key management solutions for API keys.
2. **Input Validation**: Sanitize and validate all inputs before generating embeddings.
3. **Rate Limiting**: Implement rate limiting to prevent abuse of embedding endpoints.
4. **Monitoring**: Regularly monitor embedding usage for anomalies or misuse.


## Custom Embedding Providers in R2R

You can create custom embedding providers by inheriting from the `EmbeddingProvider` class and implementing the required methods. This allows you to integrate any embedding model or service into R2R.

### Embedding Provider Structure

The Embedding system in R2R is built on two main components:

1. `EmbeddingConfig`: A configuration class for Embedding providers.
2. `EmbeddingProvider`: An abstract base class that defines the interface for all Embedding providers.

### EmbeddingConfig

The `EmbeddingConfig` class is used to configure Embedding providers:

```python
from r2r.base import ProviderConfig
from typing import Optional

class EmbeddingConfig(ProviderConfig):
    provider: Optional[str] = None
    base_model: Optional[str] = None
    base_dimension: Optional[int] = None
    rerank_model: Optional[str] = None
    rerank_dimension: Optional[int] = None
    rerank_transformer_type: Optional[str] = None
    batch_size: int = 1
    prefixes: Optional[dict[str, str]] = None

    def validate(self) -> None:
        if self.provider not in self.supported_providers:
            raise ValueError(f"Provider '{self.provider}' is not supported.")

    @property
    def supported_providers(self) -> list[str]:
        return [None, "openai", "ollama", "sentence-transformers"]
```

### EmbeddingProvider

The `EmbeddingProvider` is an abstract base class that defines the common interface for all Embedding providers:

```python
from abc import abstractmethod
from enum import Enum
from r2r.base import Provider
from r2r.abstractions.embedding import EmbeddingPurpose
from r2r.abstractions.search import VectorSearchResult

class EmbeddingProvider(Provider):
    class PipeStage(Enum):
        BASE = 1
        RERANK = 2

    def __init__(self, config: EmbeddingConfig):
        if not isinstance(config, EmbeddingConfig):
            raise ValueError("EmbeddingProvider must be initialized with a `EmbeddingConfig`.")
        super().__init__(config)

    @abstractmethod
    def get_embedding(
        self,
        text: str,
        stage: PipeStage = PipeStage.BASE,
        purpose: EmbeddingPurpose = EmbeddingPurpose.INDEX,
    ):
        pass

    @abstractmethod
    def get_embeddings(
        self,
        texts: list[str],
        stage: PipeStage = PipeStage.BASE,
        purpose: EmbeddingPurpose = EmbeddingPurpose.INDEX,
    ):
        pass

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[VectorSearchResult],
        stage: PipeStage = PipeStage.RERANK,
        limit: int = 10,
    ):
        pass

    @abstractmethod
    def tokenize_string(
        self, text: str, model: str, stage: PipeStage
    ) -> list[int]:
        pass

    def set_prefixes(self, config_prefixes: dict[str, str], base_model: str):
        # Implementation of prefix setting
        pass
```

### Creating a Custom Embedding Provider

To create a custom Embedding provider, follow these steps:

1. Create a new class that inherits from `EmbeddingProvider`.
2. Implement the required methods: `get_embedding`, `get_embeddings`, `rerank`, and `tokenize_string`.
3. (Optional) Implement async versions of methods if needed.
4. (Optional) Add any additional methods or attributes specific to your provider.

Here's an example of a custom Embedding provider:

```python
import numpy as np
from r2r.base import EmbeddingProvider, EmbeddingConfig
from r2r.abstractions.embedding import EmbeddingPurpose
from r2r.abstractions.search import VectorSearchResult

class CustomEmbeddingProvider(EmbeddingProvider):
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        # Initialize any custom attributes or models here
        self.model = self._load_custom_model(config.base_model)

    def _load_custom_model(self, model_name):
        # Load your custom embedding model here
        pass

    def get_embedding(
        self,
        text: str,
        stage: EmbeddingProvider.PipeStage = EmbeddingProvider.PipeStage.BASE,
        purpose: EmbeddingPurpose = EmbeddingPurpose.INDEX,
    ) -> list[float]:
        # Apply prefix if available
        if purpose in self.prefixes:
            text = f"{self.prefixes[purpose]}{text}"

        # Generate embedding using your custom model
        embedding = self.model.encode(text)
        return embedding.tolist()

    def get_embeddings(
        self,
        texts: list[str],
        stage: EmbeddingProvider.PipeStage = EmbeddingProvider.PipeStage.BASE,
        purpose: EmbeddingPurpose = EmbeddingPurpose.INDEX,
    ) -> list[list[float]]:
        # Apply prefixes if available
        if purpose in self.prefixes:
            texts = [f"{self.prefixes[purpose]}{text}" for text in texts]

        # Generate embeddings in batches
        all_embeddings = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i+self.config.batch_size]
            batch_embeddings = self.model.encode(batch)
            all_embeddings.extend(batch_embeddings.tolist())
        return all_embeddings

    def rerank(
        self,
        query: str,
        results: list[VectorSearchResult],
        stage: EmbeddingProvider.PipeStage = EmbeddingProvider.PipeStage.RERANK,
        limit: int = 10,
    ) -> list[VectorSearchResult]:
        if not self.config.rerank_model:
            return results[:limit]

        # Implement custom reranking logic here
        # This is a simple example using dot product similarity
        query_embedding = self.get_embedding(query, stage, EmbeddingPurpose.QUERY)
        for result in results:
            result.score = np.dot(query_embedding, result.embedding)

        reranked_results = sorted(results, key=lambda x: x.score, reverse=True)
        return reranked_results[:limit]

    def tokenize_string(
        self, text: str, model: str, stage: EmbeddingProvider.PipeStage
    ) -> list[int]:
        # Implement custom tokenization logic
        # This is a simple example using basic string splitting
        return [ord(char) for word in text.split() for char in word]

    # Optionally implement async versions of methods
    async def async_get_embedding(self, text: str, stage: EmbeddingProvider.PipeStage, purpose: EmbeddingPurpose):
        # Implement async version if needed
        return self.get_embedding(text, stage, purpose)

    async def async_get_embeddings(self, texts: list[str], stage: EmbeddingProvider.PipeStage, purpose: EmbeddingPurpose):
        # Implement async version if needed
        return self.get_embeddings(texts, stage, purpose)
```

### Registering and Using the Custom Provider

To use your custom Embedding provider in R2R:

1. Update the `EmbeddingConfig` class to include your custom provider:

```python
class EmbeddingConfig(ProviderConfig):
    # ...existing code...

    @property
    def supported_providers(self) -> list[str]:
        return [None, "openai", "ollama", "sentence-transformers", "custom"]  # Add your custom provider here
```

2. Update your R2R configuration to use the custom provider:

```toml
[embedding]
provider = "custom"
base_model = "your-custom-model"
base_dimension = 768
batch_size = 32

[embedding.prefixes]
index = "Represent this document for retrieval: "
query = "Represent this query for retrieving relevant documents: "
```

3. In your R2R application, register the custom provider:

```python
from r2r import R2R
from r2r.base import EmbeddingConfig
from your_module import CustomEmbeddingProvider

def get_embedding_provider(config: EmbeddingConfig):
    if config.provider == "custom":
        return CustomEmbeddingProvider(config)
    # ... handle other providers ...

r2r = R2R(embedding_provider_factory=get_embedding_provider)
```

Now you can use your custom Embedding provider seamlessly within your R2R application:

```python
# Ingest documents (embeddings will be generated using your custom provider)
r2r.ingest_files(["path/to/document.txt"])

# Perform a search
results = r2r.search("Your search query")

# Use RAG
rag_response = r2r.rag("Your question here")
```

By following this structure, you can integrate any embedding model or service into R2R, maintaining consistency with the existing system while adding custom functionality as needed. This approach allows for great flexibility in choosing or implementing embedding solutions that best fit your specific use case.### Embedding Prefixes

## Embedding Prefixes

R2R supports embedding prefixes to enhance embedding quality for different purposes:

1. **Index Prefixes**: Applied to documents during indexing.
2. **Query Prefixes**: Applied to search queries.

Configure prefixes in your `r2r.toml` or when initializing the EmbeddingConfig.

## Troubleshooting

Common issues and solutions:

1. **API Key Errors**: Ensure your API keys are correctly set and have the necessary permissions.
2. **Dimension Mismatch**: Verify that the `base_dimension` in your config matches the actual output of the chosen model.
3. **Out of Memory Errors**: Adjust the batch size or choose a smaller model if encountering memory issues with local models.

## Performance Considerations

1. **Batching**: Use batching for multiple, similar requests to improve throughput.
2. **Model Selection**: Balance between model capability and inference speed based on your use case.
3. **Caching**: Implement caching strategies to avoid re-embedding identical text.

## Conclusion

R2R's Embedding system provides a flexible and powerful foundation for integrating various embedding models into your applications. By understanding the available providers, configuration options, and best practices, you can effectively leverage embeddings to enhance your R2R-based projects.

For an advanced example of implementing reranking in R2R.

---


# R2R JavaScript SDK Documentation

## Installation

Before starting, make sure you have completed the [R2R installation](/documentation/installation).

Install the R2R JavaScript SDK:

```bash
npm install r2r-js
```

## Getting Started

1. Import the R2R client:

```javascript
const { r2rClient } = require('r2r-js');
```

2. Initialize the client:

```javascript
const client = new r2rClient('http://localhost:7272');
```

3. Check if R2R is running correctly:

```javascript
const healthResponse = await client.health();
// {"status":"ok"}
```

4. Login (Optional):
```javascript
// client.register("me@email.com", "my_password"),
// client.verify_email("me@email.com", "my_verification_code")
client.login("me@email.com", "my_password")
```
When using authentication the commands below automatically restrict the scope to a user's available documents.

## Additional Documentation

For more detailed information on specific functionalities of R2R, please refer to the following documentation:

- [Document Ingestion](/documentation/python-sdk/ingestion): Learn how to add, retrieve, and manage documents in R2R.
- [Search & RAG](/documentation/python-sdk/retrieval): Explore various querying techniques and Retrieval-Augmented Generation capabilities.
- [Authentication](/documentation/python-sdk/auth): Understand how to manage users and implement authentication in R2R.
- [Observability](/documentation/python-sdk/observability): Learn about analytics and monitoring tools for your R2R system.

---


R2R uses telemetry to collect **anonymous** usage information. This data helps us understand how R2R is used, prioritize new features and bug fixes, and improve overall performance and stability.

## Disabling Telemetry

To opt out of telemetry, you can set an environment variable:

```bash
export TELEMETRY_ENABLED=false
```

<Note>
Valid values to disable telemetry are `false`, `0`, or `f`. When telemetry is disabled, no events will be captured.
</Note>

## Collected Information

Our telemetry system collects basic, anonymous information such as:

- **Feature Usage**: Which features are being used and their frequency of use.

## Data Storage

<AccordionGroup>

<Accordion icon="database" title="Telemetry Data Storage">
We use [Posthog](https://posthog.com/) to store and analyze telemetry data. Posthog is an open-source platform for product analytics.

For more information about Posthog:
- Visit their website: [posthog.com](https://posthog.com/)
- Check out their GitHub repository: [github.com/posthog](https://github.com/posthog)
</Accordion>

</AccordionGroup>

## Why We Collect Telemetry

Telemetry data helps us:

1. Understand which features are most valuable to users
2. Identify areas for improvement
3. Prioritize development efforts
4. Enhance R2R's overall performance and stability

We appreciate your participation in our telemetry program, as it directly contributes to making R2R better for everyone.

<Note>
We respect your privacy. All collected data is anonymous and used solely for improving R2R.
</Note>

---


## Introduction

`R2RConfig` uses a TOML-based configuration system to customize various aspects of R2R's functionality. This guide provides a detailed overview of how to configure R2R, including all available options and their meanings.

## Configuration File Structure

The R2R configuration is stored in a TOML file, which defaults to [`r2r.toml`](https://github.com/SciPhi-AI/R2R/blob/main/r2r.toml). The file is divided into several sections, each corresponding to a different aspect of the R2R system:

- Authentication
- Completion (LLM)
- Cryptography
- Database
- Embedding
- Evaluation
- Ingestion
- Knowledge Graph
- Logging
- Prompt Management

## Loading a Configuration

To use a custom configuration, you can load it when initializing R2R:

```python
from r2r import R2RConfig, R2RBuilder

# Load a custom configuration
config = R2RConfig.from_toml("path/to/your/r2r.toml")
r2r = R2RBuilder(config).build()

# Or use the preset configuration
r2r = R2RBuilder().build()
```

## Configuration Sections

### Authentication
Refer to the [`AuthProvider`](/documentation/deep-dive/providers/auth) to learn more about how R2R supports auth providers.

```toml
[auth]
provider = "r2r"
access_token_lifetime_in_minutes = 60
refresh_token_lifetime_in_days = 7
require_authentication = false
require_email_verification = false
default_admin_email = "admin@example.com"
default_admin_password = "change_me_immediately"
```

- `provider`: Authentication provider. Currently, only "r2r" is supported.
- `access_token_lifetime_in_minutes`: Lifespan of access tokens in minutes.
- `refresh_token_lifetime_in_days`: Lifespan of refresh tokens in days.
- `require_authentication`: If true, all secure routes require authentication. Otherwise, non-authenticated requests mock superuser access.
- `require_email_verification`: If true, email verification is required for new accounts.
- `default_admin_email` and `default_admin_password`: Credentials for the default admin account.

### Completion (LLM)

Refer to the [`LLMProvider`](/documentation/deep-dive/providers/llms) to learn more about how R2R supports LLM providers.
```toml

[completion]
provider = "litellm"
concurrent_request_limit = 16

  [completion.generation_config]
  model = "openai/gpt-4o"
  temperature = 0.1
  top_p = 1
  max_tokens_to_sample = 1_024
  stream = false
  add_generation_kwargs = { }
```

- `provider`: LLM provider. Options include "litellm" and "openai".
- `concurrent_request_limit`: Maximum number of concurrent requests allowed.
- `generation_config`: Detailed configuration for text generation.
  - `model`: The specific LLM model to use.
  - `temperature`: Controls randomness in generation (0.0 to 1.0).
  - `top_p`: Parameter for nucleus sampling.
  - `max_tokens_to_sample`: Maximum number of tokens to generate.
  - Other parameters control various aspects of text generation.

### Cryptography

Refer to the [`CryptoProvider`](/documentation/deep-dive/providers/auth) to learn more about how R2R supports cryptography.

```toml
[crypto]
provider = "bcrypt"
```

- `provider`: Cryptography provider for password hashing. Currently, only "bcrypt" is supported.

### Database

Refer to the [`DatabaseProvider`](/documentation/deep-dive/providers/database) to learn more about how R2R supports databases.

```toml
[database]
provider = "postgres"
```

- `provider`: Database provider. Only "postgres" is supported.
- `user`: Default username for accessing database.
- `password`: Default password for accessing database.
- `host`: Default host for accessing database.
- `port`: Default port for accessing database.
- `db_name`: Default db_name for accessing database.

### Embedding

Refer to the [`EmbeddingProvider`](/documentation/deep-dive/providers/embeddings) to learn more about how R2R supports embeddings.

```toml
[embedding]
provider = "litellm"
base_model = "text-embedding-3-small"
base_dimension = 512
batch_size = 128
add_title_as_prefix = false
rerank_model = "None"
concurrent_request_limit = 256
```

- `provider`: Embedding provider. Options include "ollama", "openai" and "sentence-transformers".
- `base_model`: The specific embedding model to use.
- `base_dimension`: Dimension of the embedding vectors.
- `batch_size`: Number of items to process in a single batch.
- `add_title_as_prefix`: Whether to add the title as a prefix to the embedded text.
- `rerank_model`: Model used for reranking, if any.
- `concurrent_request_limit`: Maximum number of concurrent embedding requests.

### Evaluation

```toml
[eval]
provider = "None"
```

- `provider`: Evaluation provider. Set to "None" to disable evaluation functionality.

### Knowledge Graph

Refer to the [`KGProvider`](documentation/deep-dive/providers/knowledge-graph) to learn more about how R2R supports knowledge graphs.

```toml
[database]
provider = "postgres"
batch_size = 1

[database.kg_extraction_config]
model = "gpt-4o"
temperature = 0.1
top_p = 1
max_tokens_to_sample = 1_024
stream = false
add_generation_kwargs = { }
```

- `provider`: Specifies the backend used for storing and querying the knowledge graph. Options include "postgres" and "None".
- `batch_size`: Determines how many text chunks are processed at once for knowledge extraction.
- `kg_extraction_config`: Configures the language model used for extracting knowledge from text chunks.

### Logging

```toml
[logging]
provider = "local"
log_table = "logs"
log_info_table = "log_info"
```

- `provider`: Logging provider. Currently set to "local".
- `log_table`: Name of the table where logs are stored.
- `log_info_table`: Name of the table where log information is stored.

### Prompt Management

```toml
[prompt]
provider = "r2r"
```

- `provider`: Prompt management provider. Currently set to "r2r".

## Advanced Configuration

### Environment Variables

For sensitive information like API keys, it's recommended to use environment variables instead of hardcoding them in the configuration file. R2R will automatically look for environment variables for certain settings.

### Custom Providers

R2R supports custom providers for various components. To use a custom provider, you'll need to implement the appropriate interface and register it with R2R. Refer to the developer documentation for more details on creating custom providers.

### Configuration Validation

R2R performs validation on the configuration when it's loaded. If there are any missing required fields or invalid values, an error will be raised. Always test your configuration in a non-production environment before deploying.

## Best Practices

1. **Security**: Never commit sensitive information like API keys or passwords to version control. Use environment variables instead.
2. **Modularity**: Create separate configuration files for different environments (development, staging, production).
3. **Documentation**: Keep your configuration files well-commented, especially when using custom or non-standard settings.
4. **Version Control**: Track your configuration files in version control, but use `.gitignore` to exclude files with sensitive information.
5. **Regular Review**: Periodically review and update your configuration to ensure it aligns with your current needs and best practices.

## Troubleshooting

If you encounter issues with your configuration:

1. Check the R2R logs for any error messages related to configuration.
2. Verify that all required fields are present in your configuration file.
3. Ensure that the values in your configuration are of the correct type (string, number, boolean, etc.).
4. If using custom providers or non-standard settings, double-check the documentation or consult with the R2R community.

By following this guide, you should be able to configure R2R to suit your specific needs. Remember that R2R is highly customizable, so don't hesitate to explore different configuration options to optimize your setup.

---


<Note>
Occasionally this SDK documentation falls out of date, cross-check with the automatcially generated <a href="/api-reference/introduction"> API Reference documentation </a> for the latest parameters.
</Note>


## AI Powered Search

### Search

Perform a basic vector search:

```python
search_response = client.search("What was Uber's profit in 2020?")
```


<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      The search results from the R2R system.
      ```python
      {
        'results': {
          'vector_search_results': [
            {
                'fragment_id': '13a12fc0-cbce-5e35-b179-d413c15179cb',
                'extraction_id': '2b8ff2e9-c135-573d-bf8a-7a2db60a0a11',
                'document_id': '3e157b3a-8469-51db-90d9-52e7d896b49b',
                'user_id': '2acb499e-8428-543b-bd85-0d9098718220',
                'collection_ids': [],
                'score': 0.7449709925072809,
                'text': 'Net\n loss attributable to Uber Technologies, Inc. was $496 million, a 93% improvement ...',
                'metadata': {'title': 'uber_2021.pdf', 'version': 'v0', 'chunk_order': 5, 'associatedQuery': "What was Uber's profit in 2020?"}
              }, ...
          ],
          'kg_search_results': None
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>


<ParamField path="query" type="str" required>
  The search query.
</ParamField>

<ParamField path="vector_search_settings" type="Optional[Union[VectorSearchSettings, dict]]" default="None">
  Optional settings for vector search, either a dictionary, a `VectorSearchSettings` object, or `None` may be passed. If a dictionary or `None` is passed, then R2R will use server-side defaults for non-specified fields.

  <Expandable title="properties">
    <ParamField path="use_vector_search" type="bool" default="True">
    Whether to use vector search.
    </ParamField>

    <ParamField path="use_hybrid_search" type="bool" default="False">
    Whether to perform a hybrid search (combining vector and keyword search).
    </ParamField>


    <ParamField path="filters" type="dict[str, Any]" default="{}">
      Alias for `search_filters`, now `deprecated`.
    </ParamField>

    <ParamField path="search_filters" type="dict[str, Any]" default="{}">
      Filters to apply to the vector search. Allowed operators include `eq`, `neq`, `gt`, `gte`, `lt`, `lte`, `like`, `ilike`, `in`, and `nin`.

      Commonly seen filters include operations include the following:

        `{"document_id": {"$eq": "9fbe403b-..."}}`

        `{"document_id": {"$in": ["9fbe403b-...", "3e157b3a-..."]}}`

        `{"collection_ids": {"$overlap": ["122fdf6a-...", "..."]}}`

        `{"$and": {"$document_id": ..., "collection_ids": ...}}`
    </ParamField>

    <ParamField path="search_limit" type="int" default="10">
    Maximum number of results to return (1-1000).
    </ParamField>

    <ParamField path="selected_collection_ids" type="list[UUID]" default="[]">
    Group IDs to search for.
    </ParamField>

    <ParamField path="index_measure" type="IndexMeasure" default="cosine_distance">
    The distance measure to use for indexing (cosine_distance, l2_distance, or max_inner_product).
    </ParamField>

    <ParamField path="include_values" type="bool" default="True">
    Whether to include search score values in the search results.
    </ParamField>

    <ParamField path="include_metadatas" type="bool" default="True">
    Whether to include element metadata in the search results.
    </ParamField>

    <ParamField path="probes" type="Optional[int]" default="10">
    Number of ivfflat index lists to query. Higher increases accuracy but decreases speed.
    </ParamField>

    <ParamField path="ef_search" type="Optional[int]" default="40">
    Size of the dynamic candidate list for HNSW index search. Higher increases accuracy but decreases speed.
    </ParamField>

    <ParamField path="hybrid_search_settings" type="Optional[HybridSearchSettings]" default="HybridSearchSettings()">
    Settings for hybrid search.
    <Expandable title="properties">
      <ParamField path="full_text_weight" type="float" default="1.0">
      Weight to apply to full text search.
      </ParamField>

      <ParamField path="semantic_weight" type="float" default="5.0">
      Weight to apply to semantic search.
      </ParamField>

      <ParamField path="full_text_limit" type="int" default="200">
      Maximum number of results to return from full text search.
      </ParamField>

      <ParamField path="rrf_k" type="int" default="50">
      K-value for RRF (Rank Reciprocal Fusion).
      </ParamField>
    </Expandable>
    </ParamField>
  </Expandable>
</ParamField>
<ParamField path="kg_search_settings" type="Optional[Union[KGSearchSettings, dict]]" default="None">
  Optional settings for knowledge graph search, either a dictionary, a `KGSearchSettings` object, or `None` may be passed. If a dictionary or `None` is passed, then R2R will use server-side defaults for non-specified fields.

  <Expandable title="properties">
    <ParamField path="use_kg_search" type="bool" default="False">
    Whether to use knowledge graph search.
    </ParamField>

    <ParamField path="kg_search_type" type="str" default="local">
    Type of knowledge graph search. Can be 'global' or 'local'.
    </ParamField>

    <ParamField path="kg_search_level" type="Optional[str]" default="None">
    Level of knowledge graph search.
    </ParamField>

    <ParamField path="generation_config" type="Optional[GenerationConfig]" default="GenerationConfig()">
    Configuration for knowledge graph search generation.
    </ParamField>

    <ParamField path="entity_types" type="list" default="[]">
    List of entity types to use for knowledge graph search.
    </ParamField>

    <ParamField path="relationships" type="list" default="[]">
    List of relationships to use for knowledge graph search.
    </ParamField>

    <ParamField path="max_community_description_length" type="int" default="65536">
    Maximum length of community descriptions.
    </ParamField>

    <ParamField path="max_llm_queries_for_global_search" type="int" default="250">
    Maximum number of LLM queries for global search.
    </ParamField>

    <ParamField path="local_search_limits" type="dict[str, int]" default="{'__Entity__': 20, '__Relationship__': 20, '__Community__': 20}">
    Limits for local search on different types of elements.
    </ParamField>
  </Expandable>
</ParamField>



### Search custom settings

Learn more about the search [API here](/api-reference/endpoint/search). It allows searching with custom settings, such as bespoke document filters and larger search limits:
```python
# returns only chunks from documents with title `document_title`
filtered_search_response = client.search(
    "What was Uber's profit in 2020?",
    vector_search_settings={
        # restrict results to the Uber document
        "filters": {"title": {"$eq": "uber_2021.pdf"}},
        "search_limit": 100
    }
)
```


### Hybrid Search


Learn more about the dedicated knowledge graph capabilities [in R2R here](/cookbooks/hybrid-search). Combine traditional keyword-based search with vector search:

```python
hybrid_search_response = client.search(
    "What was Uber's profit in 2020?",
    vector_search_settings={
        "use_hybrid_search": True,
        "search_limit": 20,
        "hybrid_search_settings": {
            "full_text_weight": 1.0,
            "semantic_weight": 10.0,
            "full_text_limit": 200,
            "rrf_k": 25,
        },
    }
)
```

### Knowledge Graph Search

Learn more about the dedicated knowledge graph capabilities [in R2R here](/cookbooks/graphrag). You can utilize knowledge graph capabilities to enhance search results, as shown below:

```python
kg_search_response = client.search(
    "What is airbnb",
    vector_search_settings={"use_vector_search": False},
    kg_search_settings={
      "use_kg_search": True,
      "kg_search_type": "local",
      "kg_search_level": "0",
      "generation_config": {
          "model": "openai/gpt-4o-mini",
          "temperature": 0.7,
      },
      "local_search_limits": {
          "__Entity__": 20,
          "__Relationship__": 20,
          "__Community__": 20,
      },
      "max_community_description_length": 65536,
      "max_llm_queries_for_global_search": 250
    }
)
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      The knowledge graph search results from the R2R system.
      ```bash
      {
        "kg_search_results":
        [
          {
            "global_result": None,
            "local_result": {
              "communities": {
                "0": {
                  "summary": {
                    "findings": [
                      {
                        "explanation": "Aristotle is credited with the earliest study of formal logic, and his conception of it was the dominant form of Western logic until the 19th-century advances in mathematical logic. His works compiled into a set of six books ...",
                        "summary": "Aristotle's Foundational Role in Logic"
                      }
                    ],
                    "rating": 9.5,
                    "rating_explanation": "The impact severity rating is high due to Aristotle's foundational influence on multiple disciplines and his enduring legacy in Western philosophy and science.",
                    "summary": "The community revolves around Aristotle, an ancient Greek philosopher and polymath, who made significant contributions to various fields including logic, biology, political science, and economics. His works, such as 'Politics' and 'Nicomachean Ethics', have influenced numerous disciplines and thinkers from antiquity through the Middle Ages and beyond. The relationships between his various works and the fields he contributed to highlight his profound impact on Western thought.",
                    "title": "Aristotle and His Contributions"
                  }
                }
              },
              "entities": {
                "0": {
                  "description": "Aristotle was an ancient Greek philosopher and polymath, recognized as the father of various fields including logic, biology, and political science. He authored significant works such as the *Nicomachean Ethics* and *Politics*, where he explored concepts of virtue, governance, and the nature of reality, while also critiquing Platos ideas. His teachings and observations laid the groundwork for numerous disciplines, influencing thinkers ...",
                  "name": "Aristotle"
                }
              },
              "query": "Who is Aristotle?",
              "relationships": {}
            }
          }
        ],
        "vector_search_results": None
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

## Retrieval-Augmented Generation (RAG)

### Basic RAG

Generate a response using RAG:

```python
rag_response = client.rag("What was Uber's profit in 2020?")
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      The RAG response from the R2R system.
      ```bash
      {
        'results': {
          'completion': {
            'id': 'chatcmpl-9ySStnC0oEhnGPPV1k8ZYnxBKOuW8',
            'choices': [{
              'finish_reason': 'stop',
              'index': 0,
              'logprobs': None,
              'message': {
                'content': "Uber's profit in 2020 was a net loss of $6.77 billion."
              },
              ...
            }]
          },
          'search_results': {
            'vector_search_results': [...],
            'kg_search_results': None
          }
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="query" type="str" required>
  The query for RAG.
</ParamField>

<ParamField path="vector_search_settings" type="Optional[Union[VectorSearchSettings, dict]]" default="None">
  Optional settings for vector search, either a dictionary, a `VectorSearchSettings` object, or `None` may be passed. If a dictionary is used, non-specified fields will use the server-side default.

  <Expandable title="properties">
    <ParamField path="use_vector_search" type="bool" default="True">
    Whether to use vector search.
    </ParamField>

    <ParamField path="use_hybrid_search" type="bool" default="False">
    Whether to perform a hybrid search (combining vector and keyword search).
    </ParamField>


    <ParamField path="filters" type="dict[str, Any]" default="{}">
      Alias for `search_filters`, now `deprecated`.
    </ParamField>

    <ParamField path="search_filters" type="dict[str, Any]" default="{}">
      Filters to apply to the vector search. Allowed operators include `eq`, `neq`, `gt`, `gte`, `lt`, `lte`, `like`, `ilike`, `in`, and `nin`.

      Commonly seen filters include operations include the following:

        `{"document_id": {"$eq": "9fbe403b-..."}}`

        `{"document_id": {"$in": ["9fbe403b-...", "3e157b3a-..."]}}`

        `{"collection_ids": {"$overlap": ["122fdf6a-...", "..."]}}`

        `{"$and": {"$document_id": ..., "collection_ids": ...}}`
    </ParamField>

    <ParamField path="search_limit" type="int" default="10">
    Maximum number of results to return (1-1000).
    </ParamField>

    <ParamField path="selected_collection_ids" type="list[UUID]" default="[]">
    Collection Ids to search for.
    </ParamField>

    <ParamField path="index_measure" type="IndexMeasure" default="cosine_distance">
    The distance measure to use for indexing (cosine_distance, l2_distance, or max_inner_product).
    </ParamField>

    <ParamField path="include_values" type="bool" default="True">
    Whether to include search score values in the search results.
    </ParamField>

    <ParamField path="include_metadatas" type="bool" default="True">
    Whether to include element metadata in the search results.
    </ParamField>

    <ParamField path="probes" type="Optional[int]" default="10">
    Number of ivfflat index lists to query. Higher increases accuracy but decreases speed.
    </ParamField>

    <ParamField path="ef_search" type="Optional[int]" default="40">
    Size of the dynamic candidate list for HNSW index search. Higher increases accuracy but decreases speed.
    </ParamField>

    <ParamField path="hybrid_search_settings" type="Optional[HybridSearchSettings]" default="HybridSearchSettings()">
    Settings for hybrid search.
    <Expandable title="properties">
      <ParamField path="full_text_weight" type="float" default="1.0">
      Weight to apply to full text search.
      </ParamField>

      <ParamField path="semantic_weight" type="float" default="5.0">
      Weight to apply to semantic search.
      </ParamField>

      <ParamField path="full_text_limit" type="int" default="200">
      Maximum number of results to return from full text search.
      </ParamField>

      <ParamField path="rrf_k" type="int" default="50">
      K-value for RRF (Rank Reciprocal Fusion).
      </ParamField>
    </Expandable>
    </ParamField>
  </Expandable>
</ParamField>

<ParamField path="kg_search_settings" type="Optional[Union[KGSearchSettings, dict]]" default="None">
  Optional settings for knowledge graph search, either a dictionary, a `KGSearchSettings` object, or `None` may be passed. If a dictionary or `None` is passed, then R2R will use server-side defaults for non-specified fields.

  <Expandable title="properties">
    <ParamField path="use_kg_search" type="bool" default="False">
    Whether to use knowledge graph search.
    </ParamField>

    <ParamField path="kg_search_type" type="str" default="local">
    Type of knowledge graph search. Can be 'global' or 'local'.
    </ParamField>

    <ParamField path="kg_search_level" type="Optional[str]" default="None">
    Level of knowledge graph search.
    </ParamField>

    <ParamField path="generation_config" type="Optional[GenerationConfig]" default="GenerationConfig()">
    Configuration for knowledge graph search generation.
    </ParamField>

    <ParamField path="entity_types" type="list" default="[]">
    List of entity types to use for knowledge graph search.
    </ParamField>

    <ParamField path="relationships" type="list" default="[]">
    List of relationships to use for knowledge graph search.
    </ParamField>

    <ParamField path="max_community_description_length" type="int" default="65536">
    Maximum length of community descriptions.
    </ParamField>

    <ParamField path="max_llm_queries_for_global_search" type="int" default="250">
    Maximum number of LLM queries for global search.
    </ParamField>

    <ParamField path="local_search_limits" type="dict[str, int]" default="{'__Entity__': 20, '__Relationship__': 20, '__Community__': 20}">
    Limits for local search on different types of elements.
    </ParamField>
  </Expandable>
</ParamField>

<ParamField path="rag_generation_config" type="Optional[Union[GenerationConfig, dict]]" default="None">
  Optional configuration for LLM to use during RAG generation, including model selection and parameters. Will default to values specified in `r2r.toml`.
  <Expandable title="properties">
    <ParamField path="model" type="str" default="openai/gpt-4o">
    Model used in final LLM completion.
    </ParamField>

    <ParamField path="temperature" type="float" default="0.1">
    Temperature used in final LLM completion.
    </ParamField>

    <ParamField path="top_p" type="float" default="1.0">
    The `top_p` used in final LLM completion.
    </ParamField>

    <ParamField path="max_tokens_to_sample" type="int" default="1024">
    The `max_tokens_to_sample` used in final LLM completion.
    </ParamField>

    <ParamField path="functions" type="dict" default="None">
    The `functions` used in final LLM completion.
    </ParamField>

    <ParamField path="tools" type="dict" default="None">
    The `tools` used in final LLM completion.
    </ParamField>

    <ParamField path="api_base" type="str" default="None">
    The `api_base` used in final LLM completion.
    </ParamField>

  </Expandable>


</ParamField>
<ParamField path="task_prompt_override" type="Optional[str]" default="None">
  Optional custom prompt to override the default task prompt.
</ParamField>

<ParamField path="include_title_if_available" type="Optional[bool]" default="True">
  Augment document chunks with their respective document titles?
</ParamField>


### RAG with custom search settings

Learn more about the RAG [API here](/api-reference/endpoint/rag). It allows performing RAG with custom settings, such as hybrid search:

```python
hybrid_rag_response = client.rag(
    "Who is Jon Snow?",
    vector_search_settings={"use_hybrid_search": True}
)
```

### RAG with custom completion LLM

R2R supports configuration on server-side and at runtime, which you can [read about here](/documentation/configuration/rag). An example below, using Anthropic at runtime:

```python
anthropic_rag_response = client.rag(
    "What is R2R?",
    rag_generation_config={"model":"anthropic/claude-3-opus-20240229"}
)
```


### Streaming RAG

R2R supports streaming RAG responses for real-time applications:

```python
stream_response = client.rag(
    "Who was Aristotle?",
    rag_generation_config={"stream": True}
)
for chunk in stream_response:
    print(chunk, end='', flush=True)
```
<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="Generator">
    ```bash
    <search>["{\"id\":\"808c47c5-ebef-504a-a230-aa9ddcfbd87 .... </search>
    <completion>Lyft reported a net loss of $1,752,857,000 in 2020 according to [2]. Therefore, Lyft did not make a profit in 2020.</completion>
    ```
    </ResponseField>
  </Accordion>
</AccordionGroup>


### Advanced RAG Techniques

R2R supports advanced Retrieval-Augmented Generation (RAG) techniques that can be easily configured at runtime. These techniques include Hypothetical Document Embeddings (HyDE) and RAG-Fusion, which can significantly enhance the quality and relevance of retrieved information.

To use an advanced RAG technique, you can specify the `search_strategy` parameter in your vector search settings:

```python
from r2r import R2RClient

client = R2RClient()

# Using HyDE
hyde_response = client.rag(
    "What are the main themes in Shakespeare's plays?",
    vector_search_settings={
        "search_strategy": "hyde",
        "search_limit": 10
    }
)

# Using RAG-Fusion
rag_fusion_response = client.rag(
    "Explain the theory of relativity",
    vector_search_settings={
        "search_strategy": "rag_fusion",
        "search_limit": 20
    }
)
```


For a comprehensive guide on implementing and optimizing advanced RAG techniques in R2R, including HyDE and RAG-Fusion, please refer to our [Advanced RAG Cookbook](/cookbooks/advanced-rag).


### Customizing RAG

Putting everything together for highly customized RAG functionality at runtime:

```python

custom_rag_response = client.rag(
    "Who was Aristotle?",
    vector_search_settings={
        "use_hybrid_search": True,
        "search_limit": 20,
        "hybrid_search_settings": {
            "full_text_weight": 1.0,
            "semantic_weight": 10.0,
            "full_text_limit": 200,
            "rrf_k": 25,
        },
    },
    kg_search_settings={
        "use_kg_search": True,
        "kg_search_type": "local",
    },
    rag_generation_config={
        "model": "anthropic/claude-3-haiku-20240307",
        "temperature": 0.7,
        "stream": True
    },
    task_prompt_override="Only answer the question if the context is SUPER relevant!!\n\nQuery:\n{query}\n\nContext:\n{context}"
)
```

## Agents

### Multi-turn agentic RAG
The R2R application includes agents which come equipped with a search tool, enabling them to perform RAG. Using the R2R Agent for multi-turn conversations:

```python
messages = [
    {"role": "user", "content": "What was Aristotle's main contribution to philosophy?"},
    {"role": "assistant", "content": "Aristotle made numerous significant contributions to philosophy, but one of his main contributions was in the field of logic and reasoning. He developed a system of formal logic, which is considered the first comprehensive system of its kind in Western philosophy. This system, often referred to as Aristotelian logic or term logic, provided a framework for deductive reasoning and laid the groundwork for scientific thinking."},
    {"role": "user", "content": "Can you elaborate on how this influenced later thinkers?"}
]

rag_agent_response = client.agent(
    messages,
    vector_search_settings={"use_hybrid_search":True},
)
```

Note that any of the customization seen in AI powered search and RAG documentation above can be applied here.

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="Object">
    The agent endpoint will return the entire conversation as a response, including internal tool calls.
        ```bash
        {   'results':
            [
                {'role': 'system', 'content': '## You are a helpful agent that can search for information.\n\nWhen asked a question, perform a search to find relevant information and provide a response.\n\nThe response should contain line-item attributions to relevent search results, and be as informative if possible.\nIf no relevant results are found, then state that no results were found.\nIf no obvious question is present, then do not carry out a search, and instead ask for clarification.\n', 'name': None, 'function_call': None, 'tool_calls': None},
                {'role': 'user', 'content': "What was Aristotle's main contribution to philosophy?", 'name': None, 'function_call': None, 'tool_calls': None},
                {'role': 'assistant', 'content': 'Aristotle made numerous significant contributions to philosophy, but one of his main contributions was in the field of logic and reasoning. He developed a system of formal logic, which is considered the first comprehensive system of its kind in Western philosophy. This system, often referred to as Aristotelian logic or term logic, provided a framework for deductive reasoning and laid the groundwork for scientific thinking.', 'name': None, 'function_call': None, 'tool_calls': None},
                {'role': 'user', 'content': 'Can you elaborate on how this influenced later thinkers?', 'name': None, 'function_call': None, 'tool_calls': None},
                {'role': 'assistant', 'content': None, 'name': None, 'function_call': {'name': 'search', 'arguments': '{"query":"Aristotle\'s influence on later thinkers in philosophy"}'}, 'tool_calls': None},
                {'role': 'function', 'content': '1. ormation: List of writers influenced by Aristotle More than 2300 years after his death, Aristotle remains one of the most influential people who ever lived.[142][143][144] He contributed to almost every field of human knowledge then in existence, and he was the founder of many new fields. According to the philosopher Bryan Magee, "it is doubtful whether any human being has ever known as much as he did".[145]\n2. subject of contemporary philosophical discussion. Aristotle\'s views profoundly shaped medieval scholarship. The influence of his physical science extended from late antiquity and the Early Middle Ages into the Renaissance, and was not replaced systematically until the Enlightenment and theories such as classical mechanics were developed. He influenced Judeo-Islamic philosophies during the Middle Ages, as well as Christian theology, especially the Neoplatonism of the Early Church and the scholastic tradition\n3. the scholastic tradition of the Catholic Church. Aristotle was revered among medieval Muslim scholars as "The First Teacher", and among medieval Christians like Thomas Aquinas as simply "The Philosopher", while the poet Dante called him "the master of those who know". His works contain the earliest known formal study of logic, and were studied by medieval scholars such as Peter Abelard and Jean Buridan. Aristotle\'s influence on logic continued well into the 19th century. In addition, his ethics, although\n4. hilosophy\nFurther information: Peripatetic school The immediate influence of Aristotle\'s work was felt as the Lyceum grew into the Peripatetic school. Aristotle\'s students included Aristoxenus, Dicaearchus, Demetrius of Phalerum, Eudemos of Rhodes, Harpalus, Hephaestion, Mnason of Phocis, Nicomachus, and Theophrastus. Aristotle\'s influence over Alexander the Great is seen in the latter\'s bringing with him on his expedition a host of zoologists, botanists, and researchers. He had also learned a great deal\n5. scholastic philosophers. Alkindus greatly admired Aristotle\'s philosophy,[168] and Averroes spoke of Aristotle as the "exemplar" for all future philosophers.[169] Medieval Muslim scholars regularly described Aristotle as the "First Teacher".[167] The title was later used by Western philosophers (as in the famous poem of Dante) who were influenced by the tradition of Islamic philosophy.[170]\n\nMedieval Europe\nFurther information: Aristotelianism and Syllogism § Medieval\n6. those by James of Venice and William of Moerbeke. After the Scholastic Thomas Aquinas wrote his Summa Theologica, working from Moerbeke\'s translations and calling Aristotle "The Philosopher",[172] the demand for Aristotle\'s writings grew, and the Greek manuscripts returned to the West, stimulating a revival of Aristotelianism in Europe that continued into the Renaissance.[173] These thinkers blended Aristotelian philosophy with Christianity, bringing the thought of Ancient Greece into the Middle Ages.\n7. Aristotle[A] (Greek: Ἀριστοτέλης Aristotélēs, pronounced [aristotélɛːs]; 384–322 BC) was an Ancient Greek philosopher and polymath. His writings cover a broad range of subjects spanning the natural sciences, philosophy, linguistics, economics, politics, psychology, and the arts. As the founder of the Peripatetic school of philosophy in the Lyceum in Athens, he began the wider Aristotelian tradition that followed, which set the groundwork for the development of modern science.\n8. , individualism, teleology, and meteorology.[151] The scholar Taneli Kukkonen notes that "in the best 20th-century scholarship Aristotle comes alive as a thinker wrestling with the full weight of the Greek philosophical tradition."[148] What follows is an overview of the transmission and influence of his texts and ideas into the modern era.\n\nHis successor, Theophrastus\nMain articles: Theophrastus and Historia Plantarum (Theophrastus)\n9. man being has ever known as much as he did".[145] Among countless other achievements, Aristotle was the founder of formal logic,[146] pioneered the study of zoology, and left every future scientist and philosopher in his debt through his contributions to the scientific method.[2][147][148] Taneli Kukkonen, observes that his achievement in founding two sciences is unmatched, and his reach in influencing "every branch of intellectual enterprise" including Western ethical and political theory, theology,\n10. ight) in the Kitāb naʿt al-ḥayawān, c.\u20091220.[166] Aristotle was one of the most revered Western thinkers in early Islamic theology. Most of the still extant works of Aristotle,[167] as well as a number of the original Greek commentaries, were translated into Arabic and studied by Muslim philosophers, scientists and scholars. Averroes, Avicenna and Alpharabius, who wrote on Aristotle in great depth, also influenced Thomas Aquinas and other Western Christian scholastic philosophers. Alkindus greatly admired\n', 'name': 'search', 'function_call': None, 'tool_calls': None},
                {'role': 'assistant', 'content': 'Aristotle\'s contributions to philosophy, particularly his development of formal logic, had a profound influence on later thinkers across various cultures and eras. Here are some key ways in which his work influenced subsequent intellectual traditions:\n\n1. **Medieval Islamic Philosophy**: Aristotle was highly revered among medieval Muslim scholars, who referred to him as "The First Teacher." His works were extensively studied and commented upon by philosophers such as Averroes and Avicenna. These scholars played a crucial role in preserving and expanding upon Aristotle\'s ideas, which later influenced Western thought [5][10].\n\n2. **Christian Scholasticism**: In medieval Europe, Aristotle\'s works were integrated into Christian theology, particularly through the efforts of Thomas Aquinas, who referred to Aristotle as "The Philosopher." Aquinas\' synthesis of Aristotelian philosophy with Christian doctrine became a cornerstone of Scholasticism, a dominant intellectual tradition in medieval Europe [3][6].\n\n3. **Renaissance and Enlightenment**: Aristotle\'s influence persisted into the Renaissance, where his works were revived and studied extensively. This period saw a renewed interest in classical texts, and Aristotle\'s ideas continued to shape scientific and philosophical inquiry until the Enlightenment, when new scientific paradigms began to emerge [2][6].\n\n4. **Development of Logic**: Aristotle\'s system of formal logic remained the standard for centuries and was studied by medieval scholars such as Peter Abelard and Jean Buridan. His influence on logic extended well into the 19th century, shaping the development of this field [3].\n\n5. **Peripatetic School**: Aristotle\'s immediate influence was also felt through the Peripatetic school, which he founded. His students, including Theophrastus, carried on his work and further developed his ideas, ensuring that his intellectual legacy continued [4][8].\n\nOverall, Aristotle\'s contributions laid the groundwork for many fields of study and influenced a wide range of thinkers, making him one of the most significant figures in the history of Western philosophy.', 'name': None, 'function_call': None, 'tool_calls': None}
            ]
        }
        ```
    </ResponseField>
  </Accordion>
</AccordionGroup>


<ParamField path="messages" type="list[Messages]" required>
  The list of messages to pass the RAG agent.
</ParamField>
<ParamField path="vector_search_settings" type="Optional[Union[VectorSearchSettings, dict]]" default="None">
  Optional settings for vector search, either a dictionary, a `VectorSearchSettings` object, or `None` may be passed. If a dictionary is used, non-specified fields will use the server-side default.

  <Expandable title="properties">
    <ParamField path="use_vector_search" type="bool" default="True">
    Whether to use vector search.
    </ParamField>

    <ParamField path="use_hybrid_search" type="bool" default="False">
    Whether to perform a hybrid search (combining vector and keyword search).
    </ParamField>

    <ParamField path="filters" type="dict[str, Any]" default="{}">
      Alias for `search_filters`, now `deprecated`.
    </ParamField>

    <ParamField path="search_filters" type="dict[str, Any]" default="{}">
      Filters to apply to the vector search. Allowed operators include `eq`, `neq`, `gt`, `gte`, `lt`, `lte`, `like`, `ilike`, `in`, and `nin`.

      Commonly seen filters include operations include the following:

        `{"document_id": {"$eq": "9fbe403b-..."}}`

        `{"document_id": {"$in": ["9fbe403b-...", "3e157b3a-..."]}}`

        `{"collection_ids": {"$overlap": ["122fdf6a-...", "..."]}}`

        `{"$and": {"$document_id": ..., "collection_ids": ...}}`
    </ParamField>

    <ParamField path="search_limit" type="int" default="10">
    Maximum number of results to return (1-1000).
    </ParamField>

    <ParamField path="selected_collection_ids" type="list[UUID]" default="[]">
    Collection Ids to search for.
    </ParamField>

    <ParamField path="index_measure" type="IndexMeasure" default="cosine_distance">
    The distance measure to use for indexing (cosine_distance, l2_distance, or max_inner_product).
    </ParamField>

    <ParamField path="include_values" type="bool" default="True">
    Whether to include search score values in the search results.
    </ParamField>

    <ParamField path="include_metadatas" type="bool" default="True">
    Whether to include element metadata in the search results.
    </ParamField>

    <ParamField path="probes" type="Optional[int]" default="10">
    Number of ivfflat index lists to query. Higher increases accuracy but decreases speed.
    </ParamField>

    <ParamField path="ef_search" type="Optional[int]" default="40">
    Size of the dynamic candidate list for HNSW index search. Higher increases accuracy but decreases speed.
    </ParamField>

    <ParamField path="hybrid_search_settings" type="Optional[HybridSearchSettings]" default="HybridSearchSettings()">
    Settings for hybrid search.
    <Expandable title="properties">
      <ParamField path="full_text_weight" type="float" default="1.0">
      Weight to apply to full text search.
      </ParamField>

      <ParamField path="semantic_weight" type="float" default="5.0">
      Weight to apply to semantic search.
      </ParamField>

      <ParamField path="full_text_limit" type="int" default="200">
      Maximum number of results to return from full text search.
      </ParamField>

      <ParamField path="rrf_k" type="int" default="50">
      K-value for RRF (Rank Reciprocal Fusion).
      </ParamField>
    </Expandable>
    </ParamField>
  </Expandable>
</ParamField>

<ParamField path="kg_search_settings" type="Optional[Union[KGSearchSettings, dict]]" default="None">
  Optional settings for knowledge graph search, either a dictionary, a `KGSearchSettings` object, or `None` may be passed. If a dictionary or `None` is passed, then R2R will use server-side defaults for non-specified fields.

  <Expandable title="properties">
    <ParamField path="use_kg_search" type="bool" default="False">
    Whether to use knowledge graph search.
    </ParamField>

    <ParamField path="kg_search_type" type="str" default="local">
    Type of knowledge graph search. Can be 'global' or 'local'.
    </ParamField>

    <ParamField path="kg_search_level" type="Optional[str]" default="None">
    Level of knowledge graph search.
    </ParamField>

    <ParamField path="generation_config" type="Optional[GenerationConfig]" default="GenerationConfig()">
    Configuration for knowledge graph search generation.
    </ParamField>

    <ParamField path="entity_types" type="list" default="[]">
    List of entity types to use for knowledge graph search.
    </ParamField>

    <ParamField path="relationships" type="list" default="[]">
    List of relationships to use for knowledge graph search.
    </ParamField>

    <ParamField path="max_community_description_length" type="int" default="65536">
    Maximum length of community descriptions.
    </ParamField>

    <ParamField path="max_llm_queries_for_global_search" type="int" default="250">
    Maximum number of LLM queries for global search.
    </ParamField>

    <ParamField path="local_search_limits" type="dict[str, int]" default="{'__Entity__': 20, '__Relationship__': 20, '__Community__': 20}">
    Limits for local search on different types of elements.
    </ParamField>
  </Expandable>
</ParamField>


<ParamField path="rag_generation_config" type="Optional[Union[GenerationConfig, dict]]" default="None">
  Optional configuration for LLM to use during RAG generation, including model selection and parameters. Will default to values specified in `r2r.toml`.
  <Expandable title="properties">
    <ParamField path="model" type="str" default="openai/gpt-4o">
    Model used in final LLM completion.
    </ParamField>

    <ParamField path="temperature" type="float" default="0.1">
    Temperature used in final LLM completion.
    </ParamField>

    <ParamField path="top_p" type="float" default="1.0">
    The `top_p` used in final LLM completion.
    </ParamField>

    <ParamField path="max_tokens_to_sample" type="int" default="1024">
    The `max_tokens_to_sample` used in final LLM completion.
    </ParamField>

    <ParamField path="functions" type="dict" default="None">
    The `functions` used in final LLM completion.
    </ParamField>

    <ParamField path="tools" type="dict" default="None">
    The `tools` used in final LLM completion.
    </ParamField>

    <ParamField path="api_base" type="str" default="None">
    The `api_base` used in final LLM completion.
    </ParamField>

  </Expandable>

</ParamField>
<ParamField path="task_prompt_override" type="Optional[str]" default="None">
  Optional custom prompt to override the default task prompt.
</ParamField>




### Multi-turn agentic RAG with streaming
The response from the RAG agent may be streamed directly back

```python
messages = [
    {"role": "user", "content": "What was Aristotle's main contribution to philosophy?"},
    {"role": "assistant", "content": "Aristotle made numerous significant contributions to philosophy, but one of his main contributions was in the field of logic and reasoning. He developed a system of formal logic, which is considered the first comprehensive system of its kind in Western philosophy. This system, often referred to as Aristotelian logic or term logic, provided a framework for deductive reasoning and laid the groundwork for scientific thinking."},
    {"role": "user", "content": "Can you elaborate on how this influenced later thinkers?"}
]

rag_agent_response = client.agent(
    messages,
    vector_search_settings={"use_hybrid_search":True},
    rag_generation_config={"stream":True}
)
```


<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="Generator">
    The agent endpoint will stream back its response, including internal tool calls.
        ```bash
        <function_call><name>search</name><arguments>{"query":"Aristotle's influence on later thinkers in philosophy"}</arguments><results>"{\"id\":\"b234931e-0cfb-5644-8f23-560a3097f5fe\",\"score\":1.0,\"metadata\":{\"text\":\"ormation: List of writers influenced by Aristotle More than 2300 years after his death, Aristotle remains one of the most influential people who ever lived.[142][143][144] He contributed to almost every field of human knowledge then in existence, and he was the founder of many new fields. According to the philosopher Bryan Magee, \\\"it is doubtful whether any human being has ever known as much as he did\\\".[145]\",\"title\":\"aristotle.txt\",\"user_id\":\"2acb499e-8428-543b-bd85-0d9098718220\",\"document_id\":\"9fbe403b-c11c-5aae-8ade-ef22980c3ad1\",\"extraction_id\":\"69431c4a-30cf-504f-8fab-7dcfc7580c63\",\"associatedQuery\":\"Aristotle's influence on later thinkers in philosophy\"}}","{\"id\":\"1827ac2c-2a06-5bc2-ad29-aa14b4d99540\",\"score\":1.0,\"metadata\":{\"text\":\"subject of contemporary philosophical discussion. Aristotle's views profoundly shaped medieval scholarship. The influence of his physical science extended from late antiquity and the Early Middle Ages into the Renaissance, and was not replaced systematically until the Enlightenment and theories such as classical mechanics were developed. He influenced Judeo-Islamic philosophies during the Middle Ages, as well as Christian theology, especially the Neoplatonism of the Early Church and the scholastic tradition\",\"title\":\"aristotle.txt\",\"user_id\":\"2acb499e-8428-543b-bd85-0d9098718220\",\"document_id\":\"9fbe403b-c11c-5aae-8ade-ef22980c3ad1\",\"extraction_id\":\"69431c4a-30cf-504f-8fab-7dcfc7580c63\",\"associatedQuery\":\"Aristotle's influence on later thinkers in philosophy\"}}","{\"id\":\"94718936-ea92-5e29-a5ee-d4a6bc037384\",\"score\":1.0,\"metadata\":{\"text\":\"the scholastic tradition of the Catholic Church. Aristotle was revered among medieval Muslim scholars as \\\"The First Teacher\\\", and among medieval Christians like Thomas Aquinas as simply \\\"The Philosopher\\\", while the poet Dante called him \\\"the master of those who know\\\". His works contain the earliest known formal study of logic, and were studied by medieval scholars such as Peter Abelard and Jean Buridan. Aristotle's influence on logic continued well into the 19th century. In addition, his ethics, although\",\"title\":\"aristotle.txt\",\"user_id\":\"2acb499e-8428-543b-bd85-0d9098718220\",\"document_id\":\"9fbe403b-c11c-5aae-8ade-ef22980c3ad1\",\"extraction_id\":\"69431c4a-30cf-504f-8fab-7dcfc7580c63\",\"associatedQuery\":\"Aristotle's influence on later thinkers in philosophy\"}}","{\"id\":\"16483f14-f8a2-5c5c-8fcd-1bcbbd6603e4\",\"score\":1.0,\"metadata\":{\"text\":\"hilosophy\\nFurther information: Peripatetic school The immediate influence of Aristotle's work was felt as the Lyceum grew into the Peripatetic school. Aristotle's students included Aristoxenus, Dicaearchus, Demetrius of Phalerum, Eudemos of Rhodes, Harpalus, Hephaestion, Mnason of Phocis, Nicomachus, and Theophrastus. Aristotle's influence over Alexander the Great is seen in the latter's bringing with him on his expedition a host of zoologists, botanists, and researchers. He had also learned a great deal\",\"title\":\"aristotle.txt\",\"user_id\":\"2acb499e-8428-543b-bd85-0d9098718220\",\"document_id\":\"9fbe403b-c11c-5aae-8ade-ef22980c3ad1\",\"extraction_id\":\"69431c4a-30cf-504f-8fab-7dcfc7580c63\",\"associatedQuery\":\"Aristotle's influence on later thinkers in philosophy\"}}","{\"id\":\"26eb20ee-a203-5ad5-beaa-511cc526aa6e\",\"score\":1.0,\"metadata\":{\"text\":\"scholastic philosophers. Alkindus greatly admired Aristotle's philosophy,[168] and Averroes spoke of Aristotle as the \\\"exemplar\\\" for all future philosophers.[169] Medieval Muslim scholars regularly described Aristotle as the \\\"First Teacher\\\".[167] The title was later used by Western philosophers (as in the famous poem of Dante) who were influenced by the tradition of Islamic philosophy.[170]\\n\\nMedieval Europe\\nFurther information: Aristotelianism and Syllogism \u00a7 Medieval\",\"title\":\"aristotle.txt\",\"user_id\":\"2acb499e-8428-543b-bd85-0d9098718220\",\"document_id\":\"9fbe403b-c11c-5aae-8ade-ef22980c3ad1\",\"extraction_id\":\"69431c4a-30cf-504f-8fab-7dcfc7580c63\",\"associatedQuery\":\"Aristotle's influence on later thinkers in philosophy\"}}","{\"id\":\"a08fd1b4-4e6f-5487-9af6-df2f6cfe1048\",\"score\":1.0,\"metadata\":{\"text\":\"those by James of Venice and William of Moerbeke. After the Scholastic Thomas Aquinas wrote his Summa Theologica, working from Moerbeke's translations and calling Aristotle \\\"The Philosopher\\\",[172] the demand for Aristotle's writings grew, and the Greek manuscripts returned to the West, stimulating a revival of Aristotelianism in Europe that continued into the Renaissance.[173] These thinkers blended Aristotelian philosophy with Christianity, bringing the thought of Ancient Greece into the Middle Ages.\",\"title\":\"aristotle.txt\",\"user_id\":\"2acb499e-8428-543b-bd85-0d9098718220\",\"document_id\":\"9fbe403b-c11c-5aae-8ade-ef22980c3ad1\",\"extraction_id\":\"69431c4a-30cf-504f-8fab-7dcfc7580c63\",\"associatedQuery\":\"Aristotle's influence on later thinkers in philosophy\"}}","{\"id\":\"9689a804-5a95-5696-97da-a076a3eb8320\",\"score\":1.0,\"metadata\":{\"text\":\"Aristotle[A] (Greek: \u1f08\u03c1\u03b9\u03c3\u03c4\u03bf\u03c4\u03ad\u03bb\u03b7\u03c2 Aristot\u00e9l\u0113s, pronounced [aristot\u00e9l\u025b\u02d0s]; 384\u2013322 BC) was an Ancient Greek philosopher and polymath. His writings cover a broad range of subjects spanning the natural sciences, philosophy, linguistics, economics, politics, psychology, and the arts. As the founder of the Peripatetic school of philosophy in the Lyceum in Athens, he began the wider Aristotelian tradition that followed, which set the groundwork for the development of modern science.\",\"title\":\"aristotle.txt\",\"user_id\":\"2acb499e-8428-543b-bd85-0d9098718220\",\"document_id\":\"9fbe403b-c11c-5aae-8ade-ef22980c3ad1\",\"extraction_id\":\"69431c4a-30cf-504f-8fab-7dcfc7580c63\",\"associatedQuery\":\"Aristotle's influence on later thinkers in philosophy\"}}","{\"id\":\"dd19a0d6-4cef-590b-9721-35c26b1ee056\",\"score\":1.0,\"metadata\":{\"text\":\", individualism, teleology, and meteorology.[151] The scholar Taneli Kukkonen notes that \\\"in the best 20th-century scholarship Aristotle comes alive as a thinker wrestling with the full weight of the Greek philosophical tradition.\\\"[148] What follows is an overview of the transmission and influence of his texts and ideas into the modern era.\\n\\nHis successor, Theophrastus\\nMain articles: Theophrastus and Historia Plantarum (Theophrastus)\",\"title\":\"aristotle.txt\",\"user_id\":\"2acb499e-8428-543b-bd85-0d9098718220\",\"document_id\":\"9fbe403b-c11c-5aae-8ade-ef22980c3ad1\",\"extraction_id\":\"69431c4a-30cf-504f-8fab-7dcfc7580c63\",\"associatedQuery\":\"Aristotle's influence on later thinkers in philosophy\"}}","{\"id\":\"8d125c7a-0084-5adf-b094-c96c91611897\",\"score\":1.0,\"metadata\":{\"text\":\"man being has ever known as much as he did\\\".[145] Among countless other achievements, Aristotle was the founder of formal logic,[146] pioneered the study of zoology, and left every future scientist and philosopher in his debt through his contributions to the scientific method.[2][147][148] Taneli Kukkonen, observes that his achievement in founding two sciences is unmatched, and his reach in influencing \\\"every branch of intellectual enterprise\\\" including Western ethical and political theory, theology,\",\"title\":\"aristotle.txt\",\"user_id\":\"2acb499e-8428-543b-bd85-0d9098718220\",\"document_id\":\"9fbe403b-c11c-5aae-8ade-ef22980c3ad1\",\"extraction_id\":\"69431c4a-30cf-504f-8fab-7dcfc7580c63\",\"associatedQuery\":\"Aristotle's influence on later thinkers in philosophy\"}}","{\"id\":\"40d671b0-a412-5822-b088-461baf2324e6\",\"score\":1.0,\"metadata\":{\"text\":\"ight) in the Kit\u0101b na\u02bft al-\u1e25ayaw\u0101n, c.\u20091220.[166] Aristotle was one of the most revered Western thinkers in early Islamic theology. Most of the still extant works of Aristotle,[167] as well as a number of the original Greek commentaries, were translated into Arabic and studied by Muslim philosophers, scientists and scholars. Averroes, Avicenna and Alpharabius, who wrote on Aristotle in great depth, also influenced Thomas Aquinas and other Western Christian scholastic philosophers. Alkindus greatly admired\",\"title\":\"aristotle.txt\",\"user_id\":\"2acb499e-8428-543b-bd85-0d9098718220\",\"document_id\":\"9fbe403b-c11c-5aae-8ade-ef22980c3ad1\",\"extraction_id\":\"69431c4a-30cf-504f-8fab-7dcfc7580c63\",\"associatedQuery\":\"Aristotle's influence on later thinkers in philosophy\"}}"</results></function_call><completion>Aristotle's contributions to philosophy, particularly his development of formal logic, had a profound influence on later thinkers across various cultures and eras. Here are some key ways in which his work influenced subsequent intellectual traditions:

        1. **Medieval Islamic Philosophy**: Aristotle was highly revered among medieval Muslim scholars, who referred to him as "The First Teacher." His works were extensively translated into Arabic and studied by philosophers such as Averroes and Avicenna. These scholars not only preserved Aristotle's works but also expanded upon them, influencing both Islamic and Western thought [5][10].

        2. **Christian Scholasticism**: In medieval Europe, Aristotle's works were integrated into Christian theology, particularly through the efforts of Thomas Aquinas, who referred to Aristotle as "The Philosopher." Aquinas's synthesis of Aristotelian philosophy with Christian doctrine became a cornerstone of Scholasticism, a dominant intellectual tradition in medieval Europe [3][6].

        3. **Renaissance and Enlightenment**: Aristotle's influence persisted into the Renaissance and Enlightenment periods. His works on logic, ethics, and natural sciences were foundational texts for scholars during these eras. The revival of Aristotelianism during the Renaissance helped bridge the gap between ancient Greek philosophy and modern scientific thought [2][6].

        4. **Development of Modern Science**: Aristotle's method of systematic observation and classification in natural sciences laid the groundwork for the scientific method. His influence extended well into the 19th century, impacting the development of various scientific disciplines [9].

        5. **Peripatetic School**: Aristotle's immediate influence was felt through the Peripatetic school, which he founded. His students, including Theophrastus, continued to develop and disseminate his ideas, ensuring that his philosophical legacy endured [4][8].

        Overall, Aristotle's contributions to logic, ethics, natural sciences, and metaphysics created a foundation upon which much of Western intellectual tradition was built. His work influenced a wide range of fields and thinkers, making him one of the most pivotal figures in the history of philosophy.</completion>
        ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

---


<Note>
This SDK documentation is periodically updated. For the latest parameter details, please cross-reference with the <a href="/api-reference/introduction">API Reference documentation</a>.
</Note>

Inside R2R, `ingestion` refers to the complete pipeline for processing input data:
- Parsing files into text
- Chunking text into semantic units
- Generating embeddings
- Storing data for retrieval

Ingested files are stored with an associated document identifier as well as a user identifier to enable comprehensive management.

## Document Ingestion and Management

<Note>
R2R has recently expanded the available options for ingesting files using multimodal foundation models. In addition to using such models by default for images, R2R can now use them on PDFs, [like it is shown here](https://github.com/getomni-ai/zerox), by passing the following in your ingestion configuration:

```json
{
  "ingestion_config": {
    "parser_overrides": {
      "pdf": "zerox"
    }
  }
}
```

We recommend this method for achieving the highest quality ingestion results.
</Note>

### Ingest Files

Ingest files or directories into your R2R system:

```javascript
const files = [
  { path: 'path/to/file1.txt', name: 'file1.txt' },
  { path: 'path/to/file2.txt', name: 'file2.txt' }
];
const metadatas = [
  { key1: 'value1' },
  { key2: 'value2' }
];

// Runtime chunking configuration
const ingestResponse = await client.ingestFiles(files, {
  metadatas,
  user_ids: ['user-id-1', 'user-id-2'],
  ingestion_config: {
    provider: "unstructured_local",  // Local processing
    strategy: "auto",                // Automatic processing strategy
    chunking_strategy: "by_title",   // Split on title boundaries
    new_after_n_chars: 256,         // Start new chunk (soft limit)
    max_characters: 512,            // Maximum chunk size (hard limit)
    combine_under_n_chars: 64,      // Minimum chunk size
    overlap: 100,                   // Character overlap between chunks
  }
});
```

[Previous sections remain the same through the Update Files code example, then continuing with:]

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      The response from the R2R system after updating the files.
      ```bash
      {
        'results': {
          'processed_documents': [
            {
              'id': '9f375ce9-efe9-5b57-8bf2-a63dee5f3621',
              'title': 'updated_doc.txt'
            }
          ],
          'failed_documents': [],
          'skipped_documents': []
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="files" type="Array<File | { path: string; name: string }>" required>
  Array of files to update.
</ParamField>

<ParamField path="options" type="object" required>
  <ParamField path="document_ids" type="Array<string>" required>
    Document IDs corresponding to files being updated.
  </ParamField>

  <ParamField path="metadatas" type="Array<Record<string, any>>">
    Optional metadata for updated files.
  </ParamField>

  <ParamField path="ingestion_config" type="object">
    Chunking configuration options.
  </ParamField>
</ParamField>


<ParamField path="run_with_orchestration" type="Optional[bool]">
  Whether or not ingestion runs with orchestration, default is `True`. When set to `False`, the ingestion process will run synchronous and directly return the result.
</ParamField>


### Update Chunks

Update the content of an existing chunk in your R2R system:

```javascript
const documentId = "9fbe403b-c11c-5aae-8ade-ef22980c3ad1";
const extractionId = "aeba6400-1bd0-5ee9-8925-04732d675434";

const updateResponse = await client.updateChunks({
  document_id: documentId,
  extraction_id: extractionId,
  text: "Updated chunk content...",
  metadata: {
    source: "manual_edit",
    edited_at: "2024-10-24"
  }
});
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      The response from the R2R system after updating the chunk.
      ```bash
      {
        'message': 'Update chunk task queued successfully.',
        'task_id': '7e27dfca-606d-422d-b73f-2d9e138661b4',
        'document_id': '9fbe403b-c11c-5aae-8ade-ef22980c3ad1'
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="params" type="object" required>
  <ParamField path="document_id" type="string" required>
    The ID of the document containing the chunk to update.
  </ParamField>

  <ParamField path="extraction_id" type="string" required>
    The ID of the specific chunk to update.
  </ParamField>

  <ParamField path="text" type="string" required>
    The new text content to replace the existing chunk text.
  </ParamField>

  <ParamField path="metadata" type="Record<string, any>">
    An optional metadata object for the updated chunk. If provided, this will replace the existing chunk metadata.
  </ParamField>

  <ParamField path="run_with_orchestration" type="boolean">
    Whether or not the update runs with orchestration, default is `true`. When set to `false`, the update process will run synchronous and directly return the result.
  </ParamField>
</ParamField>


### Documents Overview

Retrieve high-level document information:

```javascript
// Get all documents (paginated)
const documentsOverview = await client.documentsOverview();

// Get specific documents
const specificDocs = await client.documentsOverview({
  document_ids: ['doc-id-1', 'doc-id-2'],
  offset: 0,
  limit: 10
});
```

Results are restricted to the current user's files unless the request is made by a superuser.

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="Array<object>">
      ```bash
      [
        {
          'document_id': '9fbe403b-c11c-5aae-8ade-ef22980c3ad1',
          'version': 'v1',
          'size_in_bytes': 73353,
          'ingestion_status': 'success',
          'restructuring_status': 'pending',
          'user_id': '2acb499e-8428-543b-bd85-0d9098718220',
          'title': 'aristotle.txt',
          'created_at': '2024-07-21T20:09:14.218741Z',
          'updated_at': '2024-07-21T20:09:14.218741Z',
          'metadata': {'title': 'aristotle.txt', 'version': 'v0', 'x': 'y'}
        },
        ...
      ]
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="document_ids" type="Array<string>">
  Optional array of document IDs to filter results.
</ParamField>

<ParamField path="offset" type="number">
  Starting point for pagination, defaults to 0.
</ParamField>

<ParamField path="limit" type="number">
  Maximum number of results to return, defaults to 100.
</ParamField>

### Document Chunks

Fetch and examine chunks for a particular document:

```javascript
const documentId = '9fbe403b-c11c-5aae-8ade-ef22980c3ad1';
const chunks = await client.documentChunks(
  documentId,
  0,     // offset
  100,   // limit
  false  // include_vectors
);
```

These chunks represent the atomic units of text after processing.

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="Array<object>">
      ```bash
      [
        {
          'text': 'Aristotle[A] (Greek: Ἀριστοτέλης Aristotélēs, pronounced [aristotélɛːs]; 384–322 BC) was an Ancient Greek philosopher and polymath...',
          'user_id': '2acb499e-8428-543b-bd85-0d9098718220',
          'document_id': '9fbe403b-c11c-5aae-8ade-ef22980c3ad1',
          'extraction_id': 'aeba6400-1bd0-5ee9-8925-04732d675434',
          'fragment_id': 'f48bcdad-4155-52a4-8c9d-8ba06e996ba3',
          'metadata': {
            'title': 'aristotle.txt',
            'version': 'v0',
            'chunk_order': 0,
            'document_type': 'txt',
            'unstructured_filetype': 'text/plain',
            'unstructured_languages': ['eng']
          }
        },
        ...
      ]
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="documentId" type="string" required>
  ID of the document to retrieve chunks for.
</ParamField>

<ParamField path="offset" type="number">
  Starting point for pagination, defaults to 0.
</ParamField>

<ParamField path="limit" type="number">
  Maximum number of chunks to return, defaults to 100.
</ParamField>

<ParamField path="includeVectors" type="boolean">
  Whether to include embedding vectors in response.
</ParamField>

### Delete Documents

Delete documents using filters:

```javascript
const deleteResponse = await client.delete({
  document_id: {
    "$eq": "91662726-7271-51a5-a0ae-34818509e1fd"
  }
});

// Delete multiple documents
const bulkDelete = await client.delete({
  user_id: {
    "$in": ["user-1", "user-2"]
  }
});
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      ```bash
      {'results': {}}
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="filters" type="object" required>
  Filter conditions to identify documents for deletion.
</ParamField>

## Vector Index Management

### Create Vector Index

<Note>
Vector indices significantly improve search performance for large collections but add overhead for smaller datasets. Only create indices when working with hundreds of thousands of documents or when search latency is critical.
</Note>

Create a vector index for similarity search:

```javascript
const createResponse = await client.createVectorIndex({
    tableName: "vectors",
    indexMethod: "hnsw",
    indexMeasure: "cosine_distance",
    indexArguments: {
        m: 16,                  // Number of connections
        ef_construction: 64     // Build time quality factor
    },
    concurrently: true
});
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      ```bash
      {
        'message': 'Vector index creation task queued successfully.',
        'task_id': '7d38dfca-606d-422d-b73f-2d9e138661b5'
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="tableName" type="string">
  Table to create index on: vectors, entities_document, entities_collection, communities.
</ParamField>

<ParamField path="indexMethod" type="string">
  Index method: hnsw, ivfflat, or auto.
</ParamField>

<ParamField path="indexMeasure" type="string">
  Distance measure: cosine_distance, l2_distance, or max_inner_product.
</ParamField>

<ParamField path="indexArguments" type="object">
  Configuration for chosen index method.
  <Expandable title="HNSW Parameters">
    <ParamField path="m" type="number">
      Number of connections per element (16-64).
    </ParamField>
    <ParamField path="ef_construction" type="number">
      Size of candidate list during construction (64-200).
    </ParamField>
  </Expandable>
  <Expandable title="IVFFlat Parameters">
    <ParamField path="n_lists" type="number">
      Number of clusters/inverted lists.
    </ParamField>
  </Expandable>
</ParamField>

### List Vector Indices

List existing indices:

```javascript
const indices = await client.listVectorIndices({
    tableName: "vectors"
});
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      ```bash
      {
        'indices': [
          {
            'name': 'ix_vector_cosine_ops_hnsw__20241021211541',
            'table': 'vectors',
            'method': 'hnsw',
            'measure': 'cosine_distance'
          },
          ...
        ]
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Delete Vector Index

Remove an existing index:

```javascript
const deleteResponse = await client.deleteVectorIndex({
    indexName: "ix_vector_cosine_ops_hnsw__20241021211541",
    tableName: "vectors",
    concurrently: true
});
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      ```bash
      {
        'message': 'Vector index deletion task queued successfully.',
        'task_id': '8e49efca-606d-422d-b73f-2d9e138661b6'
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

## Best Practices and Performance Optimization

### Vector Index Configuration

1. **HNSW Parameters:**
   - `m`: Higher values (16-64) improve search quality but increase memory
   - `ef_construction`: Higher values improve quality but slow construction
   - Recommended starting point: `m=16`, `ef_construction=64`

2. **Distance Measures:**
   - `cosine_distance`: Best for normalized vectors (most common)
   - `l2_distance`: Better for absolute distances
   - `max_inner_product`: Optimized for dot product similarity

3. **Production Considerations:**
   - Always use `concurrently: true` to avoid blocking operations
   - Create indexes during off-peak hours
   - Pre-warm indices with representative queries
   - Monitor memory usage during creation

### Chunking Strategy

1. **Size Guidelines:**
   - Avoid chunks >1024 characters for retrieval quality
   - Keep chunks >64 characters to maintain context
   - Use overlap for better context preservation

2. **Method Selection:**
   - Use `by_title` for structured documents
   - Use `basic` for uniform text content
   - Consider `recursive` for nested content

## Troubleshooting

### Common Issues

1. **Ingestion Failures:**
   - Verify file permissions and paths
   - Check file format support
   - Ensure metadata array length matches files
   - Monitor memory for large files

2. **Vector Index Performance:**
   - Check index creation time
   - Monitor memory usage
   - Verify warm-up queries
   - Consider rebuilding if quality degrades

3. **Chunking Issues:**
   - Adjust overlap for context preservation
   - Monitor chunk sizes
   - Verify language detection
   - Check encoding for special characters

---


<Note>
Occasionally this SDK documentation falls out of date, cross-check with the automatically generated <a href="/api-reference/introduction"> API Reference documentation </a> for the latest parameters.
</Note>

A collection in R2R is a logical grouping of users and documents that allows for efficient access control and organization. Collections enable you to manage permissions and access to documents at a collection level, rather than individually.

R2R provides a comprehensive set of collection features, allowing you to implement efficient access control and organization of users and documents in your applications.

<Note>
Collection permissioning in R2R is still under development and as a result the API will likely evolve.
</Note>

## Collection Creation and Management

### Create a Collection

Create a new collection with a name and optional description:

```javascript
const createCollectionResponse = await client.createCollection(
  "Marketing Team",
  "Collection for marketing department"
);
const collectionId = createCollectionResponse.results.collection_id; // '123e4567-e89b-12d3-a456-426614174000'
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      ```javascript
      {
        results: {
          collection_id: '123e4567-e89b-12d3-a456-426614174000',
          name: 'Marketing Team',
          description: 'Collection for marketing department',
          created_at: '2024-07-16T22:53:47.524794Z',
          updated_at: '2024-07-16T22:53:47.524794Z'
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Get Collection Details

Retrieve details about a specific collection:

```javascript
const collectionDetails = await client.getCollection(collectionId);
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      ```javascript
      {
        results: {
          collection_id: '123e4567-e89b-12d3-a456-426614174000',
          name: 'Marketing Team',
          description: 'Collection for marketing department',
          created_at: '2024-07-16T22:53:47.524794Z',
          updated_at: '2024-07-16T22:53:47.524794Z'
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Update a Collection

Update a collection's name or description:

```javascript
const updateResult = await client.updateCollection(
  collectionId,
  "Updated Marketing Team",
  "New description for marketing team"
);
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      ```javascript
      {
        results: {
          collection_id: '123e4567-e89b-12d3-a456-426614174000',
          name: 'Updated Marketing Team',
          description: 'New description for marketing team',
          created_at: '2024-07-16T22:53:47.524794Z',
          updated_at: '2024-07-16T23:15:30.123456Z'
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### List Collections

Get a list of all collections:

```javascript
const collectionsList = await client.listCollections();
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      ```javascript
      {
        results: [
          {
            collection_id: '123e4567-e89b-12d3-a456-426614174000',
            name: 'Updated Marketing Team',
            description: 'New description for marketing team',
            created_at: '2024-07-16T22:53:47.524794Z',
            updated_at: '2024-07-16T23:15:30.123456Z'
          },
          // ... other collections ...
        ]
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

## User Management in Collections

### Add User to Collection

Add a user to a collection:

```javascript
const userId = '456e789f-g01h-34i5-j678-901234567890'; // This should be a valid user ID
const addUserResult = await client.addUserToCollection(userId, collectionId);
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      ```javascript
      {
        results: {
          message: 'User successfully added to the collection'
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Remove User from Collection

Remove a user from a collection:

```javascript
const removeUserResult = await client.removeUserFromCollection(userId, collectionId);
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      ```javascript
      {
        results: {
          message: 'User successfully removed from the collection'
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### List Users in Collection

Get a list of all users in a specific collection:

```javascript
const usersInCollection = await client.getUsersInCollection(collectionId);
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      ```javascript
      {
        results: [
          {
            user_id: '456e789f-g01h-34i5-j678-901234567890',
            email: 'user@example.com',
            name: 'John Doe',
            // ... other user details ...
          },
          // ... other users ...
        ]
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Get User's Collections

Get all collections that a user is a member of:

```javascript
const userCollections = await client.getCollectionsForUser(userId);
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      ```javascript
      {
        results: [
          {
            collection_id: '123e4567-e89b-12d3-a456-426614174000',
            name: 'Updated Marketing Team',
            // ... other collection details ...
          },
          // ... other collections ...
        ]
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

## Document Management in Collections

### Assign Document to Collection

Assign a document to a collection:

```javascript
const documentId = '789g012j-k34l-56m7-n890-123456789012'; // Must be a valid document ID
const assignDocResult = await client.assignDocumentToCollection(documentId, collectionId);
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      ```javascript
      {
        results: {
          message: 'Document successfully assigned to the collection'
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Remove Document from Collection

Remove a document from a collection:

```javascript
const removeDocResult = await client.removeDocumentFromCollection(documentId, collectionId);
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      ```javascript
      {
        results: {
          message: 'Document successfully removed from the collection'
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### List Documents in Collection

Get a list of all documents in a specific collection:

```javascript
const docsInCollection = await client.getDocumentsInCollection(collectionId);
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      ```javascript
      {
        results: [
          {
            document_id: '789g012j-k34l-56m7-n890-123456789012',
            title: 'Marketing Strategy 2024',
            // ... other document details ...
          },
          // ... other documents ...
        ]
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Get Document's Collections

Get all collections that a document is assigned to:

```javascript
const documentCollections = await client.getDocumentCollections(documentId);
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      ```javascript
      {
        results: [
          {
            collection_id: '123e4567-e89b-12d3-a456-426614174000',
            name: 'Updated Marketing Team',
            // ... other collection details ...
          },
          // ... other collections ...
        ]
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

## Advanced Collection Management

### Collection Overview

Get an overview of collections, including user and document counts:

```javascript
const collectionsOverview = await client.collectionsOverview();
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      ```javascript
      {
        results: [
          {
            collection_id: '123e4567-e89b-12d3-a456-426614174000',
            name: 'Updated Marketing Team',
            description: 'New description for marketing team',
            user_count: 5,
            document_count: 10,
            created_at: '2024-07-16T22:53:47.524794Z',
            updated_at: '2024-07-16T23:15:30.123456Z'
          },
          // ... other collections ...
        ]
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Delete a Collection

Delete a collection:

```javascript
const deleteResult = await client.deleteCollection(collectionId);
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      ```javascript
      {
        results: {
          message: 'Collection successfully deleted'
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

## Pagination and Filtering

Many collection-related methods support pagination and filtering:

```javascript
// List collections with pagination
const paginatedCollections = await client.listCollections(10, 20);

// Get users in a collection with pagination
const paginatedUsers = await client.getUsersInCollection(collectionId, 5, 10);

// Get documents in a collection with pagination
const paginatedDocs = await client.getDocumentsInCollection(collectionId, 0, 50);

// Get collections overview with specific collection IDs
const specificCollectionsOverview = await client.collectionsOverview(['id1', 'id2', 'id3']);
```

## Security Considerations

When implementing collection permissions, consider the following security best practices:

1. Always use HTTPS in production to encrypt data in transit.
2. Implement the principle of least privilege by assigning the minimum necessary permissions to users and collections.
3. Regularly audit collection memberships and document assignments.
4. Ensure that only authorized users (e.g., admins) can perform collection management operations.
5. Implement comprehensive logging for all collection-related actions.
6. Consider implementing additional access controls or custom roles within your application logic for more fine-grained permissions.

For more advanced use cases or custom implementations, refer to the R2R documentation or reach out to the community for support.

---

<Warning>This installation guide is for R2R Core. For solo developers or teams prototyping, we highly recommend starting with <a href="/documentation/installation/light/local-system">R2R Light</a>.</Warning>
# R2R Local System Installation

This guide will walk you through installing and running R2R on your local system without using Docker. This method allows for more customization and control over individual components.


<Tip>
  Local installation of R2R Core is challenging due to the numerous services it integrates. We strongly recommend using Docker to get started quickly.

  If you choose to proceed with a local installation, be prepared to set up and configure the following services:

  1. **Postgres with pgvector**: A relational database with vector storage capabilities.
  2. **Unstructured.io**: A complex system for file ingestion.
  4. **Hatchet**: A RabbitMQ-based orchestration system.

  Alternatively, you can use cloud versions of these services, but you'll be responsible for enrolling in them and providing the necessary environment variables.

  Each of these components has its own requirements, potential compatibility issues, and configuration complexities. Debugging issues in a local setup can be significantly more challenging than using a pre-configured Docker environment.

</Tip>

  Unless you have a specific need for a local installation and are comfortable with advanced system configuration, we highly recommend using the Docker setup method for a smoother experience.
## Prerequisites

Before starting, ensure you have the following installed and/or available in the cloud:
- Python 3.12 or higher
- pip (Python package manager)
- Git
- Postgres + pgvector
- Unstructured file ingestion
- Hatchet workflow orchestration

## Install the R2R CLI & Python SDK

First, install the R2R CLI and Python SDK:

```bash
pip install 'r2r[core ingestion-bundle hatchet]'
```

## Environment Setup

R2R requires connections to various services. Set up the following environment variables based on your needs:

<AccordionGroup>
  <Accordion title="Cloud LLM Providers" icon="language">
    Note, cloud providers are optional as R2R can be run entirely locally.
     ```bash
      # Set cloud LLM settings
      export OPENAI_API_KEY=sk-...
      # export ANTHROPIC_API_KEY=...
      # ...
    ```
  </Accordion>
  <Accordion title="Hatchet" icon="axe">
    R2R uses [Hatchet](https://docs.hatchet.run/home) for orchestration. When building R2R locally, you will need to either (a) register for [Hatchet's cloud service](https://cloud.onhatchet.run/auth/login/) or (b) install it [locally following their instructions](https://docs.hatchet.run/self-hosting/docker-compose), or (c) deploy the R2R docker and connect with the internally provisioned Hatchet service.
     ```bash
      # Set cloud LLM settings
      export HATCHET_CLIENT_TOKEN=...
    ```
  </Accordion>
  <Accordion title="Postgres+pgvector" icon="database">
     With R2R you can connect to your own instance of Postgres+pgvector or a remote cloud instance.
     ```bash
      # Set Postgres+pgvector settings
      export R2R_POSTGRES_USER=$YOUR_POSTGRES_USER
      export R2R_POSTGRES_PASSWORD=$YOUR_POSTGRES_PASSWORD
      export R2R_POSTGRES_HOST=$YOUR_POSTGRES_HOST
      export R2R_POSTGRES_PORT=$YOUR_POSTGRES_PORT
      export R2R_POSTGRES_DBNAME=$YOUR_POSTGRES_DBNAME
      export R2R_PROJECT_NAME=$YOUR_PROJECT_NAME # see note below
    ```
    <Note>
    The `R2R_PROJECT_NAME` environment variable defines the tables within your Postgres database where the selected R2R project resides. If the specified tables do not exist then they will be created by R2R during initialization.
    </Note>
  </Accordion>
<Accordion title="Unstructured" icon="puzzle">
  By default, R2R uses [unstructured.io](https://docs.unstructured.io/welcome) to handle file ingestion. Unstructured can be:

  1. Installed locally by following their [documented instructions](https://docs.unstructured.io/open-source/introduction/overview)
  2. Connected to via the cloud

  For cloud connections, set the following environment variable:

  ```bash
  # Set Unstructured API key for cloud usage
  export UNSTRUCTURED_API_KEY=your_api_key_here
  ```

  Alternatively, R2R has its own lightweight end-to-end ingestion, which may be more appropriate in some situations. This can be specified in your server configuration by choosing `r2r` as your ingestion provider.
</Accordion>
</AccordionGroup>

## Running R2R

<Note>The full R2R installation does not use the default [`r2r.toml`](https://github.com/SciPhi-AI/R2R/blob/main/py/r2r.toml), instead it provides overrides through a pre-built custom configuration, [`full.toml`](https://github.com/SciPhi-AI/R2R/blob/main/py/core/configs/full.toml).</Note>

After setting up your environment, you can start R2R using the following command:

```bash
# requires services for unstructured, hatchet, postgres
r2r serve --config-name=full
```

For local LLM usage:

```bash
r2r serve --config-name=full_local_llm
```

## Python Development Mode

For those looking to develop R2R locally:

1. Install Poetry: Follow instructions on the [official Poetry website](https://python-poetry.org/docs/#installation).

2. Clone and install dependencies:
   ```bash
   git clone git@github.com:SciPhi-AI/R2R.git
   cd R2R/py
   poetry install -E "core ingestion-bundle hatchet"
   ```

3. Setup environment:
   Follow the steps listed in the Environment Setup section above. Additionally, you may introduce a local .env file to make development easier, and you can customize your local `r2r.toml` to suit your specific needs.

4. Start your server:
  ```bash
  poetry run r2r serve --config-name=core
  ```

## Next Steps

After successfully installing R2R:

1. **Verify Installation**: Ensure all components are running correctly by accessing the R2R API at http://localhost:7272/v2/health.

2. **Quick Start**: Follow our [R2R Quickstart Guide](/documentation/quickstart) to set up your first RAG application.

3. **In-Depth Tutorial**: For a more comprehensive understanding, work through our [R2R Walkthrough](/cookbooks/walkthrough).

4. **Customize Your Setup**: Configure R2R components with the [Configuration Guide](/documentation/configuration).

If you encounter any issues during installation or setup, please use our [Discord community](https://discord.gg/p6KqD2kjtB) or [GitHub repository](https://github.com/SciPhi-AI/R2R) to seek assistance.

---


## Introduction

R2R uses two key components for assembling and customizing applications: the `R2RBuilder` and a set of factory classes, `R2RFactory*`. These components work together to provide a flexible and intuitive way to construct R2R instances with custom configurations, providers, pipes, and pipelines.

## R2RBuilder

The `R2RBuilder` is the central component for assembling R2R applications. It employs the Builder pattern for simple application customization.

### Key Features

1. **Flexible Configuration**: Supports loading configurations from TOML files or using predefined configurations.
2. **Component Overrides**: Allows overriding default providers, pipes, and pipelines with custom implementations.
3. **Factory Customization**: Supports custom factory implementations for providers, pipes, and pipelines.
4. **Fluent Interface**: Provides a chainable API for easy and readable setup.

### Basic Usage

Here's a simple example of how to use the R2RBuilder:

```python
from r2r import R2RBuilder, R2RConfig

# Create an R2R instance with default configuration
r2r = R2RBuilder().build()

# Create an R2R instance with a custom configuration file
r2r = R2RBuilder(config=R2RConfig.from_toml("path/to/config.toml")).build()

# Create an R2R instance with a predefined configuration
r2r = R2RBuilder(config_name="full").build()
```

## Factories

R2R uses a set of factory classes to create various components of the system. These factories allow for easy customization and extension of R2R's functionality.

### Main Factory Classes

1. **R2RProviderFactory**: Creates provider instances (e.g., DatabaseProvider, EmbeddingProvider).
2. **R2RPipeFactory**: Creates individual pipes used in pipelines.
3. **R2RPipelineFactory**: Creates complete pipelines by assembling pipes.

### Factory Methods

Each factory class contains methods for creating specific components. For example, the `R2RPipeFactory` includes methods like:

- `create_parsing_pipe()`
- `create_embedding_pipe()`
- `create_vector_search_pipe()`
- `create_rag_pipe()`

### Customizing Factories

You can customize the behavior of R2R by extending these factory classes. Here's a simplified example of extending the `R2RPipeFactory`:

```python
from r2r import R2RPipeFactory
from r2r.pipes import MultiSearchPipe, QueryTransformPipe

class R2RPipeFactoryWithMultiSearch(R2RPipeFactory):
    def create_vector_search_pipe(self, *args, **kwargs):
        # Create a custom multi-search pipe
        query_transform_pipe = QueryTransformPipe(
            llm_provider=self.providers.llm,
            config=QueryTransformPipe.QueryTransformConfig(
                name="multi_search",
                task_prompt="multi_search_task_prompt",
            ),
        )

        inner_search_pipe = super().create_vector_search_pipe(*args, **kwargs)

        return MultiSearchPipe(
            query_transform_pipe=query_transform_pipe,
            inner_search_pipe=inner_search_pipe,
            config=MultiSearchPipe.PipeConfig(),
        )
```


## Builder + Factory in action

The R2RBuilder provides methods to override various components of the R2R system:

### Provider Overrides

```python
from r2r.providers import CustomAuthProvider, CustomDatabaseProvider

builder = R2RBuilder()
builder.with_auth_provider(CustomAuthProvider())
builder.with_database_provider(CustomDatabaseProvider())
r2r = builder.build()
```
Available provider override methods:
- `with_auth_provider`
- `with_database_provider`
- `with_embedding_provider`
- `with_eval_provider`
- `with_llm_provider`
- `with_crypto_provider`


### Pipe Overrides

```python
from r2r.pipes import CustomParsingPipe, CustomEmbeddingPipe

builder = R2RBuilder()
builder.with_parsing_pipe(CustomParsingPipe())
builder.with_embedding_pipe(CustomEmbeddingPipe())
r2r = builder.build()
```

Available pipe override methods:
- `with_parsing_pipe`
- `with_embedding_pipe`
- `with_vector_storage_pipe`
- `with_vector_search_pipe`
- `with_rag_pipe`
- `with_streaming_rag_pipe`
- `with_eval_pipe`
- `with_kg_pipe`
- `with_kg_storage_pipe`
- `with_kg_search_pipe`


### Pipeline Overrides

```python
from r2r.pipelines import CustomIngestionPipeline, CustomSearchPipeline

builder = R2RBuilder()
builder.with_ingestion_pipeline(CustomIngestionPipeline())
builder.with_search_pipeline(CustomSearchPipeline())
r2r = builder.build()
```
Available pipeline override methods:
- `with_ingestion_pipeline`
- `with_search_pipeline`
- `with_rag_pipeline`
- `with_streaming_rag_pipeline`
- `with_eval_pipeline`

### Factory Overrides

```python
from r2r.factory import CustomProviderFactory, CustomPipeFactory, CustomPipelineFactory

builder = R2RBuilder()
builder.with_provider_factory(CustomProviderFactory)
builder.with_pipe_factory(CustomPipeFactory)
builder.with_pipeline_factory(CustomPipelineFactory)
r2r = builder.build()
```


## Advanced Usage

For more complex scenarios, you can chain multiple customizations:

```python
from r2r import R2RBuilder, R2RConfig
from r2r.pipes import CustomRAGPipe

class MyCustomAuthProvider:
    ...

class MyCustomLLMProvider
    ...

class MyCustomRAGPipe


config = R2RConfig.from_toml("path/to/config.toml")

r2r = (
    R2RBuilder(config=config)
    .with_auth_provider(MyCustomAuthProvider())
    .with_llm_provider(MyCustomLLMProvider())
    .with_rag_pipe(MyCustomRAGPipe())
    .build()
)
```

This approach allows you to create highly customized R2R instances tailored to your specific needs.

## Best Practices

1. **Configuration Management**: Use separate configuration files for different environments (development, staging, production).
2. **Custom Components**: When creating custom providers, pipes, or pipelines, ensure they adhere to the respective interfaces defined in the R2R framework.
3. **Testability**: Create factory methods or builder configurations specifically for testing to easily mock or stub components.
4. **Logging**: Enable appropriate logging in your custom components to aid in debugging and monitoring.
5. **Error Handling**: Implement proper error handling in custom components and provide meaningful error messages.

By leveraging the R2RBuilder and customizing factories, you can create flexible, customized, and powerful R2R applications tailored to your specific use cases and requirements.

---

# R2R Installation

Welcome to the R2R installation guide. R2R offers powerful features for your RAG applications, including:

- **Flexibility**: Run with cloud-based LLMs or entirely on your local machine
- **State-of-the-Art Tech**: Advanced RAG techniques like [hybrid search](/cookbooks/hybrid-search), [GraphRAG](/cookbooks/graphrag), and [agentic RAG](/cookbooks/agent).
- **Auth & Orchestration**: Production must-haves like [auth](/cookbooks/user-auth) and [ingestion orchestration](/cookbooks/orchestration).

## Choose Your System



<CardGroup cols={2}>
  <Card title="R2R Light" icon="feather" href="/documentation/installation/light">
    A lightweight version of R2R, **perfect for quick prototyping and simpler applications**. Some advanced features, like orchestration and advanced document parsing, may not be available.
  </Card>
  <Card title="R2R" icon="server" href="/documentation/installation/full">
    The full-featured R2R system, ideal **for advanced use cases and production deployments**. Includes all components and capabilities, such as **Hatchet** for orchestration and **Unstructured** for parsing.
  </Card>
</CardGroup>

Choose the system that best aligns with your requirements and proceed with the installation guide.

---

## Introduction

R2R's `DatabaseProvider` offers a unified interface for both relational and vector database operations. R2R only provides database support through Postgres with the pgvector extension.

Postgres was selected to power R2R because it is free, open source, and widely considered be a stable state of the art database. Further, the Postgres community has implemented pgvector for efficient storage and retrieval of vector embeddings alongside traditional relational data.

## Architecture Overview

The Database Provider in R2R is built on two primary components:

1. **Vector Database Provider**: Handles vector-based operations such as similarity search and hybrid search.
2. **Relational Database Provider**: Manages traditional relational database operations, including user management and document metadata storage.

These providers work in tandem to ensure efficient data management and retrieval.

## Configuration

Update the `database` section in your `r2r.toml` file:

```toml
[database]
provider = "postgres"
user = "your_postgres_user"
password = "your_postgres_password"
host = "your_postgres_host"
port = "your_postgres_port"
db_name = "your_database_name"
your_project_name = "your_project_name"
```

Alternatively, you can set these values using environment variables:

```bash
export R2R_POSTGRES_USER=your_postgres_user
export R2R_POSTGRES_PASSWORD=your_postgres_password
export R2R_POSTGRES_HOST=your_postgres_host
export R2R_POSTGRES_PORT=your_postgres_port
export R2R_POSTGRES_DBNAME=your_database_name
export R2R_PROJECT_NAME=your_project_name
```
Environment variables take precedence over the config settings in case of conflicts. The R2R Docker includes configuration options that facilitate integration with a combined Postgres+pgvector database setup.

## Vector Database Operations

### Initialization

The vector database is automatically initialized with dimensions that correspond to your selected embedding model when the Database Provider is first created.

### Upsert Vector Entries

```python
from r2r import R2R, VectorEntry

app = R2R()

vector_entry = VectorEntry(id="unique_id", vector=[0.1, 0.2, 0.3], metadata={"key": "value"})
app.providers.database.vector.upsert(vector_entry)
```

### Search

```python
results = app.providers.database.vector.search(query_vector=[0.1, 0.2, 0.3], limit=10)
```

### Hybrid Search

```python
results = app.providers.database.vector.hybrid_search(
    query_text="search query",
    query_vector=[0.1, 0.2, 0.3],
    limit=10,
    full_text_weight=1.0,
    semantic_weight=1.0
)
```

### Delete by Metadata

```python
deleted_ids = app.providers.database.vector.delete_by_metadata(
    metadata_fields=["key1", "key2"],
    metadata_values=["value1", "value2"]
)
```

## Relational Database Operations

### User Management

#### Create User

```python
from r2r import UserCreate

new_user = UserCreate(email="user@example.com", password="secure_password")
created_user = app.providers.database.relational.create_user(new_user)
```

#### Get User by Email

```python
user = app.providers.database.relational.get_user_by_email("user@example.com")
```

#### Update User

```python
user.name = "New Name"
updated_user = app.providers.database.relational.update_user(user)
```

### Document Management

#### Upsert Document Overview

```python
from r2r import DocumentInfo

doc_info = DocumentInfo(
    document_id="doc_id",
    title="Document Title",
    user_id="user_id",
    version="1.0",
    size_in_bytes=1024,
    metadata={"key": "value"},
    status="processed"
)
app.providers.database.relational.upsert_documents_overview([doc_info])
```

#### Get Documents Overview

```python
docs = app.providers.database.relational.get_documents_overview(
    filter_document_ids=["doc_id1", "doc_id2"],
    filter_user_ids=["user_id1", "user_id2"]
)
```

## Advanced Features

### Hybrid Search Function

The Database Provider includes a custom Postgres function for hybrid search, combining full-text and vector similarity search.

### Token Blacklisting

For enhanced security, the provider supports token blacklisting:

```python
app.providers.database.relational.blacklist_token("token_to_blacklist")
is_blacklisted = app.providers.database.relational.is_token_blacklisted("token_to_check")
```

## Security Best Practices

1. **Environment Variables**: Use environment variables for sensitive information like database credentials.
2. **Connection Pooling**: The provider uses connection pooling for efficient database connections.
3. **Prepared Statements**: SQL queries use prepared statements to prevent SQL injection.
4. **Password Hashing**: User passwords are hashed before storage using the configured Crypto Provider.

## Performance Considerations

1. **Indexing**: The vector database automatically creates appropriate indexes for efficient similarity search.
2. **Batch Operations**: Use batch operations like `upsert_entries` for better performance when dealing with multiple records.
3. **Connection Reuse**: The provider reuses database connections to minimize overhead.

## Troubleshooting

Common issues and solutions:

1. **Connection Errors**: Ensure your database credentials and connection details are correct.
2. **Dimension Mismatch**: Verify that the vector dimension in your configuration matches your embeddings.
3. **Performance Issues**: Consider optimizing your queries or adding appropriate indexes.

## Conclusion

R2R's Database Provider offers a powerful and flexible solution for managing both vector and relational data. By leveraging Postgres with pgvector, it provides efficient storage and retrieval of embeddings alongside traditional relational data, enabling advanced search capabilities and robust user management.

---


## Introduction

R2R is a an engine for building user-facing Retrieval-Augmented Generation (RAG) applications. At its core, R2R uses several main components to create and manage these applications: `R2RConfig`, `R2RBuilder`, `R2REngine`, `R2RApp`, and `R2R`. In this section we will explore all of these components in detail.

## Assembly Process

The following diagram illustrates how R2R assembles a user-facing application:

```mermaid
flowchart TD
    subgraph "Assembly"
        direction TB
        Config[R2RConfig]
        Builder[R2RBuilder]

        subgraph Factories
            PF[R2RProviderFactory]
            PiF[R2RPipeFactory]
            PlF[R2RPipelineFactory]
        end

        Engine[Orchestration]
        API[API]

        Config --> Builder
        Builder --> PF
        Builder --> PiF
        Builder --> PlF
        PF --> Engine
        PiF --> Engine
        PlF --> Engine
        Engine --> API
    end

    User[Developer] --> |Customizes| Config
    User --> |Uses| Builder
    User --> |Overrides| Factories
```

## R2RConfig

R2RConfig is the configuration management class for R2R applications. It serves as the central point for defining and managing all settings required by various components of the R2R system.

Key features of R2RConfig:

1. **Configurable**: R2RConfig loads its settings from a TOML file, making it easy to modify and version control.
2. **Hierarchical Structure**: The configuration is organized into sections, each corresponding to a specific aspect of the R2R system (e.g., embedding, database, logging).
3. **Default Values**: R2RConfig provides a set of default values, which can be overridden as needed.
4. **Validation**: The class includes methods to validate the configuration, ensuring all required settings are present.
5. **Type-safe Access**: Once loaded, the configuration provides type-safe access to settings, reducing the risk of runtime errors.

Example usage:
```python
config = R2RConfig.from_toml("path/to/config.toml")
embedding_model = config.embedding.base_model
max_file_size = config.app.max_file_size_in_mb
```

## R2RBuilder

R2RBuilder is the central component for assembling R2R applications, using the builder pattern. It provides a flexible way to construct and customize the R2R system.

Key features of R2RBuilder:

1. **Fluent Interface**: Allows chaining of configuration methods for easy setup.
2. **Component Overrides**: Provides methods to override default providers, pipes, and pipelines.
3. **Factory Customization**: Supports custom factory implementations for providers, pipes, and pipelines.
4. **Configuration Integration**: Uses R2RConfig to set up the system according to specified settings.

Example usage:
```python
builder = R2RBuilder(config=config)
builder.with_embedding_provider(custom_embedding_provider)
builder.with_llm_provider(custom_llm_provider)
r2r = builder.build()
```

## R2RApp

R2RApp is the class responsible for setting up the FastAPI application that serves as the interface for the R2R system. It encapsulates the routing and CORS configuration for the R2R API.

Key features of R2RApp:

1. **FastAPI Integration**: Creates and configures a FastAPI application, setting up all necessary routes.
2. **Route Setup**: Automatically sets up routes for ingestion, management, retrieval, and authentication based on the provided R2REngine.
3. **CORS Configuration**: Applies CORS (Cross-Origin Resource Sharing) settings to the FastAPI application.
4. **Serving Capability**: Includes a `serve` method to easily start the FastAPI server.

Example usage:
```python
app = r2r.app
app.serve(host="0.0.0.0", port=7272)
```


By combining these core components, R2R provides a flexible and powerful system for building and deploying RAG applications. The R2RConfig allows for easy customization while the R2RBuilder facilitates the assembly process. Lastly, the R2RApp offers a robust API interface.

---

# R2R Docker Installation

This guide will walk you through installing and running R2R using Docker, which is the quickest and easiest way to get started.

## Prerequisites

- Docker installed on your system. If you haven't installed Docker yet, please refer to the [official Docker installation guide](https://docs.docker.com/engine/install/).

## Install the R2R CLI & Python SDK

First, install the R2R CLI and Python SDK:

```bash
pip install r2r
```

<Info>We are actively developing a distinct CLI binary for R2R for easier installation. Please reach out if you have any specific needs or feature requests.</Info>

## Start R2R with Docker

<AccordionGroup>
  <Accordion title="Cloud LLM RAG" icon="cloud" defaultOpen={true}>
  To start R2R with OpenAI as the default LLM inference and embedding provider:
      ```bash
      # Set cloud LLM settings
      export OPENAI_API_KEY=sk-...

      r2r serve --docker
      ```
  [Refer here](/documentation/configuration/llm) for more information on how to configure various LLM providers.
  </Accordion>
  <Accordion title="Local LLMs" icon="house" defaultOpen={false}>
      To start R2R with your local computer as the default LLM inference provider:
      ```bash
      r2r serve --docker --config-name=local_llm
      ```
    Then, in a separate terminal you will need to run Ollama to provide completions:
    ```bash
      ollama pull llama3.1
      ollama pull mxbai-embed-large
      ollama serve
    ```
    The code above assumes that Ollama has already been installed. If you have not yet done so, then refer to the official Ollama webpage [for installation instructions](https://ollama.com/download). For more information on local installation, [refer here](/documentation/local-rag).
  </Accordion>
  <Accordion title="Custom Configuration" icon="gear" defaultOpen={false}>
    R2R offers flexibility in selecting and configuring LLMs, allowing you to optimize your RAG pipeline for various use cases. Execute the command below run deploy R2R with your own custom configuration:
      ```bash
      r2r serve --config-path=/abs/path/to/my_r2r.toml
      ```

    Learn in detail how to [configure your deployment here](/documentation/configuration).
  </Accordion>
</AccordionGroup>

<Note>
Postgres comes bundled into the R2R Docker by default.
</Note>

The above command will automatically pull the necessary Docker images and start all the required containers, including `R2R`, `Postgres+pgvector`.

The end result is a live server at http://localhost:7272 serving the [R2R API](/api-reference/introduction).

In addition to launching a RESTful API, the R2R Docker also launches a applications at `localhost:7273` and `localhost:7274`, which you can [read more about here](/cookbooks/application).

### Stopping R2R

Safely stop your system by running `r2r docker-down` to avoid potential shutdown complications.

## Next Steps

After successfully installing R2R:

1. **Verify Installation**: Ensure all components are running correctly by accessing the R2R API at http://localhost:7272/v2/health.

2. **Quick Start**: Follow our [R2R Quickstart Guide](/documentation/quickstart) to set up your first RAG application.

3. **In-Depth Tutorial**: For a more comprehensive understanding, work through our [R2R Walkthrough](/cookbooks/walkthrough).

4. **Customize Your Setup**: Configure R2R components with the [Configuration Guide](/documentation/configuration).

If you encounter any issues during installation or setup, please use our [Discord community](https://discord.gg/p6KqD2kjtB) or [GitHub repository](https://github.com/SciPhi-AI/R2R) to seek assistance.

---


<Note>
Occasionally this SDK documentation falls out of date, cross-check with the automatcially generated <a href="/api-reference/introduction"> API Reference documentation </a> for the latest parameters.
</Note>


## User Authentication and Management

R2R provides a comprehensive set of user authentication and management features, allowing you to implement secure and feature-rich authentication systems in your applications.

### User Registration

To register a new user:

```javascript
const registerResponse = await client.register("user@example.com", "password123");
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      ```javascript
      {
        results: {
          email: 'user@example.com',
          id: 'bf417057-f104-4e75-8579-c74d26fcbed3',
          hashed_password: '$2b$12$p6a9glpAQaq.4uzi4gXQru6PN7WBpky/xMeYK9LShEe4ygBf1L.pK',
          is_superuser: false,
          is_active: true,
          is_verified: false,
          verification_code_expiry: null,
          name: null,
          bio: null,
          profile_picture: null,
          created_at: '2024-07-16T22:53:47.524794Z',
          updated_at: '2024-07-16T22:53:47.524794Z'
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Email Verification

If email verification is enabled, verify a user's email:

```javascript
const verifyResponse = await client.verifyEmail("verification_code_here");
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      ```javascript
      {
        results: {
          message: "Email verified successfully"
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### User Login

To log in and obtain access tokens:

```javascript
const loginResponse = await client.login("user@example.com", "password123");
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      ```javascript
      {
        results: {
          access_token: {
            token: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...',
            token_type: 'access'
          },
          refresh_token: {
            token: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...',
            token_type: 'refresh'
          }
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Get Current User Info

Retrieve information about the currently authenticated user:

```javascript
const user_info = client.user()
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      ```python
      {
        'results': {
          'email': 'user@example.com',
          'id': '76eea168-9f98-4672-af3b-2c26ec92d7f8',
          'hashed_password': 'null',
          'is_superuser': False,
          'is_active': True,
          'is_verified': True,
          'verification_code_expiry': None,
          'name': None,
          'bio': None,
          'profile_picture': None,
          'created_at': '2024-07-16T23:06:42.123303Z',
          'updated_at': '2024-07-16T23:22:48.256239Z'
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Refresh Access Token

Refresh an expired access token:

```javascript
const refreshResponse = await client.refreshAccessToken();
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      ```javascript
      {
        results: {
          access_token: {
            token: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...',
            token_type: 'access'
          },
          refresh_token: {
            token: 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...',
            token_type: 'refresh'
          }
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Change Password

Change the user's password:

```javascript
const changePasswordResult = await client.changePassword("password123", "new_password");
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      ```javascript
      {
        results: {
          message: "Password changed successfully"
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Request Password Reset

Request a password reset for a user:

```javascript
const resetRequestResult = await client.requestPasswordReset("user@example.com");
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      ```javascript
      {
        results: {
          message: "If the email exists, a reset link has been sent"
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Confirm Password Reset

Confirm a password reset using the reset token:

```javascript
const resetConfirmResult = await client.confirmPasswordReset("reset_token_here", "new_password");
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      ```javascript
      {
        results: {
          message: "Password reset successfully"
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Update User Profile

Update the user's profile information:

```javascript
// keeping the user's email as is:
const updateResult = client.updateUser(undefined, "John Doe", "R2R enthusiast");
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      ```python
      {
        'results': {
          'email': 'user@example.com',
          'id': '76eea168-9f98-4672-af3b-2c26ec92d7f8',
          'hashed_password': 'null',
          'is_superuser': False,
          'is_active': True,
          'is_verified': True,
          'verification_code_expiry': None,
          'name': 'John Doe',
          'bio': 'R2R enthusiast',
          'profile_picture': None,
          'created_at': '2024-07-16T23:06:42.123303Z',
          'updated_at': '2024-07-16T23:22:48.256239Z'
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Delete User Account

Delete the user's account:

```javascript
const user_id = register_response["results"]["id"] # input unique id here
const delete_result = client.delete_user(user_id, "password123")
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      ```python
      {
        'results': {
          'message': 'User account deleted successfully'
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### User Logout

Log out and invalidate the current access token:

```javascript
const logoutResponse = await client.logout();
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      ```javascript
      {
        results: {
          message: "Logged out successfully"
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Superuser Capabilities

Superusers have additional privileges, including access to system-wide operations and sensitive information. To use superuser capabilities, authenticate as a superuser or the default admin:

```javascript
// Login as admin
const loginResult = await client.login("admin@example.com", "admin_password");

// Access superuser features
const usersOverview = await client.usersOverview();
const logs = await client.logs();
const analyticsResult = await client.analytics(
  { filters: { all_latencies: "search_latency" } },
  { analysis_types: { search_latencies: ["basic_statistics", "search_latency"] } }
);
```

<Note>
Superuser actions should be performed with caution and only by authorized personnel. Ensure proper security measures are in place when using superuser capabilities.
</Note>

## Security Considerations

When implementing user authentication, consider the following best practices:

1. Always use HTTPS in production to encrypt data in transit.
2. Implement rate limiting to protect against brute-force attacks.
3. Use secure password hashing (R2R uses bcrypt by default).
4. Consider implementing multi-factor authentication (MFA) for enhanced security.
5. Conduct regular security audits of your authentication system.

For more advanced use cases or custom implementations, refer to the R2R documentation or reach out to the community for support.

---


# R2R Python SDK Documentation

## Installation

Before starting, make sure you have completed the [R2R installation](/documentation/installation).

Install the R2R Python SDK:

```bash
pip install r2r
```

## Getting Started

1. Import the R2R client:

```python
from r2r import R2RClient
```

2. Initialize the client:

```python
client = R2RClient("http://localhost:7272")
```


3. Check if R2R is running correctly:

```python
health_response = client.health()
# {"status":"ok"}
```

4. Login (Optional):
```python
client.register("me@email.com", "my_password")
# client.verify_email("me@email.com", "my_verification_code")
client.login("me@email.com", "my_password")
```
When using authentication the commands below automatically restrict the scope to a user's available documents.

## Additional Documentation

For more detailed information on specific functionalities of R2R, please refer to the following documentation:

- [Document Ingestion](/documentation/python-sdk/ingestion): Learn how to add, retrieve, and manage documents in R2R.
- [Search & RAG](/documentation/python-sdk/retrieval): Explore various querying techniques and Retrieval-Augmented Generation capabilities.
- [Authentication](/documentation/python-sdk/auth): Understand how to manage users and implement authentication in R2R.
- [Observability](/documentation/python-sdk/observability): Learn about analytics and monitoring tools for your R2R system.

---


<Warning>
This feature is currently in beta. Functionality may change, and we value your feedback around these features.
</Warning>

<Note>
Occasionally this SDK documentation falls out of date, cross-check with the automatically generated <a href="/api-reference/introduction"> API Reference documentation </a> for the latest parameters.
</Note>

## Conversation Management

### Get Conversations Overview

Retrieve an overview of existing conversations:

```javascript
const offset = 0;
const limit = 10;
const overviewResponse = await client.conversationsOverview(offset, limit);
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      The response containing an overview of conversations.
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="offset" type="number">
  The offset to start listing conversations from.
</ParamField>

<ParamField path="limit" type="number">
  The maximum number of conversations to return.
</ParamField>

### Get Conversation

Fetch a specific conversation by its UUID:

```javascript
const conversationId = '123e4567-e89b-12d3-a456-426614174000';
const conversation = await client.getConversation(conversationId);
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      The response containing the requested conversation details.
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="conversationId" type="string" required>
  The UUID of the conversation to retrieve.
</ParamField>

### Create Conversation

Create a new conversation:

```javascript
const newConversation = await client.createConversation();
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      The response containing details of the newly created conversation.
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Add Message

Add a message to an existing conversation:

```javascript
const conversationId = '123e4567-e89b-12d3-a456-426614174000';
const message = { text: 'Hello, world!' };
const parentId = '98765432-e21b-12d3-a456-426614174000';
const metadata = { key: 'value' };

const addMessageResponse = await client.addMessage(conversationId, message, parentId, metadata);
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      The response after adding the message to the conversation.
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="conversationId" type="string" required>
  The UUID of the conversation to add the message to.
</ParamField>

<ParamField path="message" type="object" required>
  The message object to add to the conversation.
</ParamField>

<ParamField path="parentId" type="string">
  An optional UUID of the parent message.
</ParamField>

<ParamField path="metadata" type="object">
  An optional metadata object for the message.
</ParamField>

### Update Message

Update an existing message in a conversation:

```javascript
const messageId = '98765432-e21b-12d3-a456-426614174000';
const updatedMessage = { text: 'Updated message content' };

const updateMessageResponse = await client.updateMessage(messageId, updatedMessage);
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      The response after updating the message.
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="messageId" type="string" required>
  The UUID of the message to update.
</ParamField>

<ParamField path="message" type="object" required>
  The updated message object.
</ParamField>

### Get Branches Overview

Retrieve an overview of branches in a conversation:

```javascript
const conversationId = '123e4567-e89b-12d3-a456-426614174000';
const branchesOverview = await client.branchesOverview(conversationId);
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      The response containing an overview of branches in the conversation.
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="conversationId" type="string" required>
  The UUID of the conversation to get branches for.
</ParamField>

### Delete Conversation

Delete a conversation by its UUID:

```javascript
const conversationId = '123e4567-e89b-12d3-a456-426614174000';
const deleteResponse = await client.deleteConversation(conversationId);
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      The response after deleting the conversation.
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="conversationId" type="string" required>
  The UUID of the conversation to delete.
</ParamField>

---


## Analytics and Observability
R2R provides various tools for analytics and observability to help you monitor and improve the performance of your RAG system.

```javascript
const scoringResponse = await client.login("message_id_here", "password123");
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      The response from the R2R system.
      ```javascript
      { 'results': 'ok' }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>


<ParamField path="message_id" type="string" required>
The unique identifier of the completion message to be scored. Found in logs for a message.
</ParamField>
<ParamField path="score" type="float" required>
The score to assign to the completion, ranging from -1 to 1.
</ParamField>

---


<Note>
Occasionally this SDK documentation falls out of date, if in doubt, cross-check with the automatcially generated <a href="/api-reference/introduction"> API Reference documentation </a>.
</Note>

## AI Powered Search

### Search

Perform a basic vector search:

```javascript
const searchResponse = await client.search("What was Uber's profit in 2020?");
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      The search results from the R2R system.
      ```javascript
      {
        results: {
          vector_search_results: [
            {
              id: '7ed3a01c-88dc-5a58-a68b-6e5d9f292df2',
              score: 0.780314067545999,
              metadata: {
                text: 'Content of the chunk...',
                title: 'file1.txt',
                version: 'v0',
                chunk_order: 0,
                document_id: 'c9bdbac7-0ea3-5c9e-b590-018bd09b127b',
                extraction_id: '472d6921-b4cd-5514-bf62-90b05c9102cb',
                // ...
              }
            }
            // ...
          ],
          kg_search_results: null
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="query" type="string" required>
  The search query.
</ParamField>

<ParamField path="vector_search_settings" type="VectorSearchSettings | Record<string, any>" default="None">
  Optional settings for vector search, either a dictionary, a `VectorSearchSettings` object, or `None` may be passed. If a dictionary or `None` is passed, then R2R will use server-side defaults for non-specified fields.

  <Expandable title="properties">
    <ParamField path="use_vector_search" type="boolean" default="True">
    Whether to use vector search.
    </ParamField>

    <ParamField path="use_hybrid_search" type="boolean" default="False">
    Whether to perform a hybrid search (combining vector and keyword search).
    </ParamField>

    <ParamField path="filters" type="Record<string, any>" default="{}">
      Commonly seen filters include operations include the following:

      `{"document_id": {"$eq": "9fbe403b-..."}}`

      `{"document_id": {"$in": ["9fbe403b-...", "3e157b3a-..."]}}`

      `{"collection_ids": {"$overlap": ["122fdf6a-...", "..."]}}`

      `{"$and": {"$document_id": ..., "collection_ids": ...}}`
    </ParamField>

    <ParamField path="search_limit" type="number" default="10">
    Maximum number of results to return (1-1000).
    </ParamField>

    <ParamField path="selected_collection_ids" type="Array<string | null>">
    Collection Ids to search for.
    </ParamField>
    <ParamField path="index_measure" type="string" default="cosine_distance">
    The distance measure to use for indexing (cosine_distance, l2_distance, or max_inner_product).
    </ParamField>

    <ParamField path="include_values" type="boolean" default="true">
    Whether to include search score values in the search results.
    </ParamField>

    <ParamField path="include_metadatas" type="boolean" default="true">
    Whether to include element metadata in the search results.
    </ParamField>

    <ParamField path="probes" type="number" default="10">
    Number of ivfflat index lists to query. Higher increases accuracy but decreases speed.
    </ParamField>

    <ParamField path="ef_search" type="number" default="40">
    Size of the dynamic candidate list for HNSW index search. Higher increases accuracy but decreases speed.
    </ParamField>
  </Expandable>
</ParamField>
<ParamField path="kg_search_settings" type="Optional[Union[KGSearchSettings, dict]]" default="None">
  Optional settings for knowledge graph search, either a dictionary, a `KGSearchSettings` object, or `None` may be passed. If a dictionary or `None` is passed, then R2R will use server-side defaults for non-specified fields.

  <Expandable title="properties">

    <ParamField path="filters" type="dict[str, Any]" default="{}">
      Alias for `search_filters`, now `deprecated`.
    </ParamField>

    <ParamField path="search_filters" type="dict[str, Any]" default="{}">
      Filters to apply to the vector search. Allowed operators include `eq`, `neq`, `gt`, `gte`, `lt`, `lte`, `like`, `ilike`, `in`, and `nin`.

      Commonly seen filters include operations include the following:

        `{"document_id": {"$eq": "9fbe403b-..."}}`

        `{"document_id": {"$in": ["9fbe403b-...", "3e157b3a-..."]}}`

        `{"collection_ids": {"$overlap": ["122fdf6a-...", "..."]}}`

        `{"$and": {"$document_id": ..., "collection_ids": ...}}`
    </ParamField>



    <ParamField path="selected_collection_ids" type="list[UUID]">
      Collection IDs to search for.
    </ParamField>

    <ParamField path="graphrag_map_system_prompt" type="str" default="graphrag_map_system_prompt">
      The system prompt for the GraphRAG map prompt.
    </ParamField>

    <ParamField path="graphrag_reduce_system_prompt" type="str" default="graphrag_reduce_system_prompt">
      The system prompt for the GraphRAG reduce prompt.
    </ParamField>

    <ParamField path="use_kg_search" type="bool" default="False">
      Whether to use knowledge graph search.
    </ParamField>

    <ParamField path="kg_search_type" type="str" default="local">
      The type of knowledge graph search to perform. Supported value: "local".
    </ParamField>

    <ParamField path="kg_search_level" type="Optional[str]" default="None">
      The level of knowledge graph search to perform.
    </ParamField>

    <ParamField path="generation_config" type="GenerationConfig">
      Configuration for text generation during graph search.
    </ParamField>

    <ParamField path="max_community_description_length" type="int" default="65536">
      The maximum length of the community description.
    </ParamField>

    <ParamField path="max_llm_queries_for_global_search" type="int" default="250">
      The maximum number of LLM queries to perform during global search.
    </ParamField>

    <ParamField path="local_search_limits" type="dict[str, int]">
      The local search limits for different entity types. The default values are:
      - `"__Entity__"`: 20
      - `"__Relationship__"`: 20
      - `"__Community__"`: 20
    </ParamField>

  </Expandable>
</ParamField>


### Search custom settings

Search with custom settings, such as bespoke document filters and larger search limits
```javascript
# returns only chunks from documents with title `document_title`
const filtered_search_response = client.search(
    "What was Uber's profit in 2020?",
    { filters: {
        $eq: "uber_2021.pdf"
    },
    search_limit: 100
    },
)
```


### Hybrid Search

Combine traditional keyword-based search with vector search:

```javascript
const hybrid_search_response = client.search(
    "What was Uber's profit in 2020?",
    { filters: {
        $eq: "uber_2021.pdf"
    },
    search_limit: 100
    },
)
```

```javascript
hybrid_search_response = client.search(
    "What was Uber's profit in 2020?",
    { use_hybrid_search: true}
)
```

### Knowledge Graph Search

Utilize knowledge graph capabilities to enhance search results:

```javascript
const kg_search_response = client.search(
    "What is a fierce nerd?",
    { use_kg_search: true },
)
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
      The knowledge graph search results from the R2R system.
      ```python
      {
        'results': {
          'vector_search_results': [],
          'kg_search_results': [
            ('MATCH (p:PERSON)-[:FOUNDED]->(c:COMPANY) WHERE c.name = "Airbnb" RETURN p.name',
             [{'name': 'Brian Chesky'}, {'name': 'Joe Gebbia'}, {'name': 'Nathan Blecharczyk'}])
          ]
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

## Retrieval-Augmented Generation (RAG)

### Basic RAG

Generate a response using RAG:

```javascript
const ragResponse = await client.rag("What was Uber's profit in 2020?");
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      The RAG response from the R2R system.
      ```python
      {
        'results': {
          'completion': {
            'id': 'chatcmpl-9ySStnC0oEhnGPPV1k8ZYnxBKOuW8',
            'choices': [{
              'finish_reason': 'stop',
              'index': 0,
              'logprobs': None,
              'message': {
                'content': "Uber's profit in 2020 was a net loss of $6.77 billion."
              },
              ...
            }]
          },
          'search_results': {
            'vector_search_results': [...],
            'kg_search_results': None
          }
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="query" type="str" required>
  The query for RAG.
</ParamField>

<ParamField path="vector_search_settings" type="Optional[Union[VectorSearchSettings, dict]]" default="None">
  Optional settings for vector search, either a dictionary, a `VectorSearchSettings` object, or `None` may be passed. If a dictionary is used, non-specified fields will use the server-side default.

  <Expandable title="properties">
    <ParamField path="use_vector_search" type="bool" default="True">
    Whether to use vector search.
    </ParamField>

    <ParamField path="use_hybrid_search" type="bool" default="False">
    Whether to perform a hybrid search (combining vector and keyword search).
    </ParamField>

    <ParamField path="filters" type="dict[str, Any]" default="{}">
    Alias for `search_filters`, now `deprecated`.
    </ParamField>

    <ParamField path="search_filters" type="dict[str, Any]" default="{}">
      Filters to apply to the vector search. Allowed operators include `eq`, `neq`, `gt`, `gte`, `lt`, `lte`, `like`, `ilike`, `in`, and `nin`.

      Commonly seen filters include operations include the following:

        `{"document_id": {"$eq": "9fbe403b-..."}}`

        `{"document_id": {"$in": ["9fbe403b-...", "3e157b3a-..."]}}`

        `{"collection_ids": {"$overlap": ["122fdf6a-...", "..."]}}`

        `{"$and": {"$document_id": ..., "collection_ids": ...}}`
    </ParamField>

    <ParamField path="search_limit" type="int" default="10">
    Maximum number of results to return (1-1000).
    </ParamField>

    <ParamField path="selected_collection_ids" type="list[UUID]" default="[]">
    Collection Ids to search for.
    </ParamField>

    <ParamField path="index_measure" type="IndexMeasure" default="cosine_distance">
    The distance measure to use for indexing (cosine_distance, l2_distance, or max_inner_product).
    </ParamField>

    <ParamField path="include_values" type="bool" default="True">
    Whether to include search score values in the search results.
    </ParamField>

    <ParamField path="include_metadatas" type="bool" default="True">
    Whether to include element metadata in the search results.
    </ParamField>

    <ParamField path="probes" type="Optional[int]" default="10">
    Number of ivfflat index lists to query. Higher increases accuracy but decreases speed.
    </ParamField>

    <ParamField path="ef_search" type="Optional[int]" default="40">
    Size of the dynamic candidate list for HNSW index search. Higher increases accuracy but decreases speed.
    </ParamField>

    <ParamField path="hybrid_search_settings" type="Optional[HybridSearchSettings]" default="HybridSearchSettings()">
    Settings for hybrid search.
    <Expandable title="properties">
      <ParamField path="full_text_weight" type="float" default="1.0">
      Weight to apply to full text search.
      </ParamField>

      <ParamField path="semantic_weight" type="float" default="5.0">
      Weight to apply to semantic search.
      </ParamField>

      <ParamField path="full_text_limit" type="int" default="200">
      Maximum number of results to return from full text search.
      </ParamField>

      <ParamField path="rrf_k" type="int" default="50">
      K-value for RRF (Rank Reciprocal Fusion).
      </ParamField>
    </Expandable>
    </ParamField>
  </Expandable>
</ParamField>

<ParamField path="kg_search_settings" type="Optional[Union[KGSearchSettings, dict]]" default="None">
  Optional settings for knowledge graph search, either a dictionary, a `KGSearchSettings` object, or `None` may be passed. If a dictionary or `None` is passed, then R2R will use server-side defaults for non-specified fields.
  <Expandable title="properties">

  The `KGSearchSettings` class allows you to configure the knowledge graph search settings for your R2R system. Here are the available options:

    <ParamField path="filters" type="dict[str, Any]" default="{}">
      Alias for `search_filters`, now `deprecated`.
    </ParamField>

    <ParamField path="search_filters" type="dict[str, Any]" default="{}">
      Filters to apply to the vector search. Allowed operators include `eq`, `neq`, `gt`, `gte`, `lt`, `lte`, `like`, `ilike`, `in`, and `nin`.

      Commonly seen filters include operations include the following:

        `{"document_id": {"$eq": "9fbe403b-..."}}`

        `{"document_id": {"$in": ["9fbe403b-...", "3e157b3a-..."]}}`

        `{"collection_ids": {"$overlap": ["122fdf6a-...", "..."]}}`

        `{"$and": {"$document_id": ..., "collection_ids": ...}}`
    </ParamField>

  <ParamField path="selected_collection_ids" type="list[UUID]">
    Collection IDs to search for.
  </ParamField>

  <ParamField path="graphrag_map_system_prompt" type="str" default="graphrag_map_system_prompt">
    The system prompt for the GraphRAG map prompt.
  </ParamField>

  <ParamField path="graphrag_reduce_system_prompt" type="str" default="graphrag_reduce_system_prompt">
    The system prompt for the GraphRAG reduce prompt.
  </ParamField>

  <ParamField path="use_kg_search" type="bool" default="False">
    Whether to use knowledge graph search.
  </ParamField>

  <ParamField path="kg_search_type" type="str" default="local">
    The type of knowledge graph search to perform. Supported value: "local".
  </ParamField>

  <ParamField path="kg_search_level" type="Optional[str]" default="None">
    The level of knowledge graph search to perform.
  </ParamField>

  <ParamField path="generation_config" type="GenerationConfig">
    Configuration for text generation during graph search.
  </ParamField>

  <ParamField path="max_community_description_length" type="int" default="65536">
    The maximum length of the community description.
  </ParamField>

  <ParamField path="max_llm_queries_for_global_search" type="int" default="250">
    The maximum number of LLM queries to perform during global search.
  </ParamField>

  <ParamField path="local_search_limits" type="dict[str, int]">
    The local search limits for different entity types. The default values are:
    - `"__Entity__"`: 20
    - `"__Relationship__"`: 20
    - `"__Community__"`: 20
  </ParamField>
  </Expandable>


</ParamField>

<ParamField path="rag_generation_config" type="Optional[Union[GenerationConfig, dict]]" default="None">
  Optional configuration for LLM to use during RAG generation, including model selection and parameters. Will default to values specified in `r2r.toml`.
  <Expandable title="properties">
    <ParamField path="model" type="str" default="openai/gpt-4o">
    Model used in final LLM completion.
    </ParamField>

    <ParamField path="temperature" type="float" default="0.1">
    Temperature used in final LLM completion.
    </ParamField>

    <ParamField path="top_p" type="float" default="1.0">
    The `top_p` used in final LLM completion.
    </ParamField>

    <ParamField path="max_tokens_to_sample" type="int" default="1024">
    The `max_tokens_to_sample` used in final LLM completion.
    </ParamField>

    <ParamField path="functions" type="dict" default="None">
    The `functions` used in final LLM completion.
    </ParamField>

    <ParamField path="tools" type="dict" default="None">
    The `tools` used in final LLM completion.
    </ParamField>

    <ParamField path="api_base" type="str" default="None">
    The `api_base` used in final LLM completion.
    </ParamField>

    <ParamField path="functions" type="Record<string, any>" default="None">
    The `functions` used in final LLM completion.
    </ParamField>

    <ParamField path="tools" type="Record<string, any>" default="None">
    The `tools` used in final LLM completion.
    </ParamField>

    <ParamField path="api_base" type="string" default="None">
    The `api_base` used in final LLM completion.
    </ParamField>

  </Expandable>


</ParamField>
<ParamField path="task_prompt_override" type="Optional[str]" default="None">
  Optional custom prompt to override the default task prompt.
</ParamField>

<ParamField path="include_title_if_available" type="Optional[bool]" default="True">
  Augment document chunks with their respective document titles?
</ParamField>

### RAG with custom search settings

Use hybrid search in RAG:

```javascript
const hybridRagResponse = await client.rag({
  query: "Who is Jon Snow?",
  use_hybrid_search: true
});
```

### RAG with custom completion LLM

Use a different LLM model for RAG:

```javascript
const customLLMRagResponse = await client.rag({
  query: "What is R2R?",
  rag_generation_config: {
    model: "anthropic/claude-3-opus-20240229"
  }
});
```

### Streaming RAG

Stream RAG responses for real-time applications:

```javascript
const streamResponse = await client.rag({
  query: "Who was Aristotle?",
  rag_generation_config: { stream: true }
});

if (streamResponse instanceof ReadableStream) {
  const reader = streamResponse.getReader();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    console.log(new TextDecoder().decode(value));
  }
}
```
### Advanced RAG Techniques

R2R supports advanced Retrieval-Augmented Generation (RAG) techniques that can be easily configured at runtime. These techniques include Hypothetical Document Embeddings (HyDE) and RAG-Fusion, which can significantly enhance the quality and relevance of retrieved information.

To use an advanced RAG technique, you can specify the `search_strategy` parameter in your vector search settings:

```javascript
const client = new R2RClient();

// Using HyDE
const hydeResponse = await client.rag(
    "What are the main themes in Shakespeare's plays?",
    {
        vector_search_settings: {
            search_strategy: "hyde",
            search_limit: 10
        }
    }
);

// Using RAG-Fusion
const ragFusionResponse = await client.rag(
    "Explain the theory of relativity",
    {
        vector_search_settings: {
            search_strategy: "rag_fusion",
            search_limit: 20
        }
    }
);
```

For a comprehensive guide on implementing and optimizing advanced RAG techniques in R2R, including HyDE and RAG-Fusion, please refer to our [Advanced RAG Cookbook](/cookbooks/advanced-rag).


### Customizing RAG

Putting everything together for highly custom RAG functionality:

```javascript
const customRagResponse = await client.rag({
  query: "Who was Aristotle?",
  use_vector_search: true,
  use_hybrid_search: true,
  use_kg_search: true,
  kg_generation_config: {},
  rag_generation_config: {
    model: "anthropic/claude-3-haiku-20240307",
    temperature: 0.7,
    stream: true
  }
});
```

## Agents

### Multi-turn agentic RAG
The R2R application includes agents which come equipped with a search tool, enabling them to perform RAG. Using the R2R Agent for multi-turn conversations:

```javascript
const messages = [
  { role: "user", content: "What was Aristotle's main contribution to philosophy?" },
  { role: "assistant", content: "Aristotle made numerous significant contributions to philosophy, but one of his main contributions was in the field of logic and reasoning. He developed a system of formal logic, which is considered the first comprehensive system of its kind in Western philosophy. This system, often referred to as Aristotelian logic or term logic, provided a framework for deductive reasoning and laid the groundwork for scientific thinking." },
  { role: "user", content: "Can you elaborate on how this influenced later thinkers?" }
];

const agentResponse = await client.agent({
  messages,
  use_vector_search: true,
  use_hybrid_search: true
});
```

Note that any of the customization seen in AI powered search and RAG documentation above can be applied here.

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="object">
    The agent endpoint will return the entire conversation as a response, including internal tool calls.
        ```javascript
        {
          results: [
            {
              role: 'system',
              content: '## You are a helpful agent that can search for information.\n\nWhen asked a question, perform a search to find relevant information and provide a response.\n\nThe response should contain line-item attributions to relevent search results, and be as informative if possible.\nIf no relevant results are found, then state that no results were found.\nIf no obvious question is present, then do not carry out a search, and instead ask for clarification.\n',
              name: null,
              function_call: null,
              tool_calls: null
            },
            {
              role: 'user',
              content: "What was Aristotle's main contribution to philosophy?",
              name: null,
              function_call: null,
              tool_calls: null
            },
            {
              role: 'assistant',
              content: "Aristotle made numerous significant contributions to philosophy, but one of his main contributions was in the field of logic and reasoning. He developed a system of formal logic, which is considered the first comprehensive system of its kind in Western philosophy. This system, often referred to as Aristotelian logic or term logic, provided a framework for deductive reasoning and laid the groundwork for scientific thinking.",
              name: null,
              function_call: null,
              tool_calls: null
            },
            {
              role: 'user',
              content: 'Can you elaborate on how this influenced later thinkers?',
              name: null,
              function_call: null,
              tool_calls: null
            },
            {
              role: 'assistant',
              content: null,
              name: null,
              function_call: {
                name: 'search',
                arguments: '{"query":"Aristotle\'s influence on later thinkers in philosophy"}'
              },
              tool_calls: null
            },
            {
              role: 'function',
              content: '1. ormation: List of writers influenced by Aristotle More than 2300 years after his death, Aristotle remains one of the most influential people who ever lived.[142][143][144] He contributed to almost every field of human knowledge then in existence, and he was the founder of many new fields. According to the philosopher Bryan Magee, "it is doubtful whether any human being has ever known as much as he did".[145]\n2. subject of contemporary philosophical discussion. Aristotle\'s views profoundly shaped medieval scholarship. The influence of his physical science extended from late antiquity and the Early Middle Ages into the Renaissance, and was not replaced systematically until the Enlightenment and theories such as classical mechanics were developed. He influenced Judeo-Islamic philosophies during the Middle Ages, as well as Christian theology, especially the Neoplatonism of the Early Church and the scholastic tradition\n3. the scholastic tradition of the Catholic Church. Aristotle was revered among medieval Muslim scholars as "The First Teacher", and among medieval Christians like Thomas Aquinas as simply "The Philosopher", while the poet Dante called him "the master of those who know". His works contain the earliest known formal study of logic, and were studied by medieval scholars such as Peter Abelard and Jean Buridan. Aristotle\'s influence on logic continued well into the 19th century. In addition, his ethics, although\n4. hilosophy\nFurther information: Peripatetic school The immediate influence of Aristotle\'s work was felt as the Lyceum grew into the Peripatetic school. Aristotle\'s students included Aristoxenus, Dicaearchus, Demetrius of Phalerum, Eudemos of Rhodes, Harpalus, Hephaestion, Mnason of Phocis, Nicomachus, and Theophrastus. Aristotle\'s influence over Alexander the Great is seen in the latter\'s bringing with him on his expedition a host of zoologists, botanists, and researchers. He had also learned a great deal\n5. scholastic philosophers. Alkindus greatly admired Aristotle\'s philosophy,[168] and Averroes spoke of Aristotle as the "exemplar" for all future philosophers.[169] Medieval Muslim scholars regularly described Aristotle as the "First Teacher".[167] The title was later used by Western philosophers (as in the famous poem of Dante) who were influenced by the tradition of Islamic philosophy.[170]\n\nMedieval Europe\nFurther information: Aristotelianism and Syllogism § Medieval\n6. those by James of Venice and William of Moerbeke. After the Scholastic Thomas Aquinas wrote his Summa Theologica, working from Moerbeke\'s translations and calling Aristotle "The Philosopher",[172] the demand for Aristotle\'s writings grew, and the Greek manuscripts returned to the West, stimulating a revival of Aristotelianism in Europe that continued into the Renaissance.[173] These thinkers blended Aristotelian philosophy with Christianity, bringing the thought of Ancient Greece into the Middle Ages.',
              name: 'search',
              function_call: null,
              tool_calls: null
            },
            {
              role: 'assistant',
              content: "Aristotle's contributions to philosophy, particularly his development of formal logic, had a profound influence on later thinkers across various cultures and eras. Here are some key ways in which his work influenced subsequent intellectual traditions:\n\n1. Medieval Islamic Philosophy: Aristotle was highly revered among medieval Muslim scholars, who referred to him as \"The First Teacher.\" His works were extensively studied and commented upon by philosophers such as Averroes and Avicenna. These scholars played a crucial role in preserving and expanding upon Aristotle's ideas, which later influenced Western thought [5].\n\n2. Christian Scholasticism: In medieval Europe, Aristotle's works were integrated into Christian theology, particularly through the efforts of Thomas Aquinas, who referred to Aristotle as \"The Philosopher.\" Aquinas' synthesis of Aristotelian philosophy with Christian doctrine became a cornerstone of Scholasticism, a dominant intellectual tradition in medieval Europe [3][6].\n\n3. Renaissance and Enlightenment: Aristotle's influence persisted into the Renaissance, where his works were revived and studied extensively. This period saw a renewed interest in classical texts, and Aristotle's ideas continued to shape scientific and philosophical inquiry until the Enlightenment, when new scientific paradigms began to emerge [2][6].\n\n4. Development of Logic: Aristotle's system of formal logic remained the standard for centuries and was studied by medieval scholars such as Peter Abelard and Jean Buridan. His influence on logic extended well into the 19th century, shaping the development of this field [3].\n\n5. Peripatetic School: Aristotle's immediate influence was also felt through the Peripatetic school, which he founded. His students, including Theophrastus, carried on his work and further developed his ideas, ensuring that his intellectual legacy continued [4].\n\nOverall, Aristotle's contributions laid the groundwork for many fields of study and influenced a wide range of thinkers, making him one of the most significant figures in the history of Western philosophy and beyond.",
              name: null,
              function_call: null,
              tool_calls: null
            }
          ]
        }
        ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

<ParamField path="messages" type="Array<Message>" required>
  The array of messages to pass to the RAG agent.
</ParamField>

<ParamField path="use_vector_search" type="boolean" default={true}>
  Whether to use vector search.
</ParamField>

<ParamField path="filters" type="object">
  Optional filters for the search.
</ParamField>

<ParamField path="search_limit" type="number" default={10}>
  The maximum number of search results to return.
</ParamField>

<ParamField path="use_hybrid_search" type="boolean" default={false}>
  Whether to perform a hybrid search (combining vector and keyword search).
</ParamField>

<ParamField path="use_kg_search" type="boolean" default={false}>
  Whether to use knowledge graph search.
</ParamField>

<ParamField path="generation_config" type="object">
  Optional configuration for knowledge graph search generation.
</ParamField>

<ParamField path="rag_generation_config" type="GenerationConfig">
  Optional configuration for RAG generation, including model selection and parameters.
</ParamField>

<ParamField path="task_prompt_override" type="string">
  Optional custom prompt to override the default task prompt.
</ParamField>

<ParamField path="include_title_if_available" type="boolean" default={true}>
  Whether to include document titles in the context if available.
</ParamField>

### Multi-turn agentic RAG with streaming

The response from the RAG agent may be streamed directly back:

```javascript
const messages = [
  { role: "user", content: "What was Aristotle's main contribution to philosophy?" },
  { role: "assistant", content: "Aristotle made numerous significant contributions to philosophy, but one of his main contributions was in the field of logic and reasoning. He developed a system of formal logic, which is considered the first comprehensive system of its kind in Western philosophy. This system, often referred to as Aristotelian logic or term logic, provided a framework for deductive reasoning and laid the groundwork for scientific thinking." },
  { role: "user", content: "Can you elaborate on how this influenced later thinkers?" }
];

const agentResponse = await client.agent({
  messages,
  use_vector_search: true,
  use_hybrid_search: true,
  rag_generation_config: { stream: true }
});

if (agentResponse instanceof ReadableStream) {
  const reader = agentResponse.getReader();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    console.log(new TextDecoder().decode(value));
  }
}
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="ReadableStream">
    The agent endpoint will stream back its response, including internal tool calls.
        ```javascript
        <function_call><name>search</name><arguments>{"query":"Aristotle's influence on later thinkers in philosophy"}</arguments><results>"{"id":"b234931e-0cfb-5644-8f23-560a3097f5fe","score":1.0,"metadata":{"text":"ormation: List of writers influenced by Aristotle More than 2300 years after his death, Aristotle remains one of the most influential people who ever lived.[142][143][144] He contributed to almost every field of human knowledge then in existence, and he was the founder of many new fields. According to the philosopher Bryan Magee, \"it is doubtful whether any human being has ever known as much as he did\".[145]","title":"aristotle.txt","user_id":"2acb499e-8428-543b-bd85-0d9098718220","document_id":"9fbe403b-c11c-5aae-8ade-ef22980c3ad1","extraction_id":"69431c4a-30cf-504f-8fab-7dcfc7580c63","associatedQuery":"Aristotle's influence on later thinkers in philosophy"}}","{"id":"1827ac2c-2a06-5bc2-ad29-aa14b4d99540","score":1.0,"metadata":{"text":"subject of contemporary philosophical discussion. Aristotle's views profoundly shaped medieval scholarship. The influence of his physical science extended from late antiquity and the Early Middle Ages into the Renaissance, and was not replaced systematically until the Enlightenment and theories such as classical mechanics were developed. He influenced Judeo-Islamic philosophies during the Middle Ages, as well as Christian theology, especially the Neoplatonism of the Early Church and the scholastic tradition","title":"aristotle.txt","user_id":"2acb499e-8428-543b-bd85-0d9098718220","document_id":"9fbe403b-c11c-5aae-8ade-ef22980c3ad1","extraction_id":"69431c4a-30cf-504f-8fab-7dcfc7580c63","associatedQuery":"Aristotle's influence on later thinkers in philosophy"}}","{"id":"94718936-ea92-5e29-a5ee-d4a6bc037384","score":1.0,"metadata":{"text":"the scholastic tradition of the Catholic Church. Aristotle was revered among medieval Muslim scholars as \"The First Teacher\", and among medieval Christians like Thomas Aquinas as simply \"The Philosopher\", while the poet Dante called him \"the master of those who know\". His works contain the earliest known formal study of logic, and were studied by medieval scholars such as Peter Abelard and Jean Buridan. Aristotle's influence on logic continued well into the 19th century. In addition, his ethics, although","title":"aristotle.txt","user_id":"2acb499e-8428-543b-bd85-0d9098718220","document_id":"9fbe403b-c11c-5aae-8ade-ef22980c3ad1","extraction_id":"69431c4a-30cf-504f-8fab-7dcfc7580c63","associatedQuery":"Aristotle's influence on later thinkers in philosophy"}}","{"id":"16483f14-f8a2-5c5c-8fcd-1bcbbd6603e4","score":1.0,"metadata":{"text":"hilosophy\nFurther information: Peripatetic school The immediate influence of Aristotle's work was felt as the Lyceum grew into the Peripatetic school. Aristotle's students included Aristoxenus, Dicaearchus, Demetrius of Phalerum, Eudemos of Rhodes, Harpalus, Hephaestion, Mnason of Phocis, Nicomachus, and Theophrastus. Aristotle's influence over Alexander the Great is seen in the latter's bringing with him on his expedition a host of zoologists, botanists, and researchers. He had also learned a great deal","title":"aristotle.txt","user_id":"2acb499e-8428-543b-bd85-0d9098718220","document_id":"9fbe403b-c11c-5aae-8ade-ef22980c3ad1","extraction_id":"69431c4a-30cf-504f-8fab-7dcfc7580c63","associatedQuery":"Aristotle's influence on later thinkers in philosophy"}}","{"id":"26eb20ee-a203-5ad5-beaa-511cc526aa6e","score":1.0,"metadata":{"text":"scholastic philosophers. Alkindus greatly admired Aristotle's philosophy,[168] and Averroes spoke of Aristotle as the \"exemplar\" for all future philosophers.[169] Medieval Muslim scholars regularly described Aristotle as the \"First Teacher\".[167] The title was later used by Western philosophers (as in the famous poem of Dante) who were influenced by the tradition of Islamic philosophy.[170]\n\nMedieval Europe\nFurther information: Aristotelianism and Syllogism § Medieval","title":"aristotle.txt","user_id":"2acb499e-8428-543b-bd85-0d9098718220","document_id":"9fbe403b-c11c-5aae-8ade-ef22980c3ad1","extraction_id":"69431c4a-30cf-504f-8fab-7dcfc7580c63","associatedQuery":"Aristotle's influence on later thinkers in philosophy"}}","{"id":"a08fd1b4-4e6f-5487-9af6-df2f6cfe1048","score":1.0,"metadata":{"text":"those by James of Venice and William of Moerbeke. After the Scholastic Thomas Aquinas wrote his Summa Theologica, working from Moerbeke's translations and calling Aristotle \"The Philosopher\",[172] the demand for Aristotle's writings grew, and the Greek manuscripts returned to the West, stimulating a revival of Aristotelianism in Europe that continued into the Renaissance.[173] These thinkers blended Aristotelian philosophy with Christianity, bringing the thought of Ancient Greece into the Middle Ages.","title":"aristotle.txt","user_id":"2acb499e-8428-543b-bd85-0d9098718220","document_id":"9fbe403b-c11c-5aae-8ade-ef22980c3ad1","extraction_id":"69431c4a-30cf-504f-8fab-7dcfc7580c63","associatedQuery":"Aristotle's influence on later thinkers in philosophy"}}"</results></function_call><completion>Aristotle's contributions to philosophy, particularly his development of formal logic, had a profound influence on later thinkers across various cultures and eras. Here are some key ways in which his work influenced subsequent intellectual traditions:

1. Medieval Islamic Philosophy: Aristotle was highly revered among medieval Muslim scholars, who referred to him as "The First Teacher." His works were extensively translated into Arabic and studied by philosophers such as Averroes and Avicenna. These scholars not only preserved Aristotle's works but also expanded upon them, influencing both Islamic and Western thought.

2. Christian Scholasticism: In medieval Europe, Aristotle's works were integrated into Christian theology, particularly through the efforts of Thomas Aquinas, who referred to Aristotle as "The Philosopher." Aquinas's synthesis of Aristotelian philosophy with Christian doctrine became a cornerstone of Scholasticism, a dominant intellectual tradition in medieval Europe.

3. Renaissance and Enlightenment: Aristotle's influence persisted into the Renaissance and Enlightenment periods. His works on logic, ethics, and natural sciences were foundational texts for scholars during these eras. The revival of Aristotelianism during the Renaissance helped bridge the gap between ancient Greek philosophy and modern scientific thought.

4. Development of Logic: Aristotle's system of formal logic remained the standard for centuries and was studied by medieval scholars such as Peter Abelard and Jean Buridan. His influence on logic extended well into the 19th century, shaping the development of this field.

5. Peripatetic School: Aristotle's immediate influence was felt through the Peripatetic school, which he founded. His students, including Theophrastus, continued to develop and disseminate his ideas, ensuring that his philosophical legacy endured.

Overall, Aristotle's contributions to logic, ethics, natural sciences, and metaphysics created a foundation upon which much of Western intellectual tradition was built. His work influenced a wide range of fields and thinkers, making him one of the most pivotal figures in the history of philosophy.</completion>
        ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

---

# Deploying R2R on Amazon Web Services (AWS)

Amazon Web Services (AWS) provides a robust and scalable platform for deploying R2R (RAG to Riches). This guide will walk you through the process of setting up R2R on an Amazon EC2 instance, making it accessible both locally and publicly.

## Overview

Deploying R2R on AWS involves the following main steps:

1. Creating an Amazon EC2 instance
2. Installing necessary dependencies
3. Setting up R2R
4. Configuring port forwarding for local access
5. Exposing ports for public access (optional)

This guide assumes you have an AWS account and the necessary permissions to create and manage EC2 instances.

## Creating an Amazon EC2 Instance

1. Log in to the [AWS Management Console](https://aws.amazon.com/console/).
2. Navigate to EC2 under "Compute" services.
3. Click "Launch Instance".
4. Choose an Amazon Machine Image (AMI):
   - Select "Ubuntu Server 22.04 LTS (HVM), SSD Volume Type"
5. Choose an Instance Type:
   - For a small-mid sized organization (< 5000 users), select t3.xlarge (4 vCPU, 16 GiB Memory) or higher
6. Configure Instance Details:
   - Leave default settings or adjust as needed
7. Add Storage:
   - Set the root volume to at least 500 GiB
8. Add Tags (optional):
   - Add any tags for easier resource management
9. Configure Security Group:
   - Create a new security group
   - Add rules to allow inbound traffic on ports 22 (SSH) and 7272 (R2R API)
10. Review and Launch:
    - Review your settings and click "Launch"
    - Choose or create a key pair for SSH access

## Installing Dependencies

SSH into your newly created EC2 instance:

```bash
ssh -i /path/to/your-key.pem ubuntu@your-instance-public-dns
```

Now, run the following commands to install the necessary R2R dependencies:

```bash
# Update package list and install Python and pip
sudo apt update
sudo apt install python3-pip

# Install R2R
pip install r2r

# Add R2R to PATH
echo 'export PATH=$PATH:$HOME/.local/bin' >> ~/.bashrc
source ~/.bashrc

# Install Docker
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add your user to the Docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify Docker installation
docker run hello-world
```

## Setting up R2R

1. Serve your R2R backend:

```bash
# Set required remote providers
export OPENAI_API_KEY=sk-...

# Optional - pass in a custom configuration here
r2r serve --docker --full
```

2. Double check the health of the system:

```bash
r2r health
```

3. Test ingesting and searching a sample document from a remote environment:

```bash
# From your local machine
r2r --base-url=http://<your-instance-public-ip>:7272 ingest-sample-file
sleep 10
r2r --base-url=http://<your-instance-public-ip>:7272 search --query='Who was aristotle?'
```

Replace `<your-instance-public-ip>` with your EC2 instance's public IP address.

## Configuring Port Forwarding for Local Access

To access R2R from your local machine, use SSH port forwarding:

```bash
ssh -i /path/to/your-key.pem -L 7273:localhost:7273 -L 7274:localhost:7274 ubuntu@your-instance-public-dns
```

## Exposing Ports for Public Access (Optional)

To make R2R publicly accessible:

1. In the AWS Management Console, go to EC2 > Security Groups.
2. Select the security group associated with your EC2 instance.
3. Click "Edit inbound rules".
4. Add a new rule:
   - Type: Custom TCP
   - Port range: 7272
   - Source: Anywhere (0.0.0.0/0)
   - Description: Allow R2R API
5. Click "Save rules".

6. Ensure R2R is configured to listen on all interfaces (0.0.0.0).

After starting your R2R application, users can access it at:

```
http://<your-instance-public-ip>:7272
```

## Security Considerations

- Use HTTPS (port 443) with a valid SSL certificate for production.
- Restrict source IP addresses in the security group rule if possible.
- Regularly update and patch your system and applications.
- Use AWS VPC for network isolation.
- Enable and configure AWS CloudTrail for auditing.
- Use AWS IAM roles for secure access management.
- Consider using AWS Certificate Manager for SSL/TLS certificates.
- Monitor incoming traffic using AWS CloudWatch.
- Remove or disable the security group rule when not needed for testing.

## Conclusion

You have now successfully deployed R2R on Amazon Web Services. The application should be accessible locally through SSH tunneling and optionally publicly through direct access to the EC2 instance. Remember to configure authentication and implement proper security measures before exposing your R2R instance to the public internet.

For more information on configuring and using R2R, refer to the [configuration documentation](/documentation/configuration/introduction).

---

## System Management

### Retrieve Analytics

Retrieve analytics data using the `analytics` command:

```bash
r2r analytics --filters '{"key1": "value1"}' --analysis-types '{"type1": true}'
```

<AccordionGroup>
  <Accordion title="Arguments">
    <ParamField path="--filters" type="str">
      Optional filters for analytics as a JSON string.
    </ParamField>

    <ParamField path="--analysis-types" type="str">
      Optional analysis types as a JSON string.
    </ParamField>
  </Accordion>
</AccordionGroup>

### Retrieve Application Settings

Retrieve application settings using the `app-settings` command:

```bash
r2r app-settings
```

This command has no additional arguments.

### Get User Overview

Get an overview of users using the `users-overview` command:

```bash
r2r users-overview --user-ids user1 user2 --offset 0 --limit 10
```

<AccordionGroup>
  <Accordion title="Arguments">
    <ParamField path="--user-ids" type="list[str]">
      Optional user IDs to overview.
    </ParamField>

    <ParamField path="--offset" type="int">
      The offset to start from. Defaults to 0.
    </ParamField>

    <ParamField path="--limit" type="int">
      The maximum number of nodes to return. Defaults to 100.
    </ParamField>
  </Accordion>
</AccordionGroup>

### Delete Documents

Delete documents based on filters using the `delete` command:

```bash
r2r delete -f key1:eq:value1 -f key2:gt:value2
```

<AccordionGroup>
  <Accordion title="Arguments">
    <ParamField path="--filter, -f" type="list[str]" required>
      Filters for deletion in the format key:operator:value.
    </ParamField>
  </Accordion>
</AccordionGroup>

### Get Document Overview

Get an overview of documents using the `documents-overview` command:

```bash
r2r documents-overview --document-ids doc1 doc2 --offset 0 --limit 10
```

<AccordionGroup>
  <Accordion title="Arguments">
    <ParamField path="--document-ids" type="list[str]">
      Optional document IDs to overview.
    </ParamField>

    <ParamField path="--offset" type="int">
      The offset to start from. Defaults to 0.
    </ParamField>

    <ParamField path="--limit" type="int">
      The maximum number of nodes to return. Defaults to 100.
    </ParamField>
  </Accordion>
</AccordionGroup>

### Get Document Chunks

Get chunks of a specific document using the `document-chunks` command:

```bash
r2r document-chunks --document-id doc1 --offset 0 --limit 10
```

<AccordionGroup>
  <Accordion title="Arguments">
    <ParamField path="--document-id" type="str" required>
      The ID of the document to retrieve chunks for.
    </ParamField>

    <ParamField path="--offset" type="int">
      The offset to start from. Defaults to 0.
    </ParamField>

    <ParamField path="--limit" type="int">
      The maximum number of nodes to return. Defaults to 100.
    </ParamField>

    <ParamField path="include_vectors" type="Optional[bool]">
      An optional value to return the vectors associated with each chunk, defaults to `False`.
    </ParamField>

  </Accordion>
</AccordionGroup>

---


# Retrieval Operations

The R2R CLI provides two main retrieval commands: `search` and `rag` (Retrieval-Augmented Generation). These commands allow you to query your document collection and generate AI-powered responses based on the retrieved content.

## Search Command

The `search` command performs document retrieval using vector search and/or knowledge graph search capabilities.

```bash
r2r search --query "Your search query"
```

### Vector Search Options

- `--use-vector-search`: Enable vector search (default: true)
- `--filters`: Apply JSON filters to the search results
  ```bash
  r2r search --filters '{"document_id":{"$in":["doc-id-1", "doc-id-2"]}}'
  ```
- `--search-limit`: Maximum number of search results to return
- `--use-hybrid-search`: Enable hybrid search combining vector and keyword search
- `--selected-collection-ids`: Specify collection IDs to search within as JSON array
- `--search-strategy`: Choose between "vanilla" search or advanced methods like query fusion or HyDE

### Knowledge Graph Search Options

- `--use-kg-search`: Enable knowledge graph search
- `--kg-search-type`: Choose between "local" or "global" search
- `--kg-search-level`: Specify the level for global KG search
- `--entity-types`: Filter by entity types (as JSON)
- `--relationships`: Filter by relationship types (as JSON)
- `--max-community-description-length`: Set maximum length for community descriptions
- `--local-search-limits`: Set limits for local search (as JSON)

## RAG Command

The `rag` command combines search capabilities with AI generation to provide contextual responses based on your document collection.

```bash
r2r rag --query "Your question"
```

### Generation Options

- `--stream`: Stream the response in real-time
- `--rag-model`: Specify the model to use for generation

### Vector Search Settings

- `--use-vector-search`: Enable vector search (default: true)
- `--filters`: Apply JSON filters to search results
- `--search-limit`: Maximum number of search results (default: 10)
- `--use-hybrid-search`: Enable hybrid search
- `--selected-collection-ids`: Specify collection IDs to search within
- `--search-strategy`: Choose search method (default: "vanilla")

### Knowledge Graph Settings

- `--use-kg-search`: Enable knowledge graph search
- `--kg-search-type`: Set to "local" or "global" (default: "local")
- `--kg-search-level`: Specify cluster level for Global KG search
- `--kg-search-model`: Choose the model for KG agent
- `--entity-types`: Filter by entity types (as JSON)
- `--relationships`: Filter by relationship types (as JSON)
- `--max-community-description-length`: Set maximum community description length
- `--local-search-limits`: Set limits for local search (as JSON)

## Examples

### Basic Search

```bash
# Simple vector search
r2r search --query "What is quantum computing?"

# Search with filters
r2r search --query "quantum computing" --filters '{"category": "physics"}'
```

### Advanced Search

```bash
# Hybrid search with collection filtering
r2r search --query "quantum computing" \
  --use-hybrid-search \
  --selected-collection-ids '["physics-collection", "computing-collection"]'

# Knowledge graph search
r2r search --query "quantum computing relationships" \
  --use-kg-search \
  --kg-search-type "local" \
  --entity-types '["Concept", "Technology"]'
```

### Basic RAG

```bash
# Simple RAG query
r2r rag --query "Explain quantum computing"

# Streaming RAG response
r2r rag --query "Explain quantum computing" --stream
```

### Advanced RAG

```bash
# RAG with custom model and hybrid search
r2r rag --query "Explain quantum computing" \
  --rag-model "gpt-4" \
  --use-hybrid-search \
  --search-limit 20

# RAG with knowledge graph integration
r2r rag --query "How do quantum computers relate to cryptography?" \
  --use-kg-search \
  --kg-search-type "global" \
  --relationships '["ENABLES", "IMPACTS"]' \
  --stream
```

## Tips for Effective Retrieval

1. **Refine Your Queries**: Be specific and clear in your search queries to get more relevant results.

2. **Use Filters**: Narrow down results using filters when you know specific document characteristics.

3. **Combine Search Types**: Use hybrid search and knowledge graph capabilities together for more comprehensive results.

4. **Adjust Search Limits**: Modify the search limit based on your needs - higher limits for broad topics, lower limits for specific queries.

5. **Stream Long Responses**: Use the `--stream` option with RAG for better user experience with longer generations.

---

<Warning>This installation guide is for R2R Core. For solo developers or teams prototyping, we highly recommend starting with <a href="/documentation/installation/light/local-system">R2R Light</a>.</Warning>


This guide will walk you through installing and running R2R using Docker, which is the quickest and easiest way to get started.

## Prerequisites

- Docker installed on your system. If you haven't installed Docker yet, please refer to the [official Docker installation guide](https://docs.docker.com/engine/install/).

## Install the R2R CLI & Python SDK

First, install the R2R CLI and Python SDK:

```bash
pip install r2r
```


<Info>We are actively developing a distinct CLI binary for R2R for easier installation. Please reach out if you have any specific needs or feature requests.</Info>

## Start R2R with Docker

<Note>The full R2R installation does not use the default [`r2r.toml`](https://github.com/SciPhi-AI/R2R/blob/main/py/r2r.toml), instead it provides overrides through a pre-built custom configuration, [`full.toml`](https://github.com/SciPhi-AI/R2R/blob/main/py/core/configs/full.toml).</Note>

<AccordionGroup>
  <Accordion title="Cloud LLM RAG" icon="cloud" defaultOpen={true}>
  To start R2R with OpenAI as the default LLM inference and embedding provider:
      ```bash
      # Set cloud LLM settings
      export OPENAI_API_KEY=sk-...

      r2r serve --docker --full
      ```
  [Refer here](/documentation/configuration/llm) for more information on how to configure various LLM providers.
  </Accordion>
  <Accordion title="Local LLMs" icon="house" defaultOpen={false}>
      To start R2R with your local computer as the default LLM inference provider:
      ```bash
      r2r serve --docker --full --config-name=full_local_llm
      ```
    Then, in a separate terminal you will need to run Ollama to provide completions:
    ```bash
      ollama pull llama3.1
      ollama pull mxbai-embed-large
      ollama serve
    ```
    The code above assumes that Ollama has already been installed. If you have not yet done so, then refer to the official Ollama webpage [for installation instructions](https://ollama.com/download). For more information on local installation, [refer here](/documentation/local-rag).
  </Accordion>
  <Accordion title="Custom Configuration" icon="gear" defaultOpen={false}>
    R2R offers flexibility in selecting and configuring LLMs, allowing you to optimize your RAG pipeline for various use cases. Execute the command below run deploy R2R with your own custom configuration:
      ```bash
      r2r serve --config-path=/abs/path/to/my_r2r.toml
      ```

    Learn in detail how to [configure your deployment here](/documentation/configuration).
  </Accordion>
</AccordionGroup>


The above command will automatically pull the necessary Docker images and start all the required containers, including `R2R`, `Hatchet`, and `Postgres+pgvector`.  The required additional services come bundled into the full R2R Docker Compose by default.

The end result is a live server at http://localhost:7272 serving the [R2R API](/api-reference/introduction).

In addition to launching a RESTful API, the R2R Docker also launches a applications at `localhost:7273` and `localhost:7274`, which you can [read more about here](/cookbooks/application).

### Stopping R2R

Safely stop your system by running `r2r docker-down` to avoid potential shutdown complications.

## Next Steps

After successfully installing R2R:

1. **Verify Installation**: Ensure all components are running correctly by accessing the R2R API at http://localhost:7272/v2/health.

2. **Quick Start**: Follow our [R2R Quickstart Guide](/documentation/quickstart) to set up your first RAG application.

3. **In-Depth Tutorial**: For a more comprehensive understanding, work through our [R2R Walkthrough](/cookbooks/walkthrough).

4. **Customize Your Setup**: Configure R2R components with the [Configuration Guide](/documentation/configuration).

If you encounter any issues during installation or setup, please use our [Discord community](https://discord.gg/p6KqD2kjtB) or [GitHub repository](https://github.com/SciPhi-AI/R2R) to seek assistance.

---


## Document Ingestion and Management

### Ingest Files

Ingest files or directories into your R2R system using the `ingest-files` command:

```bash
r2r ingest-files path/to/file1.txt path/to/file2.txt \
  --document-ids 9fbe403b-c11c-5aae-8ade-ef22980c3ad1 \
  --metadatas '{"key1": "value1"}'
```

<AccordionGroup>
  <Accordion title="Arguments">
    <ParamField path="file_paths" type="list[str]" required>
      The paths to the files to ingest.
    </ParamField>

    <ParamField path="--document-ids" type="list[str]">
      Optional document IDs to assign to the ingested files. If not provided, new document IDs will be generated.
    </ParamField>

    <ParamField path="--metadatas" type="str">
      Optional metadata to attach to the ingested files, provided as a JSON string. If ingesting multiple files, the metadata will be applied to all files.
    </ParamField>
  </Accordion>
</AccordionGroup>

### Retry Failed Ingestions

Retry ingestion for documents that previously failed using the `retry-ingest-files` command:

```bash
r2r retry-ingest-files 9fbe403b-c11c-5aae-8ade-ef22980c3ad1
```

<AccordionGroup>
  <Accordion title="Arguments">
    <ParamField path="document_ids" type="list[str]" required>
      The IDs of the documents to retry ingestion for.
    </ParamField>
  </Accordion>
</AccordionGroup>

### Update Files

Update existing documents using the `update-files` command:

```bash
r2r update-files path/to/file1_v2.txt \
  --document-ids 9fbe403b-c11c-5aae-8ade-ef22980c3ad1 \
  --metadatas '{"key1": "value2"}'
```

<AccordionGroup>
  <Accordion title="Arguments">
    <ParamField path="file_paths" type="list[str]" required>
      The paths to the updated files.
    </ParamField>

    <ParamField path="--document-ids" type="str" required>
      The IDs of the documents to update, provided as a comma-separated string.
    </ParamField>

    <ParamField path="--metadatas" type="str">
      Optional updated metadata to attach to the documents, provided as a JSON string. If updating multiple files, the metadata will be applied to all files.
    </ParamField>
  </Accordion>
</AccordionGroup>

## Vector Index Management
## Vector Index Management

### Create Vector Index

Create a new vector index for similarity search using the `create-vector-index` command:

```bash
r2r create-vector-index \
  --table-name vectors \
  --index-method hnsw \
  --index-measure cosine_distance \
  --index-arguments '{"m": 16, "ef_construction": 64}'
```

<AccordionGroup>
  <Accordion title="Arguments">
    <ParamField path="--table-name" type="str">
      Table to create index on. Options: vectors, entities_document, entities_collection, communities. Default: vectors
    </ParamField>

    <ParamField path="--index-method" type="str">
      Indexing method to use. Options: hnsw, ivfflat, auto. Default: hnsw
    </ParamField>

    <ParamField path="--index-measure" type="str">
      Distance measure for vector comparisons. Options: cosine_distance, l2_distance, max_inner_product. Default: cosine_distance
    </ParamField>

    <ParamField path="--index-arguments" type="str">
      Configuration parameters as JSON string. For HNSW: `{"m": int, "ef_construction": int}`. For IVFFlat: `{"n_lists": int}`
    </ParamField>

    <ParamField path="--index-name" type="str">
      Optional custom name for the index. If not provided, one will be auto-generated
    </ParamField>

    <ParamField path="--no-concurrent" type="flag">
      Disable concurrent index creation. Default: False
    </ParamField>
  </Accordion>
</AccordionGroup>

#### Important Considerations

Vector index creation requires careful planning and consideration of your data and performance requirements. Keep in mind:

**Resource Intensive Process**
- Index creation can be CPU and memory intensive, especially for large datasets
- For HNSW indexes, memory usage scales with both dataset size and `m` parameter
- Consider creating indexes during off-peak hours for production systems

**Performance Tuning**
1. **HNSW Parameters:**
   - `m`: Higher values (16-64) improve search quality but increase memory usage and build time
   - `ef_construction`: Higher values increase build time and quality but have diminishing returns past 100
   - Recommended starting point: `m=16`, `ef_construction=64`

```bash
# Example balanced configuration
r2r create-vector-index \
  --table-name vectors \
  --index-method hnsw \
  --index-measure cosine_distance \
  --index-arguments '{"m": 16, "ef_construction": 64}'
```

**Pre-warming Required**
- **Important:** Newly created indexes require pre-warming to achieve optimal performance
- Initial queries may be slower until the index is loaded into memory
- The first several queries will automatically warm the index
- For production systems, consider implementing explicit pre-warming by running representative queries after index creation
- Without pre-warming, you may not see the expected performance improvements

**Best Practices**
1. Always use concurrent index creation (avoid `--no-concurrent`) in production to prevent blocking other operations
2. Monitor system resources during index creation
3. Test index performance with representative queries before deploying
4. Consider creating indexes on smaller test datasets first to validate parameters
5. Implement index pre-warming strategy before handling production traffic

**Distance Measures**
Choose the appropriate measure based on your use case:
- `cosine_distance`: Best for normalized vectors (most common)
- `l2_distance`: Better for absolute distances
- `max_inner_product`: Optimized for dot product similarity

### List Vector Indices

List existing vector indices using the `list-vector-indices` command:

```bash
r2r list-vector-indices --table-name vectors
```

<AccordionGroup>
  <Accordion title="Arguments">
    <ParamField path="--table-name" type="str">
      Table to list indices from. Options: vectors, entities_document, entities_collection, communities. Default: vectors
    </ParamField>
  </Accordion>
</AccordionGroup>

### Delete Vector Index

Delete a vector index using the `delete-vector-index` command:

```bash
r2r delete-vector-index my-index-name --table-name vectors
```

<AccordionGroup>
  <Accordion title="Arguments">
    <ParamField path="index-name" type="str" required>
      Name of the index to delete
    </ParamField>

    <ParamField path="--table-name" type="str">
      Table containing the index. Options: vectors, entities_document, entities_collection, communities. Default: vectors
    </ParamField>

    <ParamField path="--no-concurrent" type="flag">
      Disable concurrent index deletion. Default: False
    </ParamField>
  </Accordion>
</AccordionGroup>

## Sample File Management

### Ingest Sample Files

Ingest one or more sample files from the R2R GitHub repository:

```bash
# Ingest a single sample file
r2r ingest-sample-file

# Ingest a smaller version of the sample file
r2r ingest-sample-file --v2

# Ingest multiple sample files
r2r ingest-sample-files
```

These commands have no additional arguments. The `--v2` flag for `ingest-sample-file` ingests a smaller version of the sample Aristotle text file.

### Ingest Local Sample Files

Ingest the local sample files in the `core/examples/data_unstructured` directory:

```bash
r2r ingest-sample-files-from-unstructured
```

This command has no additional arguments. It will ingest all files found in the `data_unstructured` directory.

---


## Deployment Management

R2R deployments consist of three main components that need to be managed:
1. The R2R Python package
2. The Docker images
3. The database schema

### Version Management

Check your current R2R version:

```bash
r2r version
```

### Update R2R

Update your R2R installation to the latest version:

```bash
r2r update
```

This command will:
- Upgrade the R2R package to the latest version using pip
- Display the update progress and confirmation
- Show any errors if they occur during the update process

<Note>
When you update R2R, the Docker image used by `r2r serve` will automatically be updated to match the new version. The system will attempt to use a version-specific image (e.g., `ragtoriches/prod:1.2.3`) or fall back to `latest` if the specific version isn't available.
</Note>

### Database Management

R2R uses database migrations to manage schema changes across versions. After updating R2R, you should always check and update your database schema:

### Check Current Migration

View the current migration state of your database:

```bash
r2r db current
```

<AccordionGroup>
  <Accordion title="Arguments">
    <ParamField path="--schema" type="str">
      Schema name to check. Defaults to R2R_PROJECT_NAME environment variable.
    </ParamField>
  </Accordion>
</AccordionGroup>

### Apply Migrations

Upgrade your database to the latest version:

```bash
r2r db upgrade
```

<AccordionGroup>
  <Accordion title="Arguments">
    <ParamField path="--schema" type="str">
      Schema name to upgrade. Defaults to R2R_PROJECT_NAME environment variable.
    </ParamField>
    <ParamField path="--revision" type="str">
      Specific revision to upgrade to. Defaults to "head" (latest version).
    </ParamField>
  </Accordion>
</AccordionGroup>

## Deployment Process

Here's the recommended process for updating an R2R deployment:

1. **Prepare for Update**
   ```bash
   # Check current versions
   r2r version
   r2r db current

   # Generate system report (optional)
   r2r generate-report
   ```

2. **Stop Running Services**
   ```bash
   # Bring down existing deployment
   r2r docker-down
   ```

3. **Update R2R**
   ```bash
   r2r update
   ```

4. **Update Database**
   ```bash
   # Check and apply any new migrations
   r2r db upgrade
   ```

5. **Restart Services**
   ```bash
   # Start the server with your configuration
   r2r serve --docker [additional options]
   ```

<AccordionGroup>
  <Accordion title="Server Configuration Options">
    <ParamField path="--host" type="str">
      Host to run the server on. Default is "0.0.0.0".
    </ParamField>
    <ParamField path="--port" type="int">
      Port to run the server on. Default comes from R2R_PORT or PORT env var, or 7272.
    </ParamField>
    <ParamField path="--docker" type="flag">
      Run using Docker (recommended for production).
    </ParamField>
    <ParamField path="--full" type="flag">
      Run the full R2R compose with Hatchet and Unstructured.
    </ParamField>
    <ParamField path="--project-name" type="str">
      Project name for Docker deployment.
    </ParamField>
    <ParamField path="--image" type="str">
      Specific Docker image to use (optional).
    </ParamField>
    <ParamField path="--exclude-postgres" type="flag">
      Exclude creating a Postgres container.
    </ParamField>
  </Accordion>

  <Accordion title="Environment Variables">
    <ParamField path="R2R_POSTGRES_HOST" type="str">
      PostgreSQL host address. Default is "localhost".
    </ParamField>
    <ParamField path="R2R_POSTGRES_PORT" type="str">
      PostgreSQL port. Default is "5432".
    </ParamField>
    <ParamField path="R2R_POSTGRES_DBNAME" type="str">
      PostgreSQL database name. Default is "postgres".
    </ParamField>
    <ParamField path="R2R_POSTGRES_USER" type="str">
      PostgreSQL username. Default is "postgres".
    </ParamField>
    <ParamField path="R2R_PROJECT_NAME" type="str">
      Project name used for schema. Default is "r2r_default".
    </ParamField>
  </Accordion>
</AccordionGroup>

## Managing Multiple Environments

For different environments (development, staging, production), use different project names and schemas:

```bash
# Development
export R2R_PROJECT_NAME=r2r_dev
r2r serve --docker --project-name r2r-dev

# Staging
export R2R_PROJECT_NAME=r2r_staging
r2r serve --docker --project-name r2r-staging

# Production
export R2R_PROJECT_NAME=r2r_prod
r2r serve --docker --project-name r2r-prod
```

## Vector Index Management

R2R uses vector indices to enable efficient similarity search across documents. For detailed information about managing vector indices, including creation, listing, and deletion, see the [Ingestion documentation](/documentation/cli/ingestion).

Key vector index management commands:
```bash
# Create a new vector index
r2r create-vector-index

# List existing indices
r2r list-vector-indices

# Delete an index
r2r delete-vector-index <index-name>
```


## Troubleshooting

If issues occur during deployment:

1. Generate a system report:
   ```bash
   r2r generate-report
   ```

2. Check container health:
   ```bash
   # Bring down existing deployment
   r2r docker-down

   # Start fresh and watch for health checks
   r2r serve --docker
   ```

3. Review the database state:
   ```bash
   r2r db current
   r2r db history
   ```

4. If needed, roll back database changes:
   ```bash
   r2r db downgrade --revision <previous-working-version>
   ```

---


## Server Management

### Check Server Health

Check the health of the server using the `health` command:

````bash
r2r health
````

This command has no additional arguments.

### Check Server Stats

Check the server stats using the `server-stats` command:

````bash
r2r server-stats
````

This command has no additional arguments.

### Retrieve Logs

Retrieve logs with an optional type filter using the `logs` command:

````bash
r2r logs --offset 0 --limit 10 --run-type-filter ingestion
````

<AccordionGroup>
  <Accordion title="Arguments">
    <ParamField path="--offset" type="int">
      Pagination offset. Default is None.
    </ParamField>

    <ParamField path="--limit" type="int">
      Pagination limit. Defaults to 100.
    </ParamField>

    <ParamField path="--run-type-filter" type="str">
      Filter for log types.
    </ParamField>
  </Accordion>
</AccordionGroup>

### Bring Down Docker Compose

Bring down the Docker Compose setup and attempt to remove the network if necessary using the `docker-down` command:

````bash
r2r docker-down --volumes --remove-orphans --project-name my-project
````

<AccordionGroup>
  <Accordion title="Arguments">
    <ParamField path="--volumes" type="flag">
      Remove named volumes declared in the `volumes` section of the Compose file.
    </ParamField>

    <ParamField path="--remove-orphans" type="flag">
      Remove containers for services not defined in the Compose file.
    </ParamField>

    <ParamField path="--project-name" type="str">
      Which Docker Compose project to bring down.
    </ParamField>
  </Accordion>
</AccordionGroup>

### Generate System Report

Generate a system report including R2R version, Docker info, and OS details using the `generate-report` command:

````bash
r2r generate-report
````

This command has no additional arguments.

### Start R2R Server

Start the R2R server using the `serve` command:

````bash
r2r serve --host localhost --port 8000 --docker --full --project-name my-project --config-name my-config --config-path /path/to/config --build --image my-image --image-env prod --exclude-postgres
````

<AccordionGroup>
  <Accordion title="Arguments">
    <ParamField path="--host" type="str">
      Host to run the server on. Default is None.
    </ParamField>

    <ParamField path="--port" type="int">
      Port to run the server on. Default is None.
    </ParamField>

    <ParamField path="--docker" type="flag">
      Run using Docker.
    </ParamField>

    <ParamField path="--full" type="flag">
      Run the full R2R compose? This includes Hatchet and Unstructured.
    </ParamField>

    <ParamField path="--project-name" type="str">
      Project name for Docker deployment. Default is None.
    </ParamField>

    <ParamField path="--config-name" type="str">
      Name of the R2R configuration to use. Default is None.
    </ParamField>

    <ParamField path="--config-path" type="str">
      Path to a custom R2R configuration file. Default is None.
    </ParamField>

    <ParamField path="--build" type="flag">
      Run in debug mode. Only for development.
    </ParamField>

    <ParamField path="--image" type="str">
      Docker image to use.
    </ParamField>

    <ParamField path="--image-env" type="str">
      Which dev environment to pull the image from? Default is "prod".
    </ParamField>

    <ParamField path="--exclude-postgres" type="flag">
      Excludes creating a Postgres container in the Docker setup.
    </ParamField>
  </Accordion>
</AccordionGroup>

### Update R2R

Update the R2R package to the latest version using the `update` command:

````bash
r2r update
````

This command has no additional arguments.

### Print R2R Version

Print the version of R2R using the `version` command:

````bash
r2r version
````

This command has no additional arguments.

---

# Deploying R2R on Google Cloud Platform

Google Cloud Platform (GCP) offers a robust and scalable environment for deploying R2R (RAG to Riches). This guide will walk you through the process of setting up R2R on a Google Compute Engine instance, making it accessible both locally and publicly.

## Overview

Deploying R2R on GCP involves the following main steps:

1. Creating a Google Compute Engine instance
2. Installing necessary dependencies
3. Setting up R2R
4. Configuring port forwarding for local access
5. Exposing ports for public access (optional)

This guide assumes you have a Google Cloud account and the necessary permissions to create and manage Compute Engine instances.

## Creating a Google Compute Engine Instance

1. Log in to the [Google Cloud Console](https://console.cloud.google.com/).
2. Navigate to "Compute Engine" > "VM instances".
3. Click "Create Instance".
4. Choose the following settings:
   - Name: Choose a name for your instance
   - Region and Zone: Select based on your location/preferences
   - Machine Configuration:
     - Series: N1
     - Machine type: n1-standard-4 (4 vCPU, 15 GB memory) or higher
   - Boot disk:
     - Operating System: Ubuntu
     - Version: Ubuntu 22.04 LTS
     - Size: 500 GB
   - Firewall: Allow HTTP and HTTPS traffic
5. Click "Create" to launch the instance.

## Installing Dependencies

SSH into your newly created instance using the Google Cloud Console or gcloud command:

```bash
gcloud compute ssh --zone "your-zone" "your-instance-name"
```

Now, run the following commands to install the necessary R2R dependencies:

```bash
# Update package list and install Python and pip
sudo apt update
sudo apt install python3-pip

# Install R2R
pip install r2r

# Add R2R to PATH
echo 'export PATH=$PATH:$HOME/.local/bin' >> ~/.bashrc
source ~/.bashrc

# Install Docker
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add your user to the Docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify Docker installation
docker run hello-world
```

## Setting up R2R

1. Serve your R2R backend:

```bash
# Set required remote providers
export OPENAI_API_KEY=sk-...

# Optional - pass in a custom configuration here
r2r serve  --docker --full
```

2. Double check the health of the system:

```bash
r2r health
```

3. Test ingesting and searching a sample document from a remote environment:

```bash
# From your local machine
r2r --base-url=http://<your-instance-external-ip>:7272 ingest-sample-file
sleep 10
r2r --base-url=http://<your-instance-external-ip>:7272 search --query='Who was aristotle?'
```

Replace `<your-instance-external-ip>` with your Google Compute Engine instance's external IP address.

## Configuring Port Forwarding for Local Access

To access R2R from your local machine, use SSH port forwarding:

```bash
gcloud compute ssh --zone "your-zone" "your-instance-name" -- -L 7273:localhost:7273 -L 7274:localhost:7274
```

## Exposing Ports for Public Access (Optional)

To make R2R publicly accessible:

1. In the Google Cloud Console, go to "VPC network" > "Firewall".
2. Click "Create Firewall Rule".
3. Configure the rule:
   - Name: Allow-R2R
   - Target tags: r2r-server
   - Source IP ranges: 0.0.0.0/0
   - Specified protocols and ports: tcp:7272
4. Click "Create".

5. Add the network tag to your instance:
   - Go to Compute Engine > VM instances.
   - Click on your instance name.
   - Click "Edit".
   - Under "Network tags", add "r2r-server".
   - Click "Save".

6. Ensure R2R is configured to listen on all interfaces (0.0.0.0).

After starting your R2R application, users can access it at:

```
http://<your-instance-external-ip>:7272
```

## Security Considerations

- Use HTTPS (port 443) with a valid SSL certificate for production.
- Restrict source IP addresses in the firewall rule if possible.
- Regularly update and patch your system and applications.
- Monitor incoming traffic for suspicious activities.
- Remove or disable the firewall rule when not needed for testing.

## Conclusion

You have now successfully deployed R2R on Google Cloud Platform. The application should be accessible locally through SSH tunneling and optionally publicly through direct access to the Compute Engine instance. Remember to configure authentication and implement proper security measures before exposing your R2R instance to the public internet.

For more information on configuring and using R2R, refer to the [configuration documentation](/documentation/configuration/introduction).

---


# R2R CLI Documentation

## Installation

Before starting, make sure you have completed the [R2R installation](/documentation/installation).

The R2R CLI is automatically installed as part of the R2R Python package. No additional installation steps are required.

## Getting Started

1. Ensure R2R is running correctly by checking the health status:

```bash
r2r health
```

You should see output similar to:
```
{"status":"ok"}
```

2. Ingest a sample file to test the ingestion process:

```bash
r2r ingest-sample-file
```

This will download and ingest a sample text file from the R2R GitHub repository. You can verify the ingestion was successful by checking the documents overview:

```bash
r2r documents-overview
```

3. Perform a search on the ingested document:

```bash
r2r search --query "What did Aristotle contribute to philosophy?"
```

This will search the ingested document for relevant information to answer the query.

4. Generate a response using RAG (Retrieval-Augmented Generation):

```bash
r2r rag --query "What were Aristotle's main contributions to logic?"
```

This will perform a search on the ingested document and use the retrieved information to generate a complete response to the query.

## Additional Documentation

For more detailed information on specific functionalities of the R2R CLI, please refer to the following documentation:

- [Document Ingestion](/documentation/cli/ingestion): Learn how to add, retrieve, and manage documents using the CLI.
- [Search & RAG](/documentation/cli/retrieval): Explore various querying techniques and Retrieval-Augmented Generation capabilities.
- [Knowledge Graphs](/documentation/cli/graph): Learn how to create and enrich knowledge graphs, and perform GraphRAG.
- [Server Management](/documentation/cli/server): Manage your R2R server, including health checks, logs, and updates.

---


# Deploying R2R on Azure

Azure provides a robust and scalable platform for deploying R2R (RAG to Riches). This guide will walk you through the process of setting up R2R on an Azure Virtual Machine, making it accessible both locally and publicly.

## Overview

Deploying R2R on Azure involves the following main steps:

1. Creating an Azure Virtual Machine
2. Installing necessary dependencies
3. Setting up R2R
4. Configuring port forwarding for local access
5. Exposing ports for public access (optional)

This guide assumes you have an Azure account and the necessary permissions to create and manage Virtual Machines.

## Creating an Azure Virtual Machine

1. Log in to the [Azure Portal](https://portal.azure.com/).
2. Click on "Create a resource" and search for "Virtual Machine".
3. Choose `Ubuntu Server 22.04 LTS - x64 Gen2` as the operating system.
4. Select a VM size with at least 16GB of RAM, 4-8 vCPU cores, and 500GB of disk for a small-mid sized organization (< 5000 users). The `D4s_v3` series is a good starting point.
5. Configure networking settings to allow inbound traffic on ports `22` (SSH), and optionally `7272` (R2R API).
6. Review and create the VM.


## Exposing Ports for Public Access (Optional)

To make R2R publicly accessible:

1. Log in to the Azure Portal.
2. Navigate to your VM > Networking > Network Security Group.
3. Add a new inbound security rule:
   - Destination port ranges: 7272
   - Protocol: TCP
   - Action: Allow
   - Priority: 1000 (or lower than conflicting rules)
   - Name: Allow_7272

4. Ensure R2R is configured to listen on all interfaces (0.0.0.0).

After starting your R2R application, users can access it at:

```
http://<your-vm-ip-address>:7272
```

<Frame caption="Opening ports 7272 in our Azure cloud deployment">
  <img src="/images/azure_security.png" />
</Frame>

## Installing Dependencies

SSH into your newly created VM with a command like `ssh -i my_pem.pem azureuser@<your-vm-ip-address>`:

<Frame caption="System info at first connection to the remote instance">
  <img src="/images/ssh_print.png" />
</Frame>

Now, run the following commands to install the necessary R2R dependencies:


```bash
# Update package list and install Python and pip
sudo apt update
sudo apt install python3-pip

# Install R2R
pip install r2r

# Add R2R to PATH
echo 'export PATH=$PATH:$HOME/.local/bin' >> ~/.bashrc
source ~/.bashrc

# Install Docker
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add your user to the Docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify Docker installation
docker run hello-world
```

<Frame caption="Successful Docker hello-world output">
  <img src="/images/docker_hello_world.png" />
</Frame>


## Setting up R2R

1. Serve your R2R backend

```bash
# Set required remote providers
export OPENAI_API_KEY=sk-...

# Optional - pass in a custom configuration here
r2r serve  --docker --full
```


<Frame caption="Printout for a successful deployment">
  <img src="images/azure_deployment.png" />
</Frame>

2. Double check the health of the system

```bash
 r2r health
```

```bash
Time taken: 0.01 seconds
{'results': {'response': 'ok'}}
```

3. Test ingesting and searching a sample document from a remote environment

```bash
# From your local machine

r2r --base-url=http://<your-vm-ip-address>:7272 ingest-sample-file
sleep 10
r2r --base-url=http://<your-vm-ip-address>:7272 search --query='Who was aristotle?'
```

```bash

Time taken: 0.43 seconds
Sample file ingestion completed. Ingest files response:

[{'message': 'Ingestion task queued successfully.', 'task_id': '887f9f99-cc18-4c1e-8f61-facf1d212334', 'document_id': '9fbe403b-c11c-5aae-8ade-ef22980c3ad1'}]
Vector search results:
{'fragment_id': 'ecc754cd-380d-585f-84ac-021542ef3c1d', 'extraction_id': '92d78034-8447-5046-bf4d-e019932fbc20', 'document_id': '9fbe403b-c11c-5aae-8ade-ef22980c3ad1', 'user_id': '2acb499e-8428-543b-bd85-0d9098718220', 'collection_ids': [], 'score': 0.7822163571248282, 'text': 'Aristotle[A] (Greek: Ἀριστοτέλης Aristotélēs, pronounced [aristotélɛːs]; 384–322 BC) was an Ancient Greek philosopher and polymath. His writings cover a broad range of subjects spanning the natural sciences, philosophy, linguistics, economics, politics, psychology, and the arts. As the founder of the Peripatetic school of philosophy in the Lyceum in Athens, he began the wider Aristotelian tradition that followed, which set the groundwork for the development of modern science.', 'metadata': {'title': 'aristotle.txt', 'version': 'v0', 'chunk_order': 0, 'document_type': 'txt', 'unstructured_filetype': 'text/plain', 'unstructured_languages': ['eng'], 'partitioned_by_unstructured': True, 'associatedQuery': 'Who was aristotle?'}}
...
```

## Configuring Port Forwarding for Local Access

To access R2R from your local machine, use SSH port forwarding:

```bash
ssh i my_pem.pem -L 7273:localhost:7273 -L 7274:localhost:7274 azureuser@<your-vm-ip-address>
```

Replace `<your-vm-ip-address>` with your Azure VM's public IP address or DNS name. Note that in the R2R dashboard you will still need to use the remote VM address as requests are made from the client-side.

## Security Considerations

- Use HTTPS (port 443) with a valid SSL certificate for production.
- Restrict source IP addresses in the security rule if possible.
- Regularly update and patch your system and applications.
- Monitor incoming traffic for suspicious activities.
- Remove or disable the rule when not needed for testing.

## Conclusion

You have now successfully deployed R2R on Azure. The application should be accessible locally through SSH tunneling and optionally publicly through direct access to the Azure VM. Remember to configure authentication and implement proper security measures before exposing your R2R instance to the public internet.

For more information on configuring and using R2R, refer to the [configuration documentation](/documentation/configuration/introduction).

---

# R2R Security Group and Firewall Configuration Guide

Proper security group and firewall configuration is crucial for securing your R2R deployment while ensuring necessary services remain accessible. This guide covers configurations for both cloud environments and local deployments.

## Cloud Environments (AWS, Azure, GCP)

### AWS Security Groups

1. Create a new security group for your R2R deployment:

```bash
aws ec2 create-security-group --group-name R2R-SecurityGroup --description "Security group for R2R deployment"
```

2. Configure inbound rules:

```bash
# Allow SSH access (restrict to your IP if possible)
aws ec2 authorize-security-group-ingress --group-name R2R-SecurityGroup --protocol tcp --port 22 --cidr 0.0.0.0/0

# Allow access to R2R API
aws ec2 authorize-security-group-ingress --group-name R2R-SecurityGroup --protocol tcp --port 7272 --cidr 0.0.0.0/0

# Allow access to R2R Dashboard
aws ec2 authorize-security-group-ingress --group-name R2R-SecurityGroup --protocol tcp --port 8001 --cidr 0.0.0.0/0

# Allow access to Hatchet Dashboard
aws ec2 authorize-security-group-ingress --group-name R2R-SecurityGroup --protocol tcp --port 8002 --cidr 0.0.0.0/0
```

### Azure Network Security Groups

1. Create a new Network Security Group:

```bash
az network nsg create --name R2R-NSG --resource-group YourResourceGroup --location YourLocation
```

2. Add inbound security rules:

```bash
# Allow SSH access
az network nsg rule create --name AllowSSH --nsg-name R2R-NSG --priority 100 --resource-group YourResourceGroup --access Allow --direction Inbound --protocol Tcp --source-address-prefixes '*' --source-port-ranges '*' --destination-address-prefixes '*' --destination-port-ranges 22

# Allow R2R API access
az network nsg rule create --name AllowR2RAPI --nsg-name R2R-NSG --priority 200 --resource-group YourResourceGroup --access Allow --direction Inbound --protocol Tcp --source-address-prefixes '*' --source-port-ranges '*' --destination-address-prefixes '*' --destination-port-ranges 7272

# Allow R2R Dashboard access
az network nsg rule create --name AllowR2RDashboard --nsg-name R2R-NSG --priority 300 --resource-group YourResourceGroup --access Allow --direction Inbound --protocol Tcp --source-address-prefixes '*' --source-port-ranges '*' --destination-address-prefixes '*' --destination-port-ranges 8001

# Allow Hatchet Dashboard access
az network nsg rule create --name AllowHatchetDashboard --nsg-name R2R-NSG --priority 400 --resource-group YourResourceGroup --access Allow --direction Inbound --protocol Tcp --source-address-prefixes '*' --source-port-ranges '*' --destination-address-prefixes '*' --destination-port-ranges 8002
```

### Google Cloud Platform Firewall Rules

1. Create firewall rules:

```bash
# Allow SSH access
gcloud compute firewall-rules create allow-ssh --direction=INGRESS --priority=1000 --network=default --action=ALLOW --rules=tcp:22 --source-ranges=0.0.0.0/0

# Allow R2R API access
gcloud compute firewall-rules create allow-r2r-api --direction=INGRESS --priority=1000 --network=default --action=ALLOW --rules=tcp:7272 --source-ranges=0.0.0.0/0

# Allow R2R Dashboard access
gcloud compute firewall-rules create allow-r2r-dashboard --direction=INGRESS --priority=1000 --network=default --action=ALLOW --rules=tcp:8001 --source-ranges=0.0.0.0/0

# Allow Hatchet Dashboard access
gcloud compute firewall-rules create allow-hatchet-dashboard --direction=INGRESS --priority=1000 --network=default --action=ALLOW --rules=tcp:8002 --source-ranges=0.0.0.0/0
```

## Local Deployments

For local deployments, you'll need to configure your operating system's firewall. Here are instructions for common operating systems:

### Ubuntu/Debian (UFW)

```bash
# Allow SSH
sudo ufw allow 22/tcp

# Allow R2R API
sudo ufw allow 7272/tcp

# Allow R2R Dashboard
sudo ufw allow 8001/tcp

# Allow Hatchet Dashboard
sudo ufw allow 8002/tcp

# Enable the firewall
sudo ufw enable
```

### CentOS/RHEL (firewalld)

```bash
# Allow SSH
sudo firewall-cmd --permanent --add-port=22/tcp

# Allow R2R API
sudo firewall-cmd --permanent --add-port=7272/tcp

# Allow R2R Dashboard
sudo firewall-cmd --permanent --add-port=8001/tcp

# Allow Hatchet Dashboard
sudo firewall-cmd --permanent --add-port=8002/tcp

# Reload firewall
sudo firewall-cmd --reload
```

### Windows (Windows Firewall)

1. Open Windows Defender Firewall with Advanced Security
2. Click on "Inbound Rules" and then "New Rule"
3. Choose "Port" and click "Next"
4. Select "TCP" and enter the specific ports (22, 7272, 8001, 8002)
5. Choose "Allow the connection" and click "Next"
6. Apply the rule to all profiles (Domain, Private, Public)
7. Give the rule a name (e.g., "R2R Ports") and click "Finish"

## Best Practices

1. **Least Privilege**: Only open ports that are absolutely necessary.
2. **IP Restrictions**: When possible, restrict access to known IP addresses or ranges.
3. **Use VPN**: For added security, consider using a VPN for accessing administrative interfaces.
4. **Regular Audits**: Periodically review and update your security group and firewall rules.
5. **Monitoring**: Implement logging and monitoring for all allowed ports.
6. **HTTPS**: Use HTTPS for all web interfaces and APIs when possible.

## Verifying Configuration

After setting up your firewall rules, verify that the necessary ports are open:

```bash
# For Linux systems
nmap -p 22,7272,8001,8002,7474 your_server_ip

# For Windows systems (requires nmap installation)
nmap -p 22,7272,8001,8002,7474 your_server_ip
```

This should show the status of each port (open or closed).

Remember to adjust these configurations based on your specific deployment needs and security requirements. Always follow your organization's security policies and best practices.

---


<Note>
Occasionally this SDK documentation falls out of date, cross-check with the automatcially generated <a href="/api-reference/introduction"> API Reference documentation </a> for the latest parameters.
</Note>


## User Authentication and Management

R2R provides a comprehensive set of user authentication and management features, allowing you to implement secure and feature-rich authentication systems in your applications.

### User Registration

To register a new user:

```python
register_response = client.register("user@example.com", "password123")
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      ```python
      {
        'results': {
          'email': 'user@example.com',
          'id': 'bf417057-f104-4e75-8579-c74d26fcbed3',
          'hashed_password': '$2b$12$p6a9glpAQaq.4uzi4gXQru6PN7WBpky/xMeYK9LShEe4ygBf1L.pK',
          'is_superuser': False,
          'is_active': True,
          'is_verified': False,
          'verification_code_expiry': None,
          'name': None,
          'bio': None,
          'profile_picture': None,
          'created_at': '2024-07-16T22:53:47.524794Z',
          'updated_at': '2024-07-16T22:53:47.524794Z'
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Email Verification

If email verification is enabled, verify a user's email:

```python
verify_response = client.verify_email("verification_code_here")
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      ```python
      {
        "results": {
          "message": "Email verified successfully"
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### User Login

To log in and obtain access tokens:

```python
login_response = client.login("user@example.com", "password123")
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      ```python
      {
        'results': {
          'access_token': {
            'token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...',
            'token_type': 'access'
          },
          'refresh_token': {
            'token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...',
            'token_type': 'refresh'
          }
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Get Current User Info

Retrieve information about the currently authenticated user:

```python
user_info = client.user()
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      ```python
      {
        'results': {
          'email': 'user@example.com',
          'id': '76eea168-9f98-4672-af3b-2c26ec92d7f8',
          'hashed_password': 'null',
          'is_superuser': False,
          'is_active': True,
          'is_verified': True,
          'verification_code_expiry': None,
          'name': None,
          'bio': None,
          'profile_picture': None,
          'created_at': '2024-07-16T23:06:42.123303Z',
          'updated_at': '2024-07-16T23:22:48.256239Z'
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Refresh Access Token

Refresh an expired access token:

```python
refresh_response = client.refresh_access_token()
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      ```python
      {
        'results': {
          'access_token': {
            'token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...',
            'token_type': 'access'
          },
          'refresh_token': {
            'token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...',
            'token_type': 'refresh'
          }
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Change Password

Change the user's password:

```python
change_password_result = client.change_password("password123", "new_password")
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      ```python
      {
        "result": {
          "message": "Password changed successfully"
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Request Password Reset

Request a password reset for a user:

```python
reset_request_result = client.request_password_reset("user@example.com")
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      ```python
      {
        "result": {
          "message": "If the email exists, a reset link has been sent"
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Confirm Password Reset

Confirm a password reset using the reset token:

```python
reset_confirm_result = client.confirm_password_reset("reset_token_here", "new_password")
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      ```python
      {
        "result": {
          "message": "Password reset successfully"
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Update User Profile

Update the user's profile information:

```python
update_result = client.update_user(name="John Doe", bio="R2R enthusiast")
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      ```python
      {
        'results': {
          'email': 'user@example.com',
          'id': '76eea168-9f98-4672-af3b-2c26ec92d7f8',
          'hashed_password': 'null',
          'is_superuser': False,
          'is_active': True,
          'is_verified': True,
          'verification_code_expiry': None,
          'name': 'John Doe',
          'bio': 'R2R enthusiast',
          'profile_picture': None,
          'created_at': '2024-07-16T23:06:42.123303Z',
          'updated_at': '2024-07-16T23:22:48.256239Z'
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Delete User Account

Delete the user's account:

```python
user_id = register_response["results"]["id"] # input unique id here
delete_result = client.delete_user(user_id, "password123")
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      ```python
      {
        'results': {
          'message': 'User account deleted successfully'
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### User Logout

Log out and invalidate the current access token:

```python
logout_response = client.logout()
```

<AccordionGroup>
  <Accordion title="Response">
    <ResponseField name="response" type="dict">
      ```python
      {
        'results': {
          'message': 'Logged out successfully'
        }
      }
      ```
    </ResponseField>
  </Accordion>
</AccordionGroup>

### Superuser Capabilities

Superusers have additional privileges, including access to system-wide operations and sensitive information. To use superuser capabilities, authenticate as a superuser or the default admin:

```python
# Login as admin
login_result = client.login("admin@example.com", "admin_password")

# Access superuser features
users_overview = client.users_overview()
logs = client.logs()
analytics_result = client.analytics(
    {"all_latencies": "search_latency"},
    {"search_latencies": ["basic_statistics", "search_latency"]}
)
```

<Note>
Superuser actions should be performed with caution and only by authorized personnel. Ensure proper security measures are in place when using superuser capabilities.
</Note>

## Security Considerations

When implementing user authentication, consider the following best practices:

1. Always use HTTPS in production to encrypt data in transit.
2. Implement rate limiting to protect against brute-force attacks.
3. Use secure password hashing (R2R uses bcrypt by default).
4. Consider implementing multi-factor authentication (MFA) for enhanced security.
5. Conduct regular security audits of your authentication system.

For more advanced use cases or custom implementations, refer to the R2R documentation or reach out to the community for support.

---

# Using the R2R Health Check Endpoint

The health check endpoint in R2R provides a quick and easy way to verify the status of your R2R deployment. This guide will walk you through how to use this endpoint, interpret its results, and integrate it into your monitoring systems.

## 1. Understanding the Health Check Endpoint

The health check endpoint is a specific URL that, when accessed, returns information about the current state of the R2R system. It typically checks various components and dependencies to ensure everything is functioning correctly.

## 2. Accessing the Health Check Endpoint

The health check endpoint is usually available at `/v2/health`. Here are different ways to access it:

### 2.1 Using cURL

You can use cURL to make a GET request to the health check endpoint:

```bash
curl http://localhost:7272/v2/health
```

### 2.2 Using a Web Browser

If your R2R instance is accessible via a web browser, you can simply navigate to:

```
http://localhost:7272/v2/health
```

Replace `localhost:7272` with the appropriate host and port if your setup is different.

### 2.3 Using the R2R CLI

R2R provides a CLI command to check the health of the system:

```bash
r2r health
```

### 2.4 Using the Python Client

If you're using the R2R Python client:

```python
from r2r import R2R

client = R2R()
health_status = client.health()
print(health_status)
```

## 3. Interpreting Health Check Results

The health check endpoint typically returns a JSON response. Here's an example of what it might look like:

```json
{
  "status": "healthy",
  "version": "3.1.22",
  "components": {
    "database": {
      "status": "healthy",
      "message": "Connected to database successfully"
    },
    "vector_store": {
      "status": "healthy",
      "message": "Vector store is operational"
    },
    "llm_service": {
      "status": "healthy",
      "message": "LLM service is responding"
    }
  },
  "timestamp": "2024-09-11T15:30:45Z"
}
```

Key elements to look for:
- Overall `status`: Should be "healthy" if everything is okay.
- `version`: Indicates the current version of R2R.
- `components`: Shows the status of individual components.
- `timestamp`: When the health check was performed.

## 4. Common Issues and Troubleshooting

If the health check returns a non-healthy status, here are some common issues and how to address them:

### 4.1 Database Connection Issues

If the database component shows as unhealthy:
- Check database credentials in your R2R configuration.
- Ensure the database server is running and accessible.
- Verify network connectivity between R2R and the database.

### 4.2 Vector Store Problems

For vector store issues:
- Check if the vector store service (e.g., Postgres with pgvector) is running.
- Verify the vector store configuration in R2R settings.

### 4.3 LLM Service Not Responding

If the LLM service is unhealthy:
- Check your API key for the LLM service (e.g., OpenAI API key).
- Ensure you have internet connectivity if using a cloud-based LLM.
- Verify the LLM service endpoint in your configuration.

## 5. Integrating Health Checks into Monitoring Systems

To ensure continuous monitoring of your R2R deployment:

### 5.1 Prometheus Integration

If you're using Prometheus for monitoring:

1. Set up a Prometheus exporter that periodically calls the health check endpoint.
2. Configure Prometheus to scrape metrics from this exporter.
3. Set up alerts for when the health status is not "healthy".

### 5.2 Kubernetes Liveness Probe

If deploying R2R in Kubernetes, use the health check as a liveness probe:

```yaml
livenessProbe:
  httpGet:
    path: /v2/health
    port: 7272
  initialDelaySeconds: 30
  periodSeconds: 10
```

### 5.3 AWS CloudWatch

For AWS deployments:

1. Create a CloudWatch synthetic canary that periodically calls the health check endpoint.
2. Set up CloudWatch alarms based on the canary's results.

## 6. Best Practices

1. Regular Checks: Perform health checks at regular intervals (e.g., every 5 minutes).
2. Alerting: Set up alerts for when the health check fails consistently.
3. Logging: Log health check results for historical analysis.
4. Trend Analysis: Monitor trends in response times of the health check endpoint.
5. Comprehensive Checks: Ensure your health check covers all critical components of your R2R deployment.

## 7. Advanced Health Check Customization

R2R allows for customization of the health check endpoint. You can add custom checks or modify existing ones by editing the health check configuration. Refer to the R2R documentation for detailed instructions on how to customize health checks for your specific deployment needs.

## Conclusion

The health check endpoint is a crucial tool for maintaining the reliability and performance of your R2R deployment. By regularly utilizing this endpoint and integrating it into your monitoring systems, you can ensure quick detection and resolution of any issues that may arise in your R2R system.

For more detailed information on R2R's features and advanced configurations, refer to the [official R2R documentation](https://r2r-docs.sciphi.ai/) or join the [R2R Discord community](https://discord.gg/p6KqD2kjtB) for support and discussions.

---


# SciPhi Enterprise: Fully Managed R2R for Your Organization

SciPhi offers a fully managed, enterprise-grade solution for deploying and scaling R2R (RAG to Riches) within your organization. The SciPhi Enterprise offering provides all the benefits of R2R, including multimodal ingestion, hybrid search, GraphRAG, user management, and observability, in a hassle-free, scalable environment.

## Why Use SciPhi Enterprise?

- **Fully Managed**: SciPhi handles the infrastructure, deployment, scaling, updates, and maintenance of R2R, so your team can focus on building RAG applications.
- **Scalable**: Seamlessly scale your R2R deployment to handle growing user bases, document collections, and query volumes.
- **Secure**: Benefit from enterprise-level security, compliance, and data privacy measures.
- **Customizable**: Tailor your R2R deployment to your organization's specific requirements and integrate with your existing systems and workflows.
- **Expert Support**: Get direct access to the SciPhi team for guidance, troubleshooting, and best practices.

## Getting Started

Getting started with SciPhi Enterprise is easy:

1. **Contact SciPhi**: Reach out to their team at [founders@sciphi.ai](mailto:founders@sciphi.ai) to discuss your organization's RAG application needs and learn more about SciPhi Enterprise.
2. **Discovery**: SciPhi's experts will work with you to understand your requirements, existing systems, and goals for R2R within your organization.
3. **Deployment**: SciPhi will handle the deployment and configuration of R2R in your environment, whether cloud-based or on-premises, and integrate with your existing systems and workflows.
4. **Onboarding**: SciPhi will provide training and support to help your developers and users get up and running with R2R quickly and effectively.
5. **Ongoing Support**: SciPhi Enterprise provides ongoing support, updates, and guidance as you scale and evolve your RAG applications.

---

# R2R Service Configuration Troubleshooting Guide: Postgres, Hatchet

This guide addresses common configuration problems for Postgres, and Hatchet services in R2R deployments.

## Postgres Configuration Issues

### 1. Connection Failures

**Symptom:** R2R cannot connect to Postgres database.

**Possible Causes and Solutions:**

a) Incorrect connection string:
   - Verify the `DATABASE_URL` environment variable.
   - Ensure it follows the format: `postgres://user:password@host:port/dbname`

b) Network issues:
   - Check if Postgres is running and accessible from the R2R container.
   - Verify network settings in Docker Compose file.

c) Authentication problems:
   - Confirm that the username and password in the connection string are correct.
   - Check Postgres logs for failed authentication attempts.

### 2. pgvector Extension Missing

**Symptom:** Vector operations fail or R2R reports missing pgvector functionality.

**Solution:**
- Ensure you're using the `pgvector/pgvector` image instead of the standard Postgres image.
- Verify that the pgvector extension is created in your database:
  ```sql
  CREATE EXTENSION IF NOT EXISTS vector;
  ```

### 3. Insufficient Resources

**Symptom:** Postgres performance issues or crashes under load.

**Solution:**
- Adjust resource limits in Docker Compose:
  ```yaml
  postgres:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
  ```
- Tune Postgres configuration parameters like `shared_buffers`, `effective_cache_size`, etc.


## Hatchet Configuration Issues

### 1. RabbitMQ Connection Problems

**Symptom:** Hatchet cannot connect to RabbitMQ.

**Solution:**
- Verify RabbitMQ connection string:
  ```yaml
  environment:
    - SERVER_TASKQUEUE_RABBITMQ_URL=amqp://user:password@hatchet-rabbitmq:5672/
  ```
- Ensure RabbitMQ service is healthy and accessible.

### 2. Hatchet API Key Issues

**Symptom:** R2R cannot authenticate with Hatchet.

**Solution:**
- Check the Hatchet API key generation process:
  ```yaml
  setup-token:
    command: >
      sh -c "
        TOKEN=$(/hatchet/hatchet-admin token create --config /hatchet/config --tenant-id your-tenant-id)
        echo $TOKEN > /hatchet_api_key/api_key.txt
      "
  ```
- Verify that R2R is correctly reading the API key:
  ```yaml
  r2r:
    environment:
      - HATCHET_CLIENT_TOKEN=${HATCHET_CLIENT_TOKEN}
    command: >
      sh -c '
        export HATCHET_CLIENT_TOKEN=$(cat /hatchet_api_key/api_key.txt)
        exec your_r2r_command
      '
  ```

### 3. Hatchet Engine Health Check Failures

**Symptom:** Hatchet Engine service fails health checks.

**Solution:**
- Verify Hatchet Engine configuration:
  ```yaml
  hatchet-engine:
    environment:
      - SERVER_GRPC_BROADCAST_ADDRESS=host.docker.internal:7077
      - SERVER_GRPC_BIND_ADDRESS=0.0.0.0
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:7077/health"]
      interval: 10s
      timeout: 5s
      retries: 5
  ```
- Check Hatchet Engine logs for startup errors.
- Ensure all required environment variables are set correctly.

## General Troubleshooting Tips

1. **Check Logs:** Always start by examining the logs of the problematic service:
   ```
   docker-compose logs postgres
   docker-compose logs hatchet-engine
   ```

2. **Verify Network Connectivity:** Ensure services can communicate:
   ```
   docker-compose exec r2r ping postgres
   docker-compose exec r2r ping hatchet-engine
   ```

3. **Check Resource Usage:** Monitor CPU and memory usage:
   ```
   docker stats
   ```

4. **Recreate Containers:** Sometimes, recreating containers can resolve issues:
   ```
   docker-compose up -d --force-recreate <service-name>
   ```

5. **Verify Volumes:** Ensure data persistence by checking volume mounts:
   ```
   docker volume ls
   docker volume inspect <volume-name>
   ```

6. **Environment Variables:** Double-check all environment variables in your `.env` file and `docker-compose.yml`.

By following this guide, you should be able to diagnose and resolve most configuration issues related to Postgres, and Hatchet in your R2R deployment. If problems persist, consider seeking help from the R2R community or support channels.

---

# R2R Troubleshooting Guide: Incorrect Database Credentials

Database connectivity issues due to incorrect credentials are a common problem when deploying R2R. This guide will help you identify and resolve these issues.

## Symptoms

- Error messages containing phrases like "authentication failed", "access denied", or "connection refused"
- R2R application fails to start or crashes shortly after starting
- Database-related operations fail while other parts of the application seem to work

## Diagnosis Steps

1. **Check Error Logs**
   - Review R2R application logs for specific error messages
   - Look for database-related errors, especially those mentioning authentication or connection issues

2. **Verify Environment Variables**
   - Ensure all database-related environment variables are set correctly
   - Common variables include:
     ```
     R2R_POSTGRES_USER
     R2R_POSTGRES_PASSWORD
     R2R_POSTGRES_HOST
     R2R_POSTGRES_PORT
     R2R_POSTGRES_DBNAME
     ```

3. **Test Database Connection**
   - Use a database client tool to test the connection with the same credentials
   - For PostgreSQL, you can use the `psql` command-line tool:
     ```
     psql -h <host> -p <port> -U <username> -d <database_name>
     ```

4. **Check Database Server Status**
   - Ensure the database server is running and accessible from the R2R container
   - Verify network connectivity between R2R and the database server

5. **Inspect Docker Compose File**
   - Review the `docker-compose.yml` file to ensure database service configuration is correct
   - Check for any discrepancies between the database service and R2R service configurations

## Resolution Steps

1. **Correct Environment Variables**
   - Update the `.env` file or set environment variables with the correct database credentials
   - Ensure these variables are properly passed to the R2R container

2. **Update Docker Compose File**
   - If using Docker Compose, update the `docker-compose.yml` file with the correct database configuration
   - Ensure the database service name matches what R2R is expecting (e.g., `postgres`)

3. **Rebuild and Restart Containers**
   - After making changes, rebuild and restart your Docker containers:
     ```
     docker-compose down
     docker-compose up --build
     ```

4. **Check Database User Permissions**
   - Ensure the database user has the necessary permissions
   - For PostgreSQL, you might need to grant permissions:
     ```sql
     GRANT ALL PRIVILEGES ON DATABASE <database_name> TO <username>;
     ```

5. **Verify Database Existence**
   - Ensure the specified database exists
   - Create it if necessary:
     ```sql
     CREATE DATABASE <database_name>;
     ```

6. **Check Network Configuration**
   - If using Docker, ensure the database and R2R services are on the same network
   - Verify firewall rules allow traffic between R2R and the database

7. **Use Secrets Management**
   - Consider using Docker secrets or a secure vault for managing sensitive credentials
   - Update your Docker Compose file to use secrets instead of environment variables for passwords

## Prevention Tips

1. Use a `.env` file for local development and CI/CD pipelines for production to manage environment variables
2. Implement a health check in your Docker Compose file to ensure the database is ready before starting R2R
3. Use database connection pooling to manage connections efficiently
4. Regularly audit and rotate database credentials
5. Use least privilege principle when setting up database users for R2R

## Debugging Commands

Here are some useful commands for debugging database connection issues:

1. Check if PostgreSQL is running:
   ```
   docker-compose ps postgres
   ```

2. View PostgreSQL logs:
   ```
   docker-compose logs postgres
   ```

3. Check R2R logs:
   ```
   docker-compose logs r2r
   ```

4. Access PostgreSQL CLI within the container:
   ```
   docker-compose exec postgres psql -U <username> -d <database_name>
   ```

## Seeking Further Help

If you've tried these steps and are still experiencing issues:

1. Check the R2R documentation for any specific database setup requirements
2. Review the R2R GitHub issues for similar problems and solutions
3. Reach out to the R2R community on Discord or GitHub for support
4. Provide detailed information about your setup, including:
   - R2R version
   - Database type and version
   - Relevant parts of your Docker Compose file
   - Specific error messages you're encountering

Remember to never share actual passwords or sensitive information when seeking help. Use placeholders instead.

By following this guide, you should be able to resolve most database credential issues in your R2R deployment. If problems persist, don't hesitate to seek help from the R2R community or support channels.

---


# Troubleshooting

Have you encountered issues with deploying your R2R system? Have no fear our troubleshooting guide is here.

# R2R Troubleshooting Guide

## Common Installation Issues

### Docker Installation
- Issue: [Docker containers fail to start](/documentation/deployment/troubleshooting/docker)
- Issue: [Missing environment variables](/documentation/deployment/troubleshooting/environment)
- Issue: [Insufficient system resources](documentation/deployment/troubleshooting/resources)

### Local System Installation
- Issue: [Dependency conflicts](/documentation/deployment/troubleshooting/dependencies)
- Issue: [Service configuration](/documentation/deployment/troubleshooting/services) problems (Postgres, Hatchet)
- Issue: [Unstructured.io](/documentation/deployment/troubleshooting/unstructured) setup difficulties

## Deployment Problems

### Cloud Deployments
- Issue: [Connection timeouts](/documentation/deployment/troubleshooting/timeouts)
- Issue: [Insufficient instance resources](/documentation/deployment/troubleshooting/insufficient_resources)
- Issue: [Security group / firewall configuration](/documentation/deployment/troubleshooting/firewall)

### Other
- Issue: [Port conflicts](/documentation/deployment/troubleshooting/port_conflicts)
- Issue: [Database connection failures](/documentation/deployment/troubleshooting/database)
- Issue: [Local LLM integration issues](/documentation/deployment/troubleshooting/local_llm)

## Runtime Errors

### API-related Issues
- Issue: [API endpoints not responding](/documentation/deployment/troubleshooting/api_connections)
- Issue: [Unexpected API responses](/documentation/deployment/troubleshooting/api_responses)

### Performance Issues
- Issue: [Slow query responses](/documentation/deployment/troubleshooting/slow_queries)
- Issue: [High resource usage](/documentation/deployment/troubleshooting/high_usage)

## Configuration Troubleshooting

### Environment Variables
- Issue: [Missing or incorrect API keys](/documentation/deployment/troubleshooting/missing_keys)
- Issue: [Incorrect database credentials](/documentation/deployment/troubleshooting/incorrect_credentials)

### Custom Configurations
- Issue: [TOML file syntax errors](/documentation/deployment/troubleshooting/toml_errors)
- Issue: [Incompatible configuration settings](/documentation/deployment/troubleshooting/bad_configuration)

## Component-specific Issues

### Postgres + pgvector
- Issue: [Vector storage problems](/documentation/deployment/troubleshooting/vector_store_issues)
- Issue: [Connection string errors](/documentation/deployment/troubleshooting/connection_strings)

### Hatchet
- Issue: [Workflow orchestration failures](/documentation/deployment/troubleshooting/workflows)
- Issue: [RabbitMQ connectivity issues](/documentation/deployment/troubleshooting/rabbit_mq)

## Debugging Tips

- [How to check R2R logs](/documentation/deployment/troubleshooting/r2r_logs))

## Getting Help

- [Report issues on GitHub](https://github.com/SciPhi-AI/R2R/issues)
- [Joining the Discord community for support](https://discord.gg/p6KqD2kjtB)

---

## Knowledge Graph Management

### Create Graph

Create a knowledge graph for a collection using the `create-graph` command:

````bash
r2r create-graph --collection-id my-collection --run --kg-creation-settings '{"key": "value"}' --force-kg-creation
````

<AccordionGroup>
  <Accordion title="Arguments">
    <ParamField path="--collection-id" type="str">
      Collection ID to create graph for. Default is "".
    </ParamField>

    <ParamField path="--run" type="flag">
      Run the graph creation process.
    </ParamField>

    <ParamField path="--kg-creation-settings" type="str">
      Settings for the graph creation process as a JSON string.
    </ParamField>

    <ParamField path="--force-kg-creation" type="flag">
      Force the graph creation process.
    </ParamField>
  </Accordion>
</AccordionGroup>

### Deduplicate Entities

Deduplicate entities in a collection using the `deduplicate-entities` command:

````bash
r2r deduplicate-entities --collection-id my-collection --run --deduplication-settings '{"key": "value"}'
````



### Enrich Graph

Enrich an existing knowledge graph using the `enrich-graph` command:

````bash
r2r enrich-graph --collection-id my-collection --run --force-kg-enrichment --kg-enrichment-settings '{"key": "value"}'
````

<AccordionGroup>
  <Accordion title="Arguments">
    <ParamField path="--collection-id" type="str">
      Collection ID to enrich graph for. Default is "".
    </ParamField>

    <ParamField path="--run" type="flag">
      Run the graph enrichment process.
    </ParamField>

    <ParamField path="--force-kg-enrichment" type="flag">
      Force the graph enrichment process.
    </ParamField>

    <ParamField path="--kg-enrichment-settings" type="str">
      Settings for the graph enrichment process as a JSON string.
    </ParamField>
  </Accordion>
</AccordionGroup>

### Get Entities

Retrieve entities from the knowledge graph using the `get-entities` command:

````bash
r2r get-entities --collection-id my-collection --offset 0 --limit 10 --entity-ids entity1 entity2
````

<AccordionGroup>
  <Accordion title="Arguments">
    <ParamField path="--collection-id" type="str" required>
      Collection ID to retrieve entities from.
    </ParamField>

    <ParamField path="--offset" type="int">
      Offset for pagination. Default is 0.
    </ParamField>

    <ParamField path="--limit" type="int">
      Limit for pagination. Default is 100.
    </ParamField>

    <ParamField path="--entity-ids" type="list[str]">
      Entity IDs to filter by.
    </ParamField>
  </Accordion>
</AccordionGroup>

### Get Triples

Retrieve triples from the knowledge graph using the `get-triples` command:

````bash
r2r get-triples --collection-id my-collection --offset 0 --limit 10 --triple-ids triple1 triple2 --entity-names entity1 entity2
````

<AccordionGroup>
  <Accordion title="Arguments">
    <ParamField path="--collection-id" type="str" required>
      Collection ID to retrieve triples from.
    </ParamField>

    <ParamField path="--offset" type="int">
      Offset for pagination. Default is 0.
    </ParamField>

    <ParamField path="--limit" type="int">
      Limit for pagination. Default is 100.
    </ParamField>

    <ParamField path="--triple-ids" type="list[str]">
      Triple IDs to filter by.
    </ParamField>

    <ParamField path="--entity-names" type="list[str]">
      Entity names to filter by.
    </ParamField>
  </Accordion>
</AccordionGroup>

### Delete Graph

Delete the graph for a collection using the `delete-graph` command:

````bash
r2r delete-graph --collection-id my-collection --cascade
````

<AccordionGroup>
  <Accordion title="Arguments">
    <ParamField path="--collection-id" type="str" required>
      Collection ID to delete the graph for.
    </ParamField>

    <ParamField path="--cascade" type="flag">
      Whether to cascade the deletion.

      NOTE: Setting this flag to true will delete entities and triples for documents that are shared across multiple collections. Do not set this flag unless you are absolutely sure that you want to delete the entities and triples for all documents in the collection.
    </ParamField>
  </Accordion>
</AccordionGroup>

---

# Troubleshooting Guide: Database Connection Failures in R2R

Database connection issues can significantly impact the functionality of your R2R deployment. This guide will help you diagnose and resolve common database connection problems for both Postgres.

## 1. General Troubleshooting Steps

Before diving into database-specific issues, try these general troubleshooting steps:

1. **Check Database Service Status**: Ensure the database service is running.
   ```bash
   docker ps | grep postgres
   ```

2. **Verify Network Connectivity**: Ensure the R2R service can reach the database.
   ```bash
   docker exec r2r-container ping postgres
   ```

3. **Check Logs**: Examine R2R and database container logs for error messages.
   ```bash
   docker logs r2r-container
   docker logs postgres-container
   ```

4. **Verify Environment Variables**: Ensure all necessary environment variables are correctly set in your Docker Compose file or deployment configuration.

## 2. Postgres Connection Issues

### 2.1 Common Postgres Error Messages

- "FATAL: password authentication failed for user"
- "FATAL: database does not exist"
- "could not connect to server: Connection refused"

### 2.2 Troubleshooting Steps for Postgres

1. **Check Postgres Connection String**:
   - Verify the `POSTGRES_*` environment variables in your R2R configuration.
   - Ensure the host, port, username, password, and database name are correct.

2. **Test Postgres Connection**:
   ```bash
   docker exec postgres-container psql -U your_username -d your_database -c "SELECT 1;"
   ```

3. **Check Postgres Logs**:
   ```bash
   docker logs postgres-container
   ```

4. **Verify Postgres User and Database**:
   ```bash
   docker exec postgres-container psql -U postgres -c "\du"
   docker exec postgres-container psql -U postgres -c "\l"
   ```

5. **Check Postgres Network Settings**:
   - Ensure Postgres is configured to accept connections from other containers.
   - Verify the `pg_hba.conf` file allows connections from the R2R container's IP range.

### 2.3 Common Solutions for Postgres Issues

- Update the Postgres connection string in R2R configuration.
- Recreate the Postgres user or database if they're missing.
- Modify Postgres network settings to allow connections from R2R.

## 3. Advanced Troubleshooting

### 3.1 Database Container Health Checks

Ensure your Docker Compose file includes proper health checks for database services:

```yaml
healthcheck:
  test: ["CMD-SHELL", "pg_isready -U postgres"]
  interval: 10s
  timeout: 5s
  retries: 5
```

### 3.2 Network Debugging

If network issues persist:

1. Inspect the Docker network:
   ```bash
   docker network inspect r2r-network
   ```

2. Use network debugging tools within containers:
   ```bash
   docker exec r2r-container netstat -tuln
   docker exec postgres-container netstat -tuln
   ```

### 3.3 Volume Permissions

Check if volume permissions are causing issues:

1. Inspect volume permissions:
   ```bash
   docker exec postgres-container ls -l /var/lib/postgresql/data
   ```

2. Adjust permissions if necessary:
   ```bash
   docker exec postgres-container chown -R postgres:postgres /var/lib/postgresql/data
   ```

## 4. Preventive Measures

To avoid future database connection issues:

1. Use Docker secrets or environment files for sensitive information.
2. Implement retry logic in your application for database connections.
3. Set up monitoring and alerting for database health and connectivity.
4. Regularly backup your database and test restoration procedures.

## 5. Seeking Further Help

If you're still experiencing issues:

1. Gather all relevant logs and configuration files.
2. Check R2R documentation and community forums.
3. Consider posting a detailed question on the R2R GitHub repository or community channels, providing:
   - Docker Compose file (with sensitive information redacted)
   - R2R and database version information
   - Detailed error messages and logs
   - Steps to reproduce the issue

By following this guide, you should be able to diagnose and resolve most database connection issues in your R2R deployment. Remember to always keep your database and R2R versions compatible and up to date.

---

# Troubleshooting Guide: TOML File Syntax Errors in R2R Configuration

TOML (Tom's Obvious, Minimal Language) is used for R2R configuration files. Syntax errors in these files can prevent R2R from starting or functioning correctly. This guide will help you identify and resolve common TOML syntax issues.

## 1. Common TOML Syntax Errors

### 1.1 Missing or Mismatched Quotes

**Symptom:** Error message mentioning unexpected character or unterminated string.

**Example of incorrect syntax:**
```toml
name = "John's Config
```

**Correct syntax:**
```toml
name = "John's Config"
```

**Solution:** Ensure all string values are properly enclosed in quotes. Use double quotes for strings containing single quotes.

### 1.2 Incorrect Array Syntax

**Symptom:** Error about invalid array literals or unexpected tokens.

**Example of incorrect syntax:**
```toml
fruits = apple, banana, cherry
```

**Correct syntax:**
```toml
fruits = ["apple", "banana", "cherry"]
```

**Solution:** Use square brackets for arrays and separate elements with commas.

### 1.3 Indentation Errors

**Symptom:** Unexpected key error or section not found.

**Example of incorrect syntax:**
```toml
[database]
  host = "localhost"
 port = 5432
```

**Correct syntax:**
```toml
[database]
host = "localhost"
port = 5432
```

**Solution:** TOML doesn't require specific indentation, but be consistent. Align key-value pairs at the same level.

### 1.4 Incorrect Table (Section) Definition

**Symptom:** Invalid table name or unexpected token after table.

**Example of incorrect syntax:**
```toml
[database settings]
```

**Correct syntax:**
```toml
[database.settings]
```

**Solution:** Use dot notation for nested tables instead of spaces.

### 1.5 Duplicate Keys

**Symptom:** Duplicate keys error or unexpected overwrite of values.

**Example of incorrect syntax:**
```toml
[server]
port = 8080
port = 9090
```

**Correct syntax:**
```toml
[server]
port = 8080
```

**Solution:** Ensure each key is unique within its table.

## 2. R2R-Specific TOML Issues

### 2.1 Incorrect LLM Provider Configuration

**Symptom:** R2R fails to start or connect to LLM provider.

**Example of incorrect syntax:**
```toml
[llm_provider]
type = "openai"
api_key = ${OPENAI_API_KEY}
```

**Correct syntax:**
```toml
[llm_provider]
type = "openai"
api_key = "${OPENAI_API_KEY}"
```

**Solution:** Ensure environment variables are properly quoted in the TOML file.

### 2.2 Misconfigurated Database Settings

**Symptom:** R2R cannot connect to the database.

**Example of incorrect syntax:**
```toml
[database]
url = postgres://user:password@localhost:5432/dbname
```

**Correct syntax:**
```toml
[database]
url = "postgres://user:password@localhost:5432/dbname"
```

**Solution:** Enclose the entire database URL in quotes.

## 3. Debugging Steps

1. **Use a TOML Validator:**
   - Online tools like [TOML Lint](https://www.toml-lint.com/) can quickly identify syntax errors.
   - For local validation, use the `toml` Python package:
     ```
     pip install toml
     python -c "import toml; toml.load('your_config.toml')"
     ```

2. **Check R2R Logs:**
   - Look for specific error messages related to configuration loading.
   - Pay attention to line numbers mentioned in error messages.

3. **Incrementally Build Configuration:**
   - Start with a minimal valid configuration and add sections gradually.
   - Test R2R after each addition to isolate the problematic section.

4. **Use Environment Variables Cautiously:**
   - Ensure all environment variables used in the TOML file are properly set.
   - Double-check the syntax for referencing environment variables.

5. **Compare with Example Configurations:**
   - Reference the R2R documentation for correct TOML structure examples.
   - Ensure your configuration matches the expected format for each section.

## 4. Best Practices

1. Use a consistent naming convention for keys (e.g., snake_case).
2. Group related settings under appropriate table headers.
3. Comment your configuration file for clarity.
4. Keep sensitive information (like API keys) in environment variables.
5. Regularly back up your configuration file before making changes.

## 5. Seeking Help

If you're still experiencing issues after following this guide:

1. Check the [R2R documentation](https://r2r-docs.sciphi.ai/) for the most up-to-date configuration guidelines.
2. Search the [R2R GitHub issues](https://github.com/SciPhi-AI/R2R/issues) for similar problems and solutions.
3. If your issue is unique, consider opening a new GitHub issue with your sanitized configuration file and the full error message.

Remember to remove any sensitive information (like API keys or passwords) before sharing your configuration publicly.

By following this guide, you should be able to resolve most TOML syntax errors in your R2R configuration. If problems persist, don't hesitate to seek help from the R2R community or support channels.

---

# Troubleshooting Guide: Local LLM Integration Issues with R2R

When integrating local Language Models (LLMs) with R2R, you may encounter various issues. This guide will help you diagnose and resolve common problems.

## 1. Ollama Connection Issues

### Symptom: R2R can't connect to Ollama

1. **Check Ollama is running:**
   ```bash
   docker ps | grep ollama
   ```
   Ensure the Ollama container is up and running.

2. **Verify Ollama API accessibility:**
   ```bash
   curl http://localhost:11434/api/tags
   ```
   This should return a list of available models.

3. **Check R2R configuration:**
   Ensure the `OLLAMA_API_BASE` environment variable is set correctly in your R2R configuration:
   ```yaml
   OLLAMA_API_BASE: http://ollama:11434
   ```

4. **Network connectivity:**
   Ensure Ollama and R2R containers are on the same Docker network.

### Solution:
- If Ollama isn't running, start it with `docker-compose up -d ollama`
- If API is inaccessible, check Ollama logs: `docker logs ollama`
- Correct the `OLLAMA_API_BASE` if necessary
- Ensure both services are on the `r2r-network` in your Docker Compose file

## 2. Model Loading Issues

### Symptom: Specified model isn't available or fails to load

1. **List available models:**
   ```bash
   curl http://localhost:11434/api/tags
   ```

2. **Attempt to pull the model:**
   ```bash
   docker exec -it ollama ollama pull <model_name>
   ```

3. **Check Ollama logs for pull errors:**
   ```bash
   docker logs ollama
   ```

### Solution:
- If the model isn't listed, pull it using the Ollama CLI
- If pull fails, check internet connectivity and Ollama's GitHub for known issues
- Ensure sufficient disk space for model storage

## 3. Performance Issues

### Symptom: Local LLM responses are slow or timeouts occur

1. **Check system resources:**
   ```bash
   docker stats
   ```
   Look for high CPU or memory usage.

2. **Verify GPU utilization** (if applicable):
   ```bash
   nvidia-smi
   ```

3. **Review Ollama configuration:**
   Check if Ollama is configured to use GPU acceleration.

### Solution:
- Increase resources allocated to the Ollama container
- Enable GPU acceleration if available
- Consider using a smaller or more efficient model

## 4. Inconsistent Responses

### Symptom: LLM responses are inconsistent or unexpected

1. **Verify model version:**
   ```bash
   docker exec -it ollama ollama list
   ```
   Ensure you're using the intended model version.

2. **Check prompt template:**
   Review the prompt template in your R2R configuration for any issues.

3. **Test model directly:**
   ```bash
   docker exec -it ollama ollama run <model_name> "Your test prompt here"
   ```
   Compare direct results with those from R2R.

### Solution:
- Update to the latest model version if necessary
- Adjust prompt templates in R2R configuration
- Ensure consistent tokenization and preprocessing in R2R

## 5. Integration Configuration Issues

### Symptom: R2R doesn't use the local LLM as expected

1. **Review R2R configuration:**
   Check your `r2r.toml` or environment variables to ensure local LLM is properly configured.

2. **Verify LLM provider settings:**
   Ensure the correct provider (e.g., 'ollama') is set in your configuration.

3. **Check R2R logs:**
   Look for any errors or warnings related to LLM initialization.

### Solution:
- Correct configuration settings in `r2r.toml` or environment variables
- Ensure the LLM provider is correctly specified
- Restart R2R after configuration changes

## 6. Memory Management Issues

### Symptom: Out of memory errors or crashes during LLM operations

1. **Monitor memory usage:**
   ```bash
   docker stats ollama
   ```

2. **Check Ollama logs for OOM errors:**
   ```bash
   docker logs ollama | grep "Out of memory"
   ```

3. **Review model specifications:**
   Ensure your hardware meets the minimum requirements for the chosen model.

### Solution:
- Increase memory allocation for the Ollama container
- Use a smaller model if hardware is limited
- Implement request queuing in R2R to manage concurrent LLM calls

## 7. API Compatibility Issues

### Symptom: R2R fails to communicate properly with Ollama

1. **Check Ollama version:**
   ```bash
   docker exec -it ollama ollama --version
   ```

2. **Review R2R documentation:**
   Ensure you're using a compatible version of Ollama for your R2R version.

3. **Test basic API calls:**
   ```bash
   curl -X POST http://localhost:11434/api/generate -d '{"model": "<model_name>", "prompt": "Hello, world!"}'
   ```

### Solution:
- Update Ollama to a compatible version
- Adjust R2R code if using custom integrations
- Check for any middleware or proxy issues affecting API calls

## Getting Further Help

If you're still experiencing issues after trying these troubleshooting steps:

1. Gather relevant logs from both R2R and Ollama
2. Note your system specifications and R2R configuration
3. Check the R2R GitHub issues for similar problems
4. Consider posting a detailed question on the R2R Discord community or GitHub discussions

Remember to provide:
- R2R version (`r2r --version`)
- Ollama version
- Docker and Docker Compose versions
- Host system specifications
- Detailed error messages and logs

By following this guide, you should be able to resolve most common issues with local LLM integration in R2R. If problems persist, don't hesitate to reach out to the R2R community for further assistance.

---

# Troubleshooting Guide: Connection String Errors in R2R Deployments

Connection string errors can occur when R2R is unable to establish a connection with a database or service. This guide will help you diagnose and resolve common connection string issues.

## 1. Identify the Error

First, locate the specific error message in your logs. Common connection string errors include:

- "Unable to connect to [service]"
- "Connection refused"
- "Authentication failed"
- "Invalid connection string"

## 2. Common Issues and Solutions

### 2.1 Incorrect Host or Port

**Symptom:** Error messages mentioning "host not found" or "connection refused"

**Possible Causes:**
- Typo in hostname or IP address
- Wrong port number
- Firewall blocking the connection

**Solutions:**
1. Double-check the hostname/IP and port in your connection string
2. Verify the service is running on the specified port
3. Check firewall rules and ensure the port is open

Example fix for PostgreSQL:
```
# Before
DATABASE_URL=postgres://user:password@wronghost:5432/dbname

# After
DATABASE_URL=postgres://user:password@correcthost:5432/dbname
```

### 2.2 Authentication Failures

**Symptom:** Errors like "authentication failed" or "access denied"

**Possible Causes:**
- Incorrect username or password
- User lacks necessary permissions

**Solutions:**
1. Verify username and password are correct
2. Ensure the user has the required permissions on the database

### 2.3 Invalid Connection String Format

**Symptom:** Errors mentioning "invalid connection string" or specific syntax errors

**Possible Causes:**
- Malformed connection string
- Missing required parameters

**Solutions:**
1. Check the connection string format for the specific service
2. Ensure all required parameters are included

Example fix for a generic connection string:
```
# Before (missing password)
CONNECTION_STRING=Service=MyService;User ID=myuser;Server=myserver

# After
CONNECTION_STRING=Service=MyService;User ID=myuser;Password=mypassword;Server=myserver
```

### 2.4 SSL/TLS Issues

**Symptom:** Errors related to SSL handshake or certificate validation

**Possible Causes:**
- SSL/TLS not properly configured
- Invalid or expired certificates

**Solutions:**
1. Ensure SSL/TLS is correctly set up on both client and server
2. Update expired certificates
3. If testing, you may temporarily disable SSL (not recommended for production)

Example fix for PostgreSQL with SSL:
```
# Before (SSL enforced)
DATABASE_URL=postgres://user:password@host:5432/dbname?sslmode=require

# After (disable SSL for testing only)
DATABASE_URL=postgres://user:password@host:5432/dbname?sslmode=disable
```

### 2.5 Database Not Found

**Symptom:** Errors like "database does not exist" or "unknown database"

**Possible Causes:**
- Typo in database name
- Database hasn't been created

**Solutions:**
1. Verify the database name is correct
2. Ensure the database exists on the server

Example fix:
```
# Before
POSTGRES_DBNAME=wrongdbname

# After
POSTGRES_DBNAME=correctdbname
```

## 3. Environment-Specific Troubleshooting

### 3.1 Docker Environment

If you're using Docker:

1. Check if the service containers are running:
   ```
   docker ps
   ```
2. Inspect the network to ensure services are on the same network:
   ```
   docker network inspect r2r-network
   ```
3. Use Docker's DNS for hostnames (e.g., use `postgres` instead of `localhost` if `postgres` is the service name)

### 3.2 Cloud Environments

For cloud deployments:

1. Verify that the database service is in the same region/zone as your application
2. Check VPC and subnet configurations
3. Ensure necessary firewall rules or security groups are set up correctly

## 4. Debugging Steps

1. **Test the connection independently:**
   Use command-line tools to test the connection outside of R2R:
   - For PostgreSQL: `psql -h <host> -U <username> -d <dbname>`

2. **Check service logs:**
   Examine logs of the service you're trying to connect to for any error messages or access attempts.

3. **Use connection string builders:**
   Many database providers offer online tools to help construct valid connection strings.

## 5. Prevention and Best Practices

1. Use environment variables for sensitive information in connection strings
2. Implement connection pooling to manage connections efficiently
3. Set up proper logging to quickly identify connection issues
4. Use secret management services for storing and retrieving connection credentials securely

## 6. Seeking Further Help

If you're still encountering issues:

1. Check R2R documentation for specific connection string requirements
2. Consult the documentation of the specific database or service you're connecting to
3. Search or ask for help in R2R community forums or support channels
4. Provide detailed error messages and environment information when seeking help

Remember to never share actual passwords or sensitive information when asking for help. Always use placeholders in examples.

By following this guide, you should be able to resolve most connection string errors in your R2R deployment. If problems persist, don't hesitate to seek help from the R2R community or support team.

---

# R2R Local System Installation

This guide will walk you through installing and running R2R on your local system without using Docker. This method allows for more customization and control over individual components.


## Prerequisites

Before starting, ensure you have the following installed and/or available in the cloud:
- Python 3.12 or higher
- pip (Python package manager)
- Git
- Postgres + pgvector

## Install the R2R CLI and extra dependencies

First, install the R2R CLI with the additional `light` dependencies:

```bash
pip install 'r2r[core,ingestion-bundle]'
```

The `core` and `ingestion-bundle` dependencies, combined with a Postgres database, provide the necessary components to deploy a user-facing R2R application into production.

If you need advanced features like orchestration or parsing with `Unstructured.io` then refer to the <a href="/documentation/installation/full/docker"> full installation </a>.

## Environment Setup

R2R requires connections to various services. Set up the following environment variables based on your needs:

<AccordionGroup>
  <Accordion title="Cloud LLM Providers" icon="language">
    Note, cloud providers are optional as R2R can be run entirely locally.
     ```bash
      # Set cloud LLM settings
      export OPENAI_API_KEY=sk-...
      # export ANTHROPIC_API_KEY=...
      # ...
    ```
  </Accordion>
  <Accordion title="Postgres+pgvector" icon="database">
     With R2R you can connect to your own instance of Postgres+pgvector or a remote cloud instance.
     ```bash
      # Set Postgres+pgvector settings
      export R2R_POSTGRES_USER=$YOUR_POSTGRES_USER
      export R2R_POSTGRES_PASSWORD=$YOUR_POSTGRES_PASSWORD
      export R2R_POSTGRES_HOST=$YOUR_POSTGRES_HOST
      export R2R_POSTGRES_PORT=$YOUR_POSTGRES_PORT
      export R2R_POSTGRES_DBNAME=$YOUR_POSTGRES_DBNAME
      export R2R_PROJECT_NAME=$YOUR_PROJECT_NAME # see note below
    ```
    <Note>
    The `R2R_PROJECT_NAME` environment variable defines the tables within your Postgres database where the selected R2R project resides. If the required tables for R2R do not exist then they will be created by R2R during initialization.
    </Note>
    If you are unfamiliar with Postgres then <a href="https://supabase.com/docs"> Supabase's free cloud offering </a> is a good place to start.
  </Accordion>
</AccordionGroup>


## Running R2R

After setting up your environment, you can start R2R using the following command:

```bash
r2r serve
```

For local LLM usage:

```bash
r2r serve --config-name=local_llm
```

## Python Development Mode

For those looking to develop R2R locally:

1. Install Poetry: Follow instructions on the [official Poetry website](https://python-poetry.org/docs/#installation).

2. Clone and install dependencies:
   ```bash
   git clone git@github.com:SciPhi-AI/R2R.git
   cd R2R/py
   poetry install -E "core ingestion-bundle"
   ```

3. Setup environment:
   Follow the steps listed in the Environment Setup section above. Additionally, you may introduce a local .env file to make development easier, and you can customize your local `r2r.toml` to suit your specific needs.

4. Start your server:
  ```bash
  poetry run r2r serve
  ```

## Next Steps

After successfully installing R2R:

1. **Verify Installation**: Ensure all components are running correctly by accessing the R2R API at http://localhost:7272/v2/health.

2. **Quick Start**: Follow our [R2R Quickstart Guide](/documentation/quickstart) to set up your first RAG application.

3. **In-Depth Tutorial**: For a more comprehensive understanding, work through our [R2R Walkthrough](/cookbooks/walkthrough).

4. **Customize Your Setup**: Configure R2R components with the [Configuration Guide](/documentation/configuration).

If you encounter any issues during installation or setup, please use our [Discord community](https://discord.gg/p6KqD2kjtB) or [GitHub repository](https://github.com/SciPhi-AI/R2R) to seek assistance.

---


# Deploying R2R

R2R (RAG to Riches) is designed to be flexible and scalable, allowing deployment in various environments. This guide provides an overview of deployment options and resource recommendations to help you get started with R2R in a production setting.

## Deployment Options

1. **Local Docker or Local Build**: Ideal for development and testing. [Start here](/documentation/installation/overview).
2. **Single Cloud Instance**: Recommended for most small to medium-sized organizations.
3. **Container Orchestration** (Docker Swarm): Suitable for larger organizations or those requiring more granular resource control

## Resource Recommendations

When running R2R, we recommend:
- At least 4 vCPU cores
- 8+GB of RAM (16GB preferred)
- 50gb + 4x raw data size (_size of data to be ingested after converting to TXT_) of disk space

## Deployment Guides

For detailed, step-by-step instructions on deploying R2R in various environments, please refer to our specific deployment guides:

- [Local Deployment](/documentation/installation/overview)
- [Azure Deployment](/documentation/deployment/azure)
- [SciPhi Cloud](/documentation/deployment/sciphi/)

Choose the guide that best fits your infrastructure and scaling needs. Each guide provides specific instructions for setting up R2R in that environment, including necessary configurations and best practices.

By following these deployment recommendations and configuration guides, you'll be well on your way to leveraging R2R's powerful RAG capabilities in your production environment.

---

# Troubleshooting Guide: Incompatible Configuration Settings in R2R

When working with R2R (RAG to Riches), you may encounter issues related to incompatible configuration settings. This guide will help you identify and resolve common configuration conflicts.

## 1. Identifying Configuration Issues

Configuration issues often manifest as error messages during startup or unexpected behavior during runtime. Look for error messages related to configuration in your logs or console output.

## 2. Common Incompatible Configurations

### 2.1 Database Conflicts

**Issue:** Conflicting database settings between different components.

**Symptoms:**
- Error messages mentioning database connection failures
- Inconsistent data retrieval or storage

**Resolution:**
1. Check your `r2r.toml` file for database settings.
2. Ensure all components (R2R, Hatchet, etc.) use the same database credentials.
3. Verify that the database URL, port, and name are consistent across all configurations.

Example correction:
```toml
[database]
url = "postgres://user:password@localhost:5432/r2r_db"
```

### 2.2 LLM Provider Conflicts

**Issue:** Multiple or incompatible LLM provider settings.

**Symptoms:**
- Errors about undefined LLM providers
- Unexpected LLM behavior or responses

**Resolution:**
1. Review your LLM provider settings in the configuration.
2. Ensure only one primary LLM provider is active.
3. Check that API keys and endpoints are correctly set for the chosen provider.

Example correction:
```toml
[llm_providers]
primary = "openai"
[llm_providers.openai]
api_key = "your-openai-api-key"
```

### 2.3 Vector Store Misconfigurations

**Issue:** Incompatible vector store settings.

**Symptoms:**
- Errors related to vector operations or storage
- Failure to store or retrieve embeddings

**Resolution:**
1. Verify that your chosen vector store (e.g., pgvector) is properly configured.
2. Ensure the vector store settings match your database configuration.
3. Check for any conflicting dimension settings in your embeddings configuration.

Example correction:
```toml
[vector_store]
type = "pgvector"
dimension = 1536  # Must match your embedding model's output dimension
```

### 2.4 Hatchet Orchestration Conflicts

**Issue:** Misconfigured Hatchet settings leading to orchestration failures.

**Symptoms:**
- Errors in task queuing or execution
- Hatchet service failing to start or communicate

**Resolution:**
1. Check Hatchet-related environment variables and configuration.
2. Ensure Hatchet API key and endpoint are correctly set.
3. Verify RabbitMQ settings if used for task queuing.

Example correction:
```toml
[orchestration]
type = "hatchet"
api_key = "your-hatchet-api-key"
endpoint = "http://localhost:7077"
```

### 2.5 File Path and Permission Issues

**Issue:** Incorrect file paths or insufficient permissions.

**Symptoms:**
- Errors about missing files or directories
- Permission denied errors when accessing resources

**Resolution:**
1. Verify all file paths in your configuration are correct and accessible.
2. Check permissions on directories used by R2R, especially in Docker environments.
3. Ensure consistency between host and container paths if using Docker.

Example correction:
```toml
[file_storage]
base_path = "/app/data"  # Ensure this path exists and is writable
```

## 3. Configuration Validation Steps

1. **Use R2R's built-in validation:**
   Run `r2r validate-config` to check for basic configuration errors.

2. **Environment variable check:**
   Ensure all required environment variables are set and not conflicting with configuration file settings.

3. **Docker configuration:**
   If using Docker, verify that your `docker-compose.yml` file correctly maps volumes and sets environment variables.

4. **Component version compatibility:**
   Ensure all components (R2R, database, vector store, LLM providers) are using compatible versions.

## 4. Advanced Troubleshooting

### 4.1 Configuration Debugging Mode

Enable debug logging to get more detailed information about configuration loading:

```toml
[logging]
level = "DEBUG"
```

### 4.2 Component Isolation

If you're unsure which component is causing the issue, try running components in isolation:

1. Start only the database and vector store.
2. Add the R2R core service.
3. Gradually add other services (Hatchet, LLM providers) one by one.

This approach can help identify which specific component or interaction is causing the incompatibility.

### 4.3 Configuration Diff Tool

Use a diff tool to compare your current configuration with a known working configuration or the default template. This can help spot unintended changes or typos.

## 5. Seeking Further Assistance

If you're still experiencing issues after trying these solutions:

1. Check the R2R documentation for any recent changes or known issues.
2. Search the R2R GitHub issues for similar problems and solutions.
3. Prepare a detailed description of your issue, including:
   - Your full R2R configuration (with sensitive information redacted)
   - Error messages and logs
   - Steps to reproduce the issue
4. Reach out to the R2R community on Discord or file a GitHub issue with the prepared information.

Remember, when sharing configurations or logs, always remove sensitive information like API keys or passwords.

By following this guide, you should be able to identify and resolve most incompatible configuration settings in your R2R setup. If problems persist, don't hesitate to seek help from the R2R community or support channels.

---

# R2R Troubleshooting Guide: Insufficient Instance Resources

When deploying R2R, you may encounter issues related to insufficient instance resources. This guide will help you identify, troubleshoot, and resolve these problems.

## Symptoms of Insufficient Resources

1. Containers fail to start or crash frequently
2. Slow response times or timeouts
3. Out of memory errors
4. High CPU usage alerts
5. Disk space warnings

## Diagnosing Resource Issues

### 1. Check Docker Resource Usage

```bash
docker stats
```

This command shows a live stream of container resource usage statistics.

### 2. Check Host System Resources

```bash
top
free -h
df -h
```

These commands show CPU, memory, and disk usage respectively.

### 3. Review Container Logs

```bash
docker logs <container_name>
```

Look for error messages related to resource constraints.

## Common Resource-Related Issues and Solutions

### 1. Insufficient Memory

**Symptom:** Container exits with out of memory error or host system shows high memory usage.

**Solution:**
- Increase Docker memory limit:
  ```bash
  docker run --memory=4g ...
  ```
- For Docker Desktop, increase memory allocation in settings.
- For cloud instances, upgrade to a larger instance type.

### 2. CPU Constraints

**Symptom:** High CPU usage, slow response times.

**Solution:**
- Limit CPU usage for non-critical containers:
  ```bash
  docker run --cpus=0.5 ...
  ```
- Upgrade to an instance with more CPU cores.

### 3. Disk Space Issues

**Symptom:** "No space left on device" errors.

**Solution:**
- Clean up unused Docker resources:
  ```bash
  docker system prune
  ```
- Increase disk space allocation for Docker (in Docker Desktop settings or cloud instance).
- Use volume mounts for large data directories.

### 4. Network Resource Constraints

**Symptom:** Network-related timeouts or slow connections.

**Solution:**
- Check and increase network resource limits:
  ```bash
  docker network inspect bridge
  ```
- In cloud environments, ensure proper network configuration and bandwidth allocation.

## R2R-Specific Resource Considerations

### 1. Postgres with pgvector

Vector operations can be CPU-intensive. Ensure your instance has sufficient CPU resources, or consider using a managed database service.

### 2. Ollama for Local LLM

Local LLM inference can be very resource-intensive. Ensure your instance has:
- At least 8GB of RAM (16GB+ recommended)
- Sufficient disk space for model storage
- A capable CPU or GPU for inference

### 3. Hatchet Engine

The Hatchet workflow engine may require significant resources depending on your workload. Monitor its resource usage and adjust as necessary.

## Optimizing Resource Usage

1. **Use Resource Limits:** Set appropriate CPU and memory limits for each container.
2. **Optimize Configurations:** Fine-tune application configs (e.g., Postgres work_mem).
3. **Scale Horizontally:** Consider splitting services across multiple smaller instances instead of one large instance.
4. **Use Managed Services:** For production, consider using managed services for databases and other resource-intensive components.
5. **Monitor and Alert:** Set up monitoring and alerting for resource usage to catch issues early.

## When to Upgrade Resources

Consider upgrading your instance or allocating more resources when:
1. You consistently see high resource utilization (>80% CPU, >90% memory).
2. Response times are consistently slow and not improving with optimization.
3. You're frequently hitting resource limits and it's affecting system stability.

## Seeking Further Help

If you've tried these solutions and still face resource issues:
1. Review the R2R documentation for specific resource recommendations.
2. Check the R2R GitHub issues for similar problems and solutions.
3. Reach out to the R2R community on Discord or GitHub for advice.
4. Consider engaging with R2R maintainers or professional services for complex deployments.

Remember to always test in a non-production environment before making significant changes to resource allocations or instance types in a production setting.

---

# R2R Troubleshooting Guide: Dependency Conflicts

Dependency conflicts can occur when different components of the R2R system require incompatible versions of the same library or when there are conflicts between system-level dependencies. This guide will help you identify and resolve common dependency issues.

## 1. Identifying Dependency Conflicts

### Symptoms:
- Error messages mentioning version conflicts
- Unexpected behavior in specific components
- Installation or startup failures

### Common Error Messages:
- "ImportError: cannot import name X from Y"
- "AttributeError: module X has no attribute Y"
- "ModuleNotFoundError: No module named X"

## 2. Python Package Conflicts

### 2.1 Diagnosing the Issue

1. Check your Python environment:
   ```bash
   python --version
   pip list
   ```

2. Look for conflicting versions in the pip list output.

3. Use `pip check` to identify dependency conflicts:
   ```bash
   pip check
   ```

### 2.2 Resolving Python Package Conflicts

1. **Update R2R and its dependencies:**
   ```bash
   pip install --upgrade r2r[core]
   ```

2. **Use a virtual environment:**
   ```bash
   python -m venv r2r_env
   source r2r_env/bin/activate  # On Windows, use r2r_env\Scripts\activate
   pip install r2r[core]
   ```

3. **Manually resolve conflicts:**
   - Identify the conflicting packages
   - Upgrade or downgrade specific packages:
     ```bash
     pip install package_name==specific_version
     ```

4. **Use `pip-compile` for deterministic builds:**
   ```bash
   pip install pip-tools
   pip-compile requirements.in
   pip-sync requirements.txt
   ```

## 3. System-level Dependency Conflicts

### 3.1 Diagnosing System Conflicts

1. Check system library versions:
   ```bash
   ldd --version
   ldconfig -p | grep library_name
   ```

2. Look for error messages related to shared libraries:
   - "error while loading shared libraries"
   - "symbol lookup error"

### 3.2 Resolving System-level Conflicts

1. **Update system packages:**
   ```bash
   sudo apt update && sudo apt upgrade  # For Ubuntu/Debian
   sudo yum update  # For CentOS/RHEL
   ```

2. **Install missing libraries:**
   ```bash
   sudo apt install library_name  # For Ubuntu/Debian
   sudo yum install library_name  # For CentOS/RHEL
   ```

3. **Use container isolation:**
   - Consider using Docker to isolate the R2R environment from the host system.

## 4. Docker-specific Dependency Issues

### 4.1 Diagnosing Docker Issues

1. Check Docker image versions:
   ```bash
   docker images
   ```

2. Inspect Docker logs:
   ```bash
   docker logs container_name
   ```

### 4.2 Resolving Docker Dependency Conflicts

1. **Update Docker images:**
   ```bash
   docker pull ragtoriches/prod:main-unstructured
   ```

2. **Rebuild with no cache:**
   ```bash
   docker-compose build --no-cache
   ```

3. **Check Docker Compose file:**
   - Ensure all services are using compatible versions
   - Update service versions if necessary


## 6. Advanced Troubleshooting

### 6.1 Use Dependency Visualization Tools

1. Install `pipdeptree`:
   ```bash
   pip install pipdeptree
   ```

2. Visualize dependencies:
   ```bash
   pipdeptree -p r2r
   ```

### 6.2 Analyze Startup Sequences

1. Use verbose logging:
   ```bash
   R2R_LOG_LEVEL=DEBUG r2r serve
   ```

2. Analyze logs for import errors or version conflicts

### 6.3 Temporary Workarounds

1. **Pin problematic dependencies:**
   - Create a `constraints.txt` file with specific versions
   - Install with constraints:
     ```bash
     pip install -c constraints.txt r2r[core]
     ```

2. **Use compatibility mode:**
   - If available, run R2R with a compatibility flag to use older versions of certain components

## 7. Seeking Further Help

If you've tried these steps and still encounter issues:

1. Check the [R2R GitHub Issues](https://github.com/SciPhi-AI/R2R/issues) for similar problems and solutions
2. Join the [R2R Discord community](https://discord.gg/p6KqD2kjtB) for real-time support
3. Open a new issue on GitHub with:
   - Detailed description of the problem
   - Steps to reproduce
   - Environment details (OS, Python version, pip list output)
   - Relevant log snippets

Remember, when dealing with dependency conflicts, it's crucial to document your changes and test thoroughly after each modification to ensure you haven't introduced new issues while solving existing ones.

---

# Troubleshooting Guide: Unexpected R2R API Responses

When working with the R2R API, you might encounter responses that don't match your expectations. This guide will help you diagnose and resolve common issues related to unexpected API behavior.

## 1. Verify API Endpoint and Request

First, ensure you're using the correct API endpoint and making the proper request.

### 1.1 Check API URL

- Confirm you're using the correct base URL for your R2R instance.
- Verify the endpoint path is correct (e.g., `/v2/search` for search requests).

### 1.2 Review Request Method

- Ensure you're using the appropriate HTTP method (GET, POST, PUT, DELETE) for the endpoint.

### 1.3 Validate Request Headers

- Check that you've included all required headers, especially:
  - `Content-Type: application/json` for POST requests
  - Authorization header if required

### 1.4 Inspect Request Body

- For POST requests, verify the JSON structure of your request body.
- Ensure all required fields are present and correctly formatted.

## 2. Common Unexpected Responses

### 2.1 Empty Results

**Symptom:** API returns an empty list or no results when you expect data.

**Possible Causes:**
- No matching data in the database
- Incorrect search parameters
- Issues with data ingestion

**Troubleshooting Steps:**
1. Verify your search query or filter parameters.
2. Check if the data you're expecting has been successfully ingested.
3. Try a broader search to see if any results are returned.

### 2.2 Incorrect Data Format

**Symptom:** API returns data in an unexpected format or structure.

**Possible Causes:**
- Changes in API version
- Misconfiguration in R2R settings

**Troubleshooting Steps:**
1. Check the API documentation for the correct response format.
2. Verify you're using the latest version of the API.
3. Review your R2R configuration settings.

### 2.3 Unexpected Error Responses

**Symptom:** API returns error codes or messages you don't expect.

**Possible Causes:**
- Server-side issues
- Rate limiting
- Authorization problems

**Troubleshooting Steps:**
1. Check the error message for specific details.
2. Verify your API key or authentication token.
3. Ensure you're not exceeding rate limits.
4. Check R2R server logs for more information.

## 3. Debugging Techniques

### 3.1 Use Verbose Logging

Enable verbose logging in your API requests to get more detailed information:

```python
import requests

response = requests.get('http://your-r2r-api-url/v2/endpoint',
                        params={'verbose': True})
print(response.json())
```

### 3.2 Check R2R Server Logs

Access the R2R server logs to look for any error messages or warnings:

```bash
docker logs r2r-container-name
```

### 3.3 Use API Testing Tools

Utilize tools like Postman or cURL to isolate API issues:

```bash
curl -X GET "http://your-r2r-api-url/v2/health" -H "accept: application/json"
```

## 4. Common Issues and Solutions

### 4.1 Inconsistent Search Results

**Issue:** Search results vary unexpectedly between requests.

**Solution:**
- Check if your data is being updated concurrently.
- Verify the consistency of your vector database (Postgres+pgvector).
- Ensure your search parameters are consistent.

### 4.2 Slow API Response Times

**Issue:** API requests take longer than expected to return results.

**Solution:**
- Check the size of your dataset and consider optimization.
- Verify the performance of your database queries.
- Consider scaling your R2R deployment if dealing with large datasets.

### 4.3 Unexpected Data Relationships

**Issue:** API returns relationships between data that don't match expectations.

**Solution:**
- Review your knowledge graph structure.
- Check the logic in your data ingestion and relationship creation processes.
- Verify that your query is correctly traversing the graph.

## 5. When to Seek Further Help

If you've gone through this guide and are still experiencing issues:

1. Gather all relevant information:
   - API request details (endpoint, method, headers, body)
   - Full response (including headers and body)
   - R2R server logs
   - Any error messages or unexpected behavior

2. Check the R2R documentation and community forums for similar issues.

3. If the problem persists, consider reaching out to the R2R community or support channels:
   - Post a detailed question on the R2R GitHub repository
   - Join the R2R Discord community for real-time assistance
   - Contact R2R support if you have a support agreement

Remember to always provide as much context as possible when seeking help, including your R2R version, deployment method, and steps to reproduce the issue.

By following this guide, you should be able to diagnose and resolve most unexpected API response issues in your R2R deployment. If you encounter persistent problems, don't hesitate to seek help from the R2R community or support channels.

---

# R2R Troubleshooting Guide: Insufficient System Resources

When running R2R in Docker, you may encounter issues related to insufficient system resources. This guide will help you identify, diagnose, and resolve these problems.

## Symptoms of Insufficient Resources

1. Docker containers fail to start or crash unexpectedly
2. Slow performance or unresponsiveness of R2R services
3. Error messages related to memory, CPU, or disk space
4. Unexpected termination of processes within containers

## Diagnosing Resource Issues

### 1. Check Docker Resource Usage

Use the following command to view resource usage for all containers:

```bash
docker stats
```

Look for containers using high percentages of CPU or memory.

### 2. Check Host System Resources

On Linux:
```bash
top
free -h
df -h
```

On macOS:
```bash
top
vm_stat
df -h
```

On Windows:
```
Task Manager > Performance tab
```

### 3. Review Docker Logs

Check logs for specific containers:

```bash
docker logs <container_name>
```

Look for out-of-memory errors or other resource-related messages.

## Common Issues and Solutions

### 1. Insufficient Memory

**Symptom:** Containers crash with out-of-memory errors.

**Solutions:**
a. Increase Docker's memory limit:
   - On Docker Desktop, go to Settings > Resources > Advanced and increase memory allocation.

b. Optimize memory usage in R2R configuration:
   - Modify Postgres memory settings in `postgresql.conf`

c. Add or increase swap space on your host system.

### 2. CPU Constraints

**Symptom:** Slow performance or high CPU usage warnings.

**Solutions:**
a. Increase Docker's CPU limit:
   - On Docker Desktop, go to Settings > Resources > Advanced and increase CPU allocation.

b. Optimize R2R and dependent services:
   - Adjust thread pool sizes in configurations
   - Consider using CPU-specific Docker options like `--cpus` or `--cpu-shares`

### 3. Disk Space Issues

**Symptom:** "No space left on device" errors or containers failing to write data.

**Solutions:**
a. Clean up Docker resources:
```bash
docker system prune
```

b. Increase disk space allocation for Docker:
   - On Docker Desktop, go to Settings > Resources > Advanced and increase disk image size.

c. Monitor and manage log file sizes:
   - Implement log rotation for services like Postgres
   - Use Docker's logging options to limit log file sizes:
     ```yaml
     logging:
       options:
         max-size: "10m"
         max-file: "3"
     ```

### 4. Network Resource Constraints

**Symptom:** Connection timeouts or network-related errors.

**Solutions:**
a. Check and increase ulimit for open files:
```bash
ulimit -n 65535
```

b. Optimize Docker network settings:
   - Use `host` network mode for better performance (if security allows)
   - Adjust MTU settings if necessary

## Preventive Measures

1. **Regular Monitoring:** Set up monitoring tools like Prometheus and Grafana to track resource usage over time.

2. **Resource Limits:** Set appropriate resource limits in your Docker Compose file:

   ```yaml
   services:
     r2r:
       deploy:
         resources:
           limits:
             cpus: '0.50'
             memory: 512M
           reservations:
             cpus: '0.25'
             memory: 256M
   ```

3. **Performance Testing:** Conduct regular performance tests to identify resource bottlenecks before they become critical.

4. **Scaling Strategy:** Develop a strategy for horizontal scaling of R2R services as your usage grows.

## When to Upgrade Hardware

Consider upgrading your hardware or moving to a more powerful cloud instance if:

1. You consistently hit resource limits despite optimization efforts.
2. Performance degrades as your data or user base grows.
3. You need to scale beyond what your current hardware can support.

## Seeking Further Help

If you've tried these solutions and still face resource issues:

1. Gather detailed logs and resource usage data.
2. Check the R2R documentation for specific resource recommendations.
3. Consult the R2R community forums or support channels.
4. Consider reaching out to a DevOps or Docker specialist for advanced troubleshooting.

Remember, optimal resource allocation often requires iterative testing and adjustment based on your specific use case and workload.

---

# R2R Troubleshooting Guide: Connection Timeouts

Connection timeouts can occur at various points in an R2R deployment, affecting different components and services. This guide will help you identify, diagnose, and resolve connection timeout issues.

## 1. Identifying Connection Timeouts

Connection timeouts typically manifest as:
- Slow or unresponsive API calls
- Error messages mentioning "timeout" or "connection refused"
- Services failing to start or becoming unhealthy
- Incomplete data processing or retrieval

## 2. Common Causes of Connection Timeouts

1. Network issues
2. Misconfigured firewall rules
3. Overloaded services
4. Incorrect connection settings
5. DNS resolution problems
6. Service dependencies not ready

## 3. Diagnosing Connection Timeouts

### 3.1 Check Service Health

```bash
docker-compose ps
```

Look for services in an unhealthy or exit state.

### 3.2 Inspect Service Logs

```bash
docker-compose logs <service_name>
```

Search for timeout-related error messages.

### 3.3 Network Connectivity Test

Test network connectivity between services:

```bash
docker-compose exec <service_name> ping <target_service>
```

### 3.4 Check Resource Usage

Monitor CPU, memory, and disk usage:

```bash
docker stats
```

## 4. Resolving Common Connection Timeout Issues

### 4.1 R2R API Timeouts

If the R2R API is timing out:

1. Check the R2R service logs:
   ```bash
   docker-compose logs r2r
   ```

2. Verify the R2R service is healthy:
   ```bash
   docker-compose exec r2r curl -f http://localhost:7272/v2/health
   ```

3. Increase the timeout settings in your R2R configuration file.

### 4.2 Database Connection Timeouts

For Postgres connection issues:

1. Verify database service is running:
   ```bash
   docker-compose ps postgres
   ```

2. Check database logs:
   ```bash
   docker-compose logs postgres
   ```

3. Ensure correct connection strings in R2R configuration.

4. Increase connection pool size or timeout settings.

### 4.3 Hatchet Engine Timeouts

If Hatchet Engine is experiencing timeouts:

1. Check Hatchet Engine logs:
   ```bash
   docker-compose logs hatchet-engine
   ```

2. Verify RabbitMQ is running and accessible:
   ```bash
   docker-compose exec hatchet-rabbitmq rabbitmqctl status
   ```

3. Increase Hatchet client timeout settings in R2R configuration.

### 4.4 Ollama LLM Timeouts

For timeouts related to Ollama:

1. Check Ollama service status:
   ```bash
   docker-compose ps ollama
   ```

2. Verify Ollama logs:
   ```bash
   docker-compose logs ollama
   ```

3. Ensure Ollama models are properly loaded:
   ```bash
   docker-compose exec ollama ollama list
   ```

4. Increase LLM timeout settings in R2R configuration.

## 5. Advanced Troubleshooting

### 5.1 Network Diagnostics

Use network diagnostic tools within containers:

```bash
docker-compose exec r2r sh -c "apt-get update && apt-get install -y iputils-ping net-tools"
docker-compose exec r2r netstat -tuln
```

### 5.2 DNS Resolution

Check DNS resolution within containers:

```bash
docker-compose exec r2r nslookup postgres
```

### 5.3 Firewall Rules

Verify firewall settings on the host machine:

```bash
sudo iptables -L
```

Ensure necessary ports are open for inter-container communication.

### 5.4 Docker Network Inspection

Inspect the Docker network:

```bash
docker network inspect r2r-network
```

Verify all services are properly connected to the network.

## 6. Preventive Measures

1. Implement robust health checks for all services.
2. Use appropriate timeout and retry mechanisms in your application code.
3. Monitor system resources and scale services as needed.
4. Regularly update and maintain all components of your R2R deployment.
5. Implement logging and monitoring solutions for early detection of issues.

## 7. Seeking Further Assistance

If connection timeout issues persist:

1. Collect comprehensive logs from all services.
2. Document the steps you've taken to troubleshoot.
3. Check the R2R documentation and community forums for similar issues.
4. Consider reaching out to the R2R support channels or community for specialized assistance.

Remember to provide detailed information about your deployment environment, configuration settings, and the specific timeout scenarios you're encountering when seeking help.

By following this guide, you should be able to diagnose and resolve most connection timeout issues in your R2R deployment. If problems persist, don't hesitate to seek additional support from the R2R community or professional services.

---

# R2R Troubleshooting Guide: Missing Environment Variables

When deploying R2R, missing environment variables can cause containers to fail to start or lead to unexpected behavior. This guide will help you identify and resolve issues related to missing environment variables.

## 1. Identifying the Problem

Signs that you might be dealing with missing environment variables:

- Containers exit immediately after starting
- Error messages in logs mentioning undefined or null values
- Specific features or integrations not working as expected

## 2. Common Missing Environment Variables

Here are some critical environment variables for R2R:

- Database credentials (e.g., `R2R_POSTGRES_USER`, `R2R_POSTGRES_PASSWORD`)
- API keys (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)
- Configuration settings (e.g., `R2R_CONFIG_NAME`, `R2R_CONFIG_PATH`)

## 3. Checking for Missing Variables

### 3.1 Review Docker Compose File

1. Open your `docker-compose.yml` file.
2. Look for the `environment` section under the `r2r` service.
3. Ensure all required variables are listed.

Example:
```yaml
services:
  r2r:
    environment:
      - R2R_POSTGRES_USER=${R2R_POSTGRES_USER:-postgres}
      - OPENAI_API_KEY=${OPENAI_API_KEY:-}
      - R2R_CONFIG_NAME=${R2R_CONFIG_NAME:-}
      # ... other variables
```

### 3.2 Check .env File

1. Ensure you have a `.env` file in the same directory as your `docker-compose.yml`.
2. Verify that all variables used in `docker-compose.yml` are defined in `.env`.

Example `.env` file:
```
R2R_POSTGRES_USER=myuser
OPENAI_API_KEY=sk-...
R2R_CONFIG_NAME=default
```

### 3.3 Verify Environment in Running Container

If the container starts but behaves unexpectedly:

1. Access the container's shell:
   ```
   docker exec -it r2r-container-name /bin/bash
   ```
2. Check environment variables:
   ```
   env | grep VARIABLE_NAME
   ```

## 4. Resolving Missing Variables

### 4.1 Update .env File

1. Add any missing variables to your `.env` file.
2. Ensure values are correct and properly formatted.

### 4.2 Use Default Values

In `docker-compose.yml`, provide default values for non-sensitive variables:

```yaml
environment:
  - VARIABLE_NAME=${VARIABLE_NAME:-default_value}
```

### 4.3 Inject Variables at Runtime

For sensitive data, inject variables when running the container:

```bash
docker run -e SENSITIVE_VAR=value ...
```

Or with docker-compose:

```bash
SENSITIVE_VAR=value docker-compose up
```

### 4.4 Use Docker Secrets

For enhanced security, consider using Docker secrets for sensitive data:

1. Create a secret:
   ```
   echo "mysecretvalue" | docker secret create my_secret -
   ```

2. Use in `docker-compose.yml`:
   ```yaml
   secrets:
     - my_secret
   ```

## 5. Specific R2R Environment Variables

Ensure these key R2R variables are set:

- `R2R_CONFIG_NAME` or `R2R_CONFIG_PATH`: Specifies which configuration to use.
- `R2R_POSTGRES_*`: Database connection details.
- `OPENAI_API_KEY`: If using OpenAI services.
- `ANTHROPIC_API_KEY`: If using Anthropic models.
- `OLLAMA_API_BASE`: For local LLM integration.
- `HATCHET_CLIENT_TOKEN`: For Hatchet integration.

## 6. Debugging Tips

- Use `docker-compose config` to see the final composed configuration with resolved variables.
- Temporarily echo sensitive variables in your container's entrypoint script for debugging (remove in production).
- Check for typos in variable names both in `docker-compose.yml` and `.env` files.

## 7. Best Practices

- Use version control for your `docker-compose.yml`, but not for `.env` files containing secrets.
- Consider using a secret management service for production deployments.
- Document all required environment variables in your project's README.
- Use CI/CD pipelines to validate the presence of required variables before deployment.

By following this guide, you should be able to identify and resolve issues related to missing environment variables in your R2R deployment. Remember to always handle sensitive data securely and never commit secrets to version control.

---

# R2R Troubleshooting Guide: Slow Query Responses

If you're experiencing slow query responses in your R2R deployment, this guide will help you identify and resolve common causes of performance issues.

## 1. Identify the Bottleneck

Before diving into specific solutions, it's crucial to identify where the slowdown is occurring:

- Is it specific to certain types of queries?
- Is it affecting all queries or only queries to a particular data source?
- Is the slowdown consistent or intermittent?

## 2. Check Database Performance

### 2.1 Postgres

1. **Monitor query execution time:**
   ```sql
   SELECT query, calls, total_time, mean_time
   FROM pg_stat_statements
   ORDER BY mean_time DESC
   LIMIT 10;
   ```

2. **Check for missing indexes:**
   ```sql
   SELECT relname, seq_scan, idx_scan
   FROM pg_stat_user_tables
   WHERE seq_scan > idx_scan
   ORDER BY seq_scan DESC;
   ```

3. **Analyze and vacuum the database:**
   ```sql
   ANALYZE;
   VACUUM ANALYZE;
   ```

## 3. Optimize Vector Search

1. **Check vector index:**
   Ensure your vector index is properly created and optimized.

2. **Adjust search parameters:**
   Experiment with different `ef_search` values to balance speed and accuracy.

3. **Monitor vector dimension and dataset size:**
   Large vector dimensions or dataset sizes can slow down searches.

## 4. LLM Integration Issues

1. **Check LLM response times:**
   Monitor the time taken by the LLM to generate responses.

2. **Verify API rate limits:**
   Ensure you're not hitting rate limits for cloud-based LLMs.

3. **For local LLMs (e.g., Ollama):**
   - Check resource utilization (CPU, GPU, memory)
   - Consider using a more efficient model or quantized version

## 5. Network Latency

1. **Check network latency between components:**
   ```bash
   ping <component_host>
   ```

2. **Use tools like `traceroute` to identify network bottlenecks:**
   ```bash
   traceroute <component_host>
   ```

3. **If using cloud services, ensure all components are in the same region.**

## 6. Resource Constraints

1. **Monitor system resources:**
   ```bash
   top
   htop  # If available
   ```

2. **Check Docker resource allocation:**
   ```bash
   docker stats
   ```

3. **Increase resources if necessary:**
   - Adjust Docker resource limits
   - Scale up cloud instances
   - Add more nodes to your cluster

## 7. Caching

1. **Implement or optimize caching strategies:**
   - Use Redis or Memcached for frequently accessed data
   - Implement application-level caching

2. **Check cache hit rates:**
   Monitor your caching system's performance metrics.

## 8. Query Optimization

1. **Review and optimize complex queries:**
   - Break down complex queries into simpler ones
   - Use appropriate JOIN types in SQL queries

2. **Use query parameterization:**
   Avoid string concatenation in queries to leverage query plan caching.

## 9. Hatchet Workflow Optimization

1. **Review workflow designs:**
   Ensure workflows are optimized and not causing unnecessary delays.

2. **Monitor Hatchet logs:**
   Check for any warnings or errors that might indicate performance issues.

## 10. Logging and Monitoring

1. **Implement comprehensive logging:**
   Use logging to identify slow operations and bottlenecks.

2. **Set up monitoring and alerting:**
   Use tools like Prometheus and Grafana to monitor system performance.

## 11. Data Volume and Scaling

1. **Check data volume:**
   Large amounts of data can slow down queries. Consider data archiving or partitioning.

2. **Implement sharding:**
   For very large datasets, consider implementing database sharding.

3. **Scale horizontally:**
   Add more nodes to your database cluster to distribute the load.

## Conclusion

Resolving slow query responses often requires a systematic approach to identify and address bottlenecks. Start with the most likely culprits based on your specific setup and gradually work through the list. Remember to test thoroughly after each change to ensure the issue is resolved without introducing new problems.

If you continue to experience issues after trying these steps, consider reaching out to the R2R community or support channels for more specialized assistance.

---

# Troubleshooting Guide: Workflow Orchestration Failures in R2R

Workflow orchestration is a critical component of R2R, managed by Hatchet. When orchestration failures occur, they can disrupt the entire data processing pipeline. This guide will help you identify and resolve common issues.

## 1. Check Hatchet Service Status

First, ensure that the Hatchet service is running properly:

```bash
docker ps | grep hatchet
```

Look for containers with names like `hatchet-engine`, `hatchet-api`, and `hatchet-rabbitmq`.

## 2. Examine Hatchet Logs

View the logs for the Hatchet engine:

```bash
docker logs r2r-hatchet-engine-1
```

Look for error messages or warnings that might indicate the source of the problem.

## 3. Common Issues and Solutions

### 3.1 Connection Issues with RabbitMQ

**Symptom:** Hatchet logs show connection errors to RabbitMQ.

**Solution:**
1. Check if RabbitMQ is running:
   ```bash
   docker ps | grep rabbitmq
   ```
2. Verify RabbitMQ credentials in the Hatchet configuration.
3. Ensure the RabbitMQ container is in the same Docker network as Hatchet.

### 3.2 Database Connection Problems

**Symptom:** Errors in Hatchet logs related to database connections.

**Solution:**
1. Verify Postgres container is running and healthy:
   ```bash
   docker ps | grep postgres
   ```
2. Check database connection settings in Hatchet configuration.
3. Ensure the Postgres container is in the same Docker network as Hatchet.

### 3.3 Workflow Definition Errors

**Symptom:** Specific workflows fail to start or execute properly.

**Solution:**
1. Review the workflow definition in your R2R configuration.
2. Check for syntax errors or invalid step definitions.
3. Verify that all required environment variables for the workflow are set.

### 3.4 Resource Constraints

**Symptom:** Workflows start but fail due to timeout or resource exhaustion.

**Solution:**
1. Check system resources (CPU, memory) on the host machine.
2. Adjust resource limits for Docker containers if necessary.
3. Consider scaling up your infrastructure or optimizing workflow resource usage.

### 3.5 Version Incompatibility

**Symptom:** Unexpected errors after updating R2R or Hatchet.

**Solution:**
1. Ensure all components (R2R, Hatchet, RabbitMQ) are compatible versions.
2. Check the R2R documentation for any breaking changes in recent versions.
3. Consider rolling back to a known working version if issues persist.

## 4. Advanced Debugging

### 4.1 Inspect Hatchet API

Use the Hatchet API to get more details about workflow executions:

1. Find the Hatchet API container:
   ```bash
   docker ps | grep hatchet-api
   ```
2. Use `curl` to query the API (replace `<container-id>` with the actual ID):
   ```bash
   docker exec <container-id> curl http://localhost:7077/api/v1/workflows
   ```

### 4.2 Check Hatchet Dashboard

If you have the Hatchet dashboard set up:

1. Access the dashboard (typically at `http://localhost:8002` if using default ports).
2. Navigate to the Workflows section to view detailed execution status and logs.

### 4.3 Analyze RabbitMQ Queues

Inspect RabbitMQ queues to check for message backlogs or routing issues:

1. Access the RabbitMQ management interface (typically at `http://localhost:15672`).
2. Check queue lengths, message rates, and any dead-letter queues.

## 5. Common Workflow-Specific Issues

### 5.1 Document Ingestion Failures

**Symptom:** Documents fail to process through the ingestion workflow.

**Solution:**
1. Check the Unstructured API configuration and connectivity.
2. Verify file permissions and formats of ingested documents.
3. Examine R2R logs for specific ingestion errors.

### 5.2 Vector Store Update Issues

**Symptom:** Vector store (Postgres with pgvector) not updating correctly.

**Solution:**
1. Check Postgres logs for any errors related to vector operations.
2. Verify the pgvector extension is properly installed and enabled.
3. Ensure the R2R configuration correctly specifies the vector store settings.

## 6. Seeking Further Help

If you're still experiencing issues after trying these solutions:

1. Gather all relevant logs (R2R, Hatchet, RabbitMQ, Postgres).
2. Document the steps to reproduce the issue.
3. Check the R2R GitHub repository for similar reported issues.
4. Consider opening a new issue on the R2R GitHub repository with your findings.

Remember to provide:
- R2R version (`r2r version`)
- Docker Compose configuration
- Relevant parts of your R2R configuration
- Detailed error messages and logs

By following this guide, you should be able to diagnose and resolve most workflow orchestration issues in R2R. If problems persist, don't hesitate to seek help from the R2R community or support channels.

---


**On this page**
1. Core Architecture
2. Document Processing Pipeline
3. Search and Retrieval System
4. Response Generation
5. System Components

## Core Architecture

R2R operates as a distributed system with several key components:

**API Layer**
- RESTful API for all operations
- Authentication and access control
- Request routing and validation

**Storage Layer**
- Document storage
- Vector embeddings
- User and permission data
- Knowledge graphs

**Processing Pipeline**
- Document parsing
- Chunking and embedding
- Relationship extraction
- Task orchestration

## Document Processing Pipeline

When you ingest a document into R2R:

1. **Document Parsing**
   - Files are processed based on type (PDF, text, images, etc.)
   - Text is extracted and cleaned
   - Metadata is preserved

2. **Chunking**
   - Documents are split into semantic units
   - Chunk size and overlap are configurable
   - Headers and structure are maintained

3. **Embedding Generation**
   - Each chunk is converted to a vector embedding
   - Multiple embedding models supported
   - Embeddings are optimized for search

4. **Knowledge Graph Creation**
   - Relationships between chunks are identified
   - Entities are extracted and linked
   - Graph structure is built and maintained

## Search and Retrieval System

R2R uses a sophisticated search system:

**Vector Search**
- High-dimensional vector similarity search
- Optimized indices for fast retrieval
- Configurable distance metrics

**Hybrid Search**
```
Query → [Vector Search Branch] → Semantic Results
     → [Keyword Search Branch] → Lexical Results
     → [Fusion Layer] → Final Ranked Results
```

**Ranking**
- Reciprocal rank fusion
- Configurable weights
- Result deduplication

## Response Generation

When generating responses:

1. **Context Building**
   - Relevant chunks are retrieved
   - Context is formatted for the LLM
   - Citations are prepared

2. **LLM Integration**
   - Context is combined with the query
   - System prompts guide response format
   - Streaming support for real-time responses

3. **Post-processing**
   - Response validation
   - Citation linking
   - Format cleaning

## System Components

R2R consists of several integrated services:

**Core Services**
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   API       │ ↔  │  Processor  │ ↔  │  Storage    │
└─────────────┘    └─────────────┘    └─────────────┘
       ↕                  ↕                  ↕
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Auth Server │ ↔  │ Orchestrator│ ↔  │ Search      │
└─────────────┘    └─────────────┘    └─────────────┘
```

**Database Layer**
- PostgreSQL for structured data
- pgvector for vector storage
- Graph data for relationships

**External Integrations**
- LLM providers (OpenAI, Anthropic, etc.)
- Authentication providers
- Storage systems

## Performance Considerations

R2R optimizes for several key metrics:

**Latency**
- Cached embeddings
- Optimized vector indices
- Request batching

**Scalability**
- Horizontal scaling support
- Distributed processing
- Load balancing

**Reliability**
- Task queuing
- Error handling
- Automatic retries

## Resource Management

R2R efficiently manages system resources:

1. **Memory Usage**
   - Vector index optimization
   - Chunk size management
   - Cache control

2. **Processing Power**
   - Parallel processing
   - Batch operations
   - Priority queuing

3. **Storage**
   - Efficient vector storage
   - Document versioning
   - Metadata indexing

For detailed deployment configurations and optimization strategies, refer to our [Configuration Guide](/documentation/configuration).

---

# Troubleshooting Guide: Docker Port Conflicts

Port conflicts are a common issue when deploying Docker containers, especially in complex setups like R2R. This guide will help you identify, diagnose, and resolve port conflicts in your Docker environment.

## Understanding Port Conflicts

A port conflict occurs when two processes attempt to use the same network port. In Docker, this typically happens when:

1. A container tries to bind to a port already in use by the host system.
2. Multiple containers attempt to use the same port.
3. A container's port mapping conflicts with another container or host process.

## Identifying Port Conflicts

Signs of a port conflict include:

- Error messages during container startup mentioning "port is already allocated" or "address already in use".
- Services failing to start or be accessible.
- Unexpected behavior in applications that rely on specific ports.

## Steps to Diagnose and Resolve Port Conflicts

### 1. Check for Used Ports

First, identify which ports are already in use on your system:

```bash
sudo lsof -i -P -n | grep LISTEN
```

or

```bash
netstat -tuln
```

### 2. Review Docker Compose File

Examine your `docker-compose.yml` file for port mappings:

```yaml
services:
  myservice:
    ports:
      - "8080:80"  # Host port 8080 maps to container port 80
```

### 3. Modify Port Mappings

If you identify a conflict, you can:

a. Change the host port in your Docker Compose file:

```yaml
services:
  myservice:
    ports:
      - "8081:80"  # Changed from 8080 to 8081
```

b. Use automatic port assignment:

```yaml
services:
  myservice:
    ports:
      - "80"  # Docker will assign a random available host port
```

### 4. Stop Conflicting Services

If a host service is using the required port:

```bash
sudo service conflicting_service stop
```

### 5. Release Docker Resources

Sometimes, stopping and removing all Docker containers and networks can help:

```bash
docker-compose down
docker system prune
```

### 6. Check for Docker Network Conflicts

Ensure your Docker networks don't have overlapping subnets:

```bash
docker network ls
docker network inspect network_name
```

### 7. Use Network Host Mode (Caution)

As a last resort, you can use host network mode, but this bypasses Docker's network isolation:

```yaml
services:
  myservice:
    network_mode: "host"
```

### 8. Debugging with Docker Logs

Check container logs for more detailed error messages:

```bash
docker-compose logs service_name
```

## Specific R2R Port Conflict Scenarios

### R2R API Server Conflict

If the R2R API server (default port 7272) is conflicting:

1. Check if any other service is using port 7272:
   ```bash
   sudo lsof -i :7272
   ```

2. Modify the R2R service in your docker-compose.yml:
   ```yaml
   services:
     r2r:
       ports:
         - "7273:7272"  # Changed host port to 7273
   ```

3. Update your environment variables:
   ```
   PORT=7273
   ```

### Hatchet Engine Conflict

If the Hatchet engine (default port 7077) is conflicting:

1. Check for conflicts:
   ```bash
   sudo lsof -i :7077
   ```

2. Modify the Hatchet engine service:
   ```yaml
   services:
     hatchet-engine:
       ports:
         - "7078:7077"
   ```

3. Update the `SERVER_GRPC_BROADCAST_ADDRESS` environment variable for the Hatchet engine service.

## Preventing Future Conflicts

1. Use environment variables for port numbers in your Docker Compose file.
2. Document the ports used by each service in your project.
3. Consider using tools like Traefik or Nginx as reverse proxies to manage port allocation dynamically.

By following this guide, you should be able to identify and resolve most port conflicts in your Docker and R2R setup. Remember, after making changes to your Docker Compose file or configuration, you'll need to rebuild and restart your services:

```bash
docker-compose down
docker-compose up --build
```

If problems persist, check the R2R documentation or seek help from the community support channels.

---


**On this page**
1. Before you begin
2. What is RAG?
3. Set up RAG with R2R
4. Configure RAG settings
5. How RAG works in R2R

RAG (Retrieval-Augmented Generation) combines the power of large language models with precise information retrieval from your own documents. When users ask questions, RAG first retrieves relevant information from your document collection, then uses this context to generate accurate, contextual responses. This ensures AI responses are both relevant and grounded in your specific knowledge base.

**Before you begin**

RAG in R2R has the following requirements:
* A running R2R instance (local or deployed)
* Access to an LLM provider (OpenAI, Anthropic, or local models)
* Documents ingested into your R2R system
* Basic configuration for document processing and embedding generation

## What is RAG?

RAG operates in three main steps:
1. **Retrieval**: Finding relevant information from your documents
2. **Augmentation**: Adding this information as context for the AI
3. **Generation**: Creating responses using both the context and the AI's knowledge

Benefits over traditional LLM applications:
* More accurate responses based on your specific documents
* Reduced hallucination by grounding answers in real content
* Ability to work with proprietary or recent information
* Better control over AI outputs

## Set up RAG with R2R

To start using RAG in R2R:

1. Install and start R2R:
```bash
pip install r2r
r2r serve --docker
```

2. Ingest your documents:
```bash
r2r ingest-files --file-paths /path/to/your/documents
```

3. Test basic RAG functionality:
```bash
r2r rag --query="your question here"
```

## Configure RAG settings

R2R offers several ways to customize RAG behavior:

1. **Retrieval Settings**:
```python
# Using hybrid search (combines semantic and keyword search)
client.rag(
    query="your question",
    vector_search_settings={"use_hybrid_search": True}
)

# Adjusting number of retrieved chunks
client.rag(
    query="your question",
    vector_search_settings={"top_k": 5}
)
```

2. **Generation Settings**:
```python
# Adjusting response style
client.rag(
    query="your question",
    rag_generation_config={
        "temperature": 0.7,
        "model": "openai/gpt-4"
    }
)
```

## How RAG works in R2R

R2R's RAG implementation uses a sophisticated pipeline:

**Document Processing**
* Documents are split into semantic chunks
* Each chunk is embedded using AI models
* Chunks are stored with metadata and relationships

**Retrieval Process**
* Queries are processed using hybrid search
* Both semantic similarity and keyword matching are considered
* Results are ranked by relevance scores

**Response Generation**
* Retrieved chunks are formatted as context
* The LLM generates responses using this context
* Citations and references can be included

**Advanced Features**
* GraphRAG for relationship-aware responses
* Multi-step RAG for complex queries
* Agent-based RAG for interactive conversations

## Best Practices

1. **Document Processing**
   * Use appropriate chunk sizes (256-1024 tokens)
   * Maintain document metadata
   * Consider document relationships

2. **Query Optimization**
   * Use hybrid search for better retrieval
   * Adjust relevance thresholds
   * Monitor and analyze search performance

3. **Response Generation**
   * Balance temperature for creativity vs accuracy
   * Use system prompts for consistent formatting
   * Implement error handling and fallbacks

For more detailed information, visit our [RAG Configuration Guide](/documentation/configuration/rag) or try our [Quickstart](/documentation/quickstart).

---




**On this page**
1. What does R2R do?
2. What can R2R do for my applications?
3. What can R2R do for my developers?
4. What can R2R do for my business?
5. Getting started

Companies like OpenAI, Anthropic, and Google have shown the incredible potential of AI for understanding and generating human language. But building reliable AI applications that can work with your organization's specific knowledge and documents requires significant expertise and infrastructure. Your company isn't an AI infrastructure company: **it doesn't make sense for you to build a complete RAG system from scratch.**

R2R (RAG to Riches) provides the infrastructure and tools to help you implement **efficient, scalable, and reliable AI-powered document understanding** in your applications.

## What does R2R do?

R2R consists of three main components: **document processing**, **AI-powered search**, and **analytics**. The document processing and search capabilities make it easier for your developers to create intelligent applications that can understand and work with your organization's knowledge. The analytics tools enable your teams to monitor performance, understand usage patterns, and continuously improve the system.

## What can R2R do for my applications?

R2R provides your applications with production-ready RAG capabilities:
- Fast and accurate document search using both semantic and keyword matching
- Intelligent document processing that works with PDFs, images, audio, and more
- Automatic relationship extraction to build knowledge graphs
- Built-in user management and access controls
- Simple integration through REST APIs and SDKs

## What can R2R do for my developers?

R2R provides a complete toolkit that simplifies building AI-powered applications:
- **Ready-to-use Docker deployment** for quick setup and testing
- **Python and JavaScript SDKs** for easy integration
- **RESTful API** for language-agnostic access
- **Built-in orchestration** for handling large-scale document processing
- **Flexible configuration** through intuitive config files
- **Comprehensive documentation** and examples
- **Local deployment option** for working with sensitive data

## What can R2R do for my business?

R2R provides the infrastructure to build AI applications that can:
- **Make your documents searchable** with state-of-the-art AI
- **Answer questions** using your organization's knowledge
- **Process and understand** documents at scale
- **Secure sensitive information** through built-in access controls
- **Monitor usage and performance** through analytics
- **Scale efficiently** as your needs grow

## Getting Started

The fastest way to start with R2R is through Docker:
```bash
pip install r2r
r2r serve --docker
```

This gives you a complete RAG system running at http://localhost:7272 with:
- Document processing pipeline
- Vector search capabilities
- GraphRAG features
- User management
- Analytics dashboard

Visit our [Quickstart Guide](/documentation/quickstart) to begin building with R2R.

---


## Version 0.3.20 — Sep. 6, 2024

### New Features
- [R2R Light](https://r2r-docs.sciphi.ai/documentation/installation/light/local-system) installation added
- Removed Neo4j and implemented GraphRAG inside of Postgres
- Improved efficiency and configurability of knowledge graph construction process

### Bug Fixes
- Minor bug fixes around config logic and other.

---

# Troubleshooting Guide: R2R API Endpoints Not Responding

When R2R API endpoints fail to respond, it can disrupt your entire workflow. This guide will help you diagnose and resolve issues related to unresponsive API endpoints.

## 1. Verify API Service Status

First, ensure that the R2R API service is running:

```bash
docker ps | grep r2r
```

Look for a container with "r2r" in its name and check its status.

## 2. Check API Logs

Examine the logs of the R2R API container:

```bash
docker logs <r2r-container-id>
```

Look for error messages or exceptions that might indicate why the API is not responding.

## 3. Common Issues and Solutions

### 3.1 Network Connectivity

**Symptom:** Unable to reach the API from outside the Docker network.

**Solutions:**
- Verify port mappings in your Docker Compose file.
- Ensure the host firewall isn't blocking the API port.
- Check if the API is bound to the correct network interface (0.0.0.0 for all interfaces).

### 3.2 Dependencies Not Ready

**Symptom:** API starts but immediately exits or fails to initialize.

**Solutions:**
- Verify that all required services (Postgres, etc.) are up and healthy.
- Check if the `depends_on` conditions in the Docker Compose file are correct.
- Increase the retry count or add a delay in the API service startup script.

### 3.3 Configuration Errors

**Symptom:** API logs show configuration-related errors.

**Solutions:**
- Double-check environment variables in the Docker Compose file.
- Verify that the config file (if used) is correctly mounted and accessible.
- Ensure all required configuration parameters are set.

### 3.4 Resource Constraints

**Symptom:** API becomes unresponsive under load or fails to start due to lack of resources.

**Solutions:**
- Check Docker host resources (CPU, memory, disk space).
- Adjust resource limits in Docker Compose file if necessary.
- Consider scaling the API service or upgrading the host machine.

### 3.5 Database Connection Issues

**Symptom:** API logs show database connection errors.

**Solutions:**
- Verify database credentials and connection strings.
- Check if the database service is running and accessible.
- Ensure the database is initialized with the correct schema.

### 3.6 Hatchet Integration Problems

**Symptom:** API fails to communicate with Hatchet service.

**Solutions:**
- Verify Hatchet service is running and healthy.
- Check Hatchet API key and configuration.
- Ensure network connectivity between R2R and Hatchet services.

## 4. API-specific Debugging Steps

1. **Test individual endpoints:**
   Use tools like cURL or Postman to test specific endpoints and isolate the problem.

   ```bash
   curl http://localhost:7272/v2/health
   ```

2. **Check API documentation:**
   Verify that you're using the correct endpoint URLs and request formats.

3. **Monitor API metrics:**
   If available, check API metrics for response times, error rates, and request volumes.

4. **Verify API versioning:**
   Ensure you're using the correct API version in your requests.

## 5. Advanced Troubleshooting

### 5.1 Network Debugging

Use network debugging tools to diagnose connectivity issues:

```bash
docker network inspect r2r-network
```

### 5.2 Interactive Debugging

Access the R2R container interactively to run diagnostics:

```bash
docker exec -it <r2r-container-id> /bin/bash
```

### 5.3 API Server Logs

If the API uses a separate web server (e.g., uvicorn), check its logs:

```bash
docker exec <r2r-container-id> cat /path/to/uvicorn.log
```

## 6. Preventive Measures

1. Implement robust health checks in your Docker Compose file.
2. Use logging and monitoring tools to proactively detect issues.
3. Implement circuit breakers and fallback mechanisms in your application.
4. Regularly update R2R and its dependencies to the latest stable versions.

## 7. Seeking Help

If you're still experiencing issues:

1. Gather all relevant logs, configurations, and error messages.
2. Check the R2R documentation and GitHub issues for similar problems.
3. Reach out to the R2R community on Discord or GitHub for support.
4. When reporting an issue, provide:
   - R2R version
   - Docker and Docker Compose versions
   - Host OS and version
   - Detailed description of the problem and steps to reproduce
   - Relevant logs and configuration files (with sensitive information redacted)

By following this guide, you should be able to diagnose and resolve most issues with unresponsive R2R API endpoints. Remember to approach the problem systematically and gather as much information as possible before seeking external help.

---

# Troubleshooting Guide: High Resource Usage in R2R Docker Deployments

High resource usage in R2R Docker deployments can lead to performance issues, system instability, or even service outages. This guide will help you identify the cause of high resource usage and provide steps to mitigate the problem.

## 1. Identifying the Issue

First, determine which resources are being overused:

### 1.1 Check Overall System Resources

Use the `top` or `htop` command to get an overview of system resource usage:

```bash
top
```

Look for high CPU usage, memory consumption, or swap usage.

### 1.2 Check Docker-specific Resource Usage

Use Docker's built-in commands to check resource usage for containers:

```bash
docker stats
```

This will show CPU, memory, and I/O usage for each container.

## 2. Common Causes and Solutions

### 2.1 High CPU Usage

#### Possible Causes:
- Inefficient queries or data processing
- Continuous background tasks
- Improperly configured LLM inference

#### Solutions:
1. **Optimize queries:**
   - Review and optimize database queries, especially those involving large datasets.

2. **Adjust background task frequency:**
   - Review Hatchet workflows and adjust the frequency of recurring tasks.
   - Implement rate limiting for resource-intensive operations.

3. **LLM configuration:**
   - If using local LLMs via Ollama, consider adjusting model parameters or switching to a lighter model.
   - For cloud LLMs, implement caching to reduce redundant API calls.

4. **Scale horizontally:**
   - Consider distributing the workload across multiple R2R instances.

### 2.2 High Memory Usage

#### Possible Causes:
- Memory leaks in custom code
- Inefficient caching
- Large in-memory datasets

#### Solutions:
1. **Identify memory-hungry containers:**
   ```bash
   docker stats --format "table {{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}"
   ```

2. **Analyze R2R application logs:**
   - Look for patterns of increasing memory usage over time.

3. **Optimize memory usage:**
   - Implement proper garbage collection in custom code.
   - Review and optimize caching strategies.
   - Consider using streaming for large data processing tasks instead of loading entire datasets into memory.

4. **Adjust container memory limits:**
   - Update the Docker Compose file to set appropriate memory limits for containers:
     ```yaml
     services:
       r2r:
         deploy:
           resources:
             limits:
               memory: 4G
     ```

### 2.3 High Disk I/O

#### Possible Causes:
- Frequent logging
- Inefficient database operations
- Large file ingestion processes

#### Solutions:
1. **Monitor disk I/O:**
   ```bash
   docker stats --format "table {{.Name}}\t{{.BlockIO}}"
   ```

2. **Optimize logging:**
   - Reduce log verbosity for non-critical information.
   - Implement log rotation to manage file sizes.

3. **Database optimizations:**
   - Ensure proper indexing in Postgres.
   - Optimize query patterns to reduce full table scans.

4. **File ingestion improvements:**
   - Implement chunking for large file ingestion.
   - Consider using a dedicated storage service for large files.

## 3. Monitoring and Prevention

Implement proactive monitoring to catch resource issues early:

1. **Set up Docker monitoring:**
   - Use tools like Prometheus and Grafana to monitor Docker metrics.
   - Set up alerts for when resource usage exceeds certain thresholds.

2. **Implement application-level metrics:**
   - Use libraries like `prometheus_client` in Python to expose custom metrics.
   - Monitor key performance indicators specific to your R2R usage.

3. **Regular performance audits:**
   - Periodically review resource usage patterns.
   - Conduct load testing to identify potential bottlenecks before they impact production.

## 4. Advanced Debugging

For persistent issues:

1. **Profile the R2R application:**
   - Use Python profiling tools like cProfile or memory_profiler to identify resource-intensive code sections.

2. **Analyze Docker logs:**
   ```bash
   docker logs r2r-container-name
   ```

3. **Inspect container details:**
   ```bash
   docker inspect r2r-container-name
   ```

4. **Review orchestration logs:**
   - Check Hatchet logs for insights into task execution and resource allocation.

## 5. Scaling Considerations

If high resource usage persists despite optimizations:

1. **Vertical scaling:**
   - Increase resources (CPU, RAM) for the Docker host.

2. **Horizontal scaling:**
   - Implement load balancing across multiple R2R instances.
   - Consider using Docker Swarm or Kubernetes for orchestration.

3. **Service separation:**
   - Move resource-intensive components (e.g., database, LLM inference) to dedicated hosts.

## Conclusion

High resource usage in R2R Docker deployments can be challenging but is often resolvable through careful analysis and optimization. Always ensure you have proper monitoring in place, and regularly review your deployment's performance to catch issues early. If problems persist, don't hesitate to reach out to the R2R community or consider consulting with cloud infrastructure experts.

---

# Troubleshooting Guide: RabbitMQ Connectivity Issues in R2R

RabbitMQ is a critical component in the R2R architecture, used for message queuing and task orchestration. Connectivity issues can disrupt the entire system. This guide will help you diagnose and resolve common RabbitMQ connectivity problems.

## 1. Verify RabbitMQ Service Status

First, ensure that the RabbitMQ service is running:

```bash
docker ps | grep rabbitmq
```

If you don't see the RabbitMQ container running, start it:

```bash
docker-compose up -d hatchet-rabbitmq
```

## 2. Check RabbitMQ Logs

View the RabbitMQ container logs:

```bash
docker logs r2r-hatchet-rabbitmq-1
```

Look for error messages related to connectivity, authentication, or resource issues.

## 3. Verify RabbitMQ Connection Settings

Ensure that the connection settings in your R2R configuration match the RabbitMQ service:

1. Check the `SERVER_TASKQUEUE_RABBITMQ_URL` environment variable in the `hatchet-setup-config` service.
2. Verify that the URL format is correct: `amqp://user:password@hatchet-rabbitmq:5672/`

## 4. Common Issues and Solutions

### 4.1 Authentication Failures

**Symptom:** Logs show authentication errors.

**Solution:**
1. Verify the RabbitMQ credentials:
   ```bash
   docker exec r2r-hatchet-rabbitmq-1 rabbitmqctl list_users
   ```
2. If necessary, reset the password:
   ```bash
   docker exec r2r-hatchet-rabbitmq-1 rabbitmqctl change_password user newpassword
   ```
3. Update the `SERVER_TASKQUEUE_RABBITMQ_URL` in your R2R configuration with the new credentials.

### 4.2 Network Connectivity

**Symptom:** Services can't connect to RabbitMQ.

**Solution:**
1. Ensure all services are on the same Docker network:
   ```bash
   docker network inspect r2r-network
   ```
2. Verify that the RabbitMQ service is accessible within the network:
   ```bash
   docker run --rm --network r2r-network alpine ping hatchet-rabbitmq
   ```

### 4.3 Port Conflicts

**Symptom:** RabbitMQ fails to start due to port conflicts.

**Solution:**
1. Check if the ports are already in use:
   ```bash
   sudo lsof -i :5672
   sudo lsof -i :15672
   ```
2. Modify the port mappings in your Docker Compose file if necessary.

### 4.4 Resource Constraints

**Symptom:** RabbitMQ becomes unresponsive or crashes frequently.

**Solution:**
1. Check RabbitMQ resource usage:
   ```bash
   docker stats r2r-hatchet-rabbitmq-1
   ```
2. Increase resources allocated to the RabbitMQ container in your Docker Compose file:
   ```yaml
   hatchet-rabbitmq:
     # ... other configurations ...
     deploy:
       resources:
         limits:
           cpus: '1'
           memory: 1G
   ```

### 4.5 File Descriptor Limits

**Symptom:** RabbitMQ logs show warnings about file descriptor limits.

**Solution:**
1. Increase the file descriptor limit for the RabbitMQ container:
   ```yaml
   hatchet-rabbitmq:
     # ... other configurations ...
     ulimits:
       nofile:
         soft: 65536
         hard: 65536
   ```

## 5. Advanced Troubleshooting

### 5.1 RabbitMQ Management Interface

Access the RabbitMQ Management Interface for detailed diagnostics:

1. Enable the management plugin if not already enabled:
   ```bash
   docker exec r2r-hatchet-rabbitmq-1 rabbitmq-plugins enable rabbitmq_management
   ```
2. Access the interface at `http://localhost:15672` (use the credentials defined in your Docker Compose file).

### 5.2 Network Packet Capture

If you suspect network issues, capture and analyze network traffic:

```bash
docker run --net=container:r2r-hatchet-rabbitmq-1 --rm -v $(pwd):/cap nicolaka/netshoot tcpdump -i eth0 -w /cap/rabbitmq_traffic.pcap
```

Analyze the captured file with Wireshark for detailed network diagnostics.

### 5.3 RabbitMQ Cluster Status

If you're running a RabbitMQ cluster, check its status:

```bash
docker exec r2r-hatchet-rabbitmq-1 rabbitmqctl cluster_status
```

## 6. Preventive Measures

1. Implement health checks in your Docker Compose file:
   ```yaml
   hatchet-rabbitmq:
     # ... other configurations ...
     healthcheck:
       test: ["CMD", "rabbitmqctl", "status"]
       interval: 30s
       timeout: 10s
       retries: 5
   ```

2. Set up monitoring and alerting for RabbitMQ using tools like Prometheus and Grafana.

3. Regularly backup RabbitMQ definitions and data:
   ```bash
   docker exec r2r-hatchet-rabbitmq-1 rabbitmqctl export_definitions /tmp/rabbitmq_defs.json
   docker cp r2r-hatchet-rabbitmq-1:/tmp/rabbitmq_defs.json ./rabbitmq_defs.json
   ```

By following this guide, you should be able to diagnose and resolve most RabbitMQ connectivity issues in your R2R deployment. If problems persist, consider seeking help from the RabbitMQ community or consulting the official RabbitMQ documentation for more advanced troubleshooting techniques.

---


## System Diagram


```mermaid
graph TD
    U((User)) -->|Query| GW[Traefik Gateway]
    GW -->|Route| API[R2R API Cluster]
    API -->|Authenticate| AS[Auth Service]

    R2R[R2R Application] -->|Use| API

    subgraph "Core Services"
        AS
        ReS[Retrieval Service]
        IS[Ingestion Service]
        GBS[Graph Builder Service]
        AMS[App Management Service]
    end

    subgraph "Providers"
        EP[Embedding Provider]
        LP[LLM Provider]
        AP[Auth Provider]
        IP[Ingestion Provider]
    end

    IS & GBS & ReS <-->|Coordinate| O

    ReS -->|Use| EP
    ReS -->|Use| LP
    IS -->|Use| EP
    IS -->|Use| IP
    GBS -->|Use| LP
    AS -->|Use| AP

    subgraph "Orchestration"
        O[Orchestrator]
        RMQ[RabbitMQ]
        O <-->|Use| RMQ
    end

    subgraph "Storage"
        PG[(Postgres + pgvector)]
        FS[File Storage]
    end

    AS & AMS & ReS -->|Use| PG
    GBS & ReS -->|Use| Neo
    IS -->|Use| FS

    classDef gateway fill:#2b2b2b,stroke:#ffffff,stroke-width:2px;
    classDef api fill:#4444ff,stroke:#ffffff,stroke-width:2px;
    classDef orchestrator fill:#007acc,stroke:#ffffff,stroke-width:2px;
    classDef messagequeue fill:#2ca02c,stroke:#ffffff,stroke-width:2px;
    classDef storage fill:#336791,stroke:#ffffff,stroke-width:2px;
    classDef providers fill:#ff7f0e,stroke:#ffffff,stroke-width:2px;
    classDef auth fill:#ff0000,stroke:#ffffff,stroke-width:2px;
    classDef application fill:#9932cc,stroke:#ffffff,stroke-width:2px;
    class GW gateway;
    class API api;
    class O orchestrator;
    class RMQ messagequeue;
    class PG,Neo,FS storage;
    class EP,LP,AP,IP providers;
    class AS auth;
    class R2R application;
```

## System Overview

R2R is built on a modular, service-oriented architecture designed for scalability and flexibility:

1. **API Layer**: A FastAPI-based cluster handles incoming requests, routing them to appropriate services.

2. **Core Services**: Specialized services for authentication, retrieval, ingestion, graph building, and app management.

3. **Orchestration**: Manages complex workflows and long-running tasks using a message queue system.

4. **Storage**: Utilizes PostgreSQL with pgvector for vector storage and search, and graph search.

5. **Providers**: Pluggable components for embedding, LLM, auth, and ingestion services, supporting multimodal ingestion and flexible model integration.

6. **R2R Application**: A React+Next.js app providing a user interface for interacting with the R2R system.

This architecture enables R2R to handle everything from simple RAG applications to complex, production-grade systems with advanced features like hybrid search and GraphRAG.

Ready to get started? Check out our [Docker installation guide](/documentation/installation/full/docker) and [Quickstart tutorial](/documentation/quickstart) to begin your R2R journey.

---

# Troubleshooting Guide: Vector Storage Problems in R2R

Vector storage is a crucial component in R2R (RAG to Riches) for efficient similarity searches. This guide focuses on troubleshooting common vector storage issues, particularly with Postgres and pgvector.

## 1. Connection Issues

### Symptom: R2R can't connect to the vector database

1. **Check Postgres Connection:**
   ```bash
   psql -h localhost -U your_username -d your_database
   ```
   If this fails, the issue might be with Postgres itself, not specifically vector storage.

2. **Verify Environment Variables:**
   Ensure these are correctly set in your R2R configuration:
   - `R2R_POSTGRES_USER`
   - `R2R_POSTGRES_PASSWORD`
   - `R2R_POSTGRES_HOST`
   - `R2R_POSTGRES_PORT`
   - `R2R_POSTGRES_DBNAME`
   - `R2R_PROJECT_NAME`

3. **Check Docker Network:**
   If using Docker, ensure the R2R and Postgres containers are on the same network:
   ```bash
   docker network inspect r2r-network
   ```

## 2. pgvector Extension Issues

### Symptom: "extension pgvector does not exist" error

1. **Check if pgvector is Installed:**
   Connect to your database and run:
   ```sql
   SELECT * FROM pg_extension WHERE extname = 'vector';
   ```

2. **Install pgvector:**
   If not installed, run:
   ```sql
   CREATE EXTENSION vector;
   ```

3. **Verify Postgres Version:**
   pgvector requires Postgres 11 or later. Check your version:
   ```sql
   SELECT version();
   ```

## 3. Vector Dimension Mismatch

### Symptom: Error inserting vectors or during similarity search

1. **Check Vector Dimensions:**
   Verify the dimension of vectors you're trying to insert matches your schema:
   ```sql
   SELECT * FROM information_schema.columns
   WHERE table_name = 'your_vector_table' AND data_type = 'vector';
   ```

2. **Verify R2R Configuration:**
   Ensure the vector dimension in your R2R configuration matches your database schema.

3. **Recreate Table with Correct Dimensions:**
   If dimensions are mismatched, you may need to recreate the table:
   ```sql
   DROP TABLE your_vector_table;
   CREATE TABLE your_vector_table (id bigserial PRIMARY KEY, embedding vector(384));
   ```

## 4. Performance Issues

### Symptom: Slow similarity searches

1. **Check Index:**
   Ensure you have an appropriate index:
   ```sql
   CREATE INDEX ON your_vector_table USING ivfflat (embedding vector_cosine_ops);
   ```

2. **Analyze Table:**
   Run ANALYZE to update statistics:
   ```sql
   ANALYZE your_vector_table;
   ```

3. **Monitor Query Performance:**
   Use `EXPLAIN ANALYZE` to check query execution plans:
   ```sql
   EXPLAIN ANALYZE SELECT * FROM your_vector_table
   ORDER BY embedding <=> '[your_vector]' LIMIT 10;
   ```

4. **Adjust Work Memory:**
   If dealing with large vectors, increase work_mem:
   ```sql
   SET work_mem = '1GB';
   ```

## 5. Data Integrity Issues

### Symptom: Unexpected search results or missing data

1. **Check Vector Normalization:**
   Ensure vectors are normalized before insertion if using cosine similarity.

2. **Verify Data Insertion:**
   Check if data is being correctly inserted:
   ```sql
   SELECT COUNT(*) FROM your_vector_table;
   ```

3. **Inspect Random Samples:**
   Look at some random entries to ensure data quality:
   ```sql
   SELECT * FROM your_vector_table ORDER BY RANDOM() LIMIT 10;
   ```

## 6. Disk Space Issues

### Symptom: Insertion failures or database unresponsiveness

1. **Check Disk Space:**
   ```bash
   df -h
   ```

2. **Monitor Postgres Disk Usage:**
   ```sql
   SELECT pg_size_pretty(pg_database_size('your_database'));
   ```

3. **Identify Large Tables:**
   ```sql
   SELECT relname, pg_size_pretty(pg_total_relation_size(relid))
   FROM pg_catalog.pg_statio_user_tables
   ORDER BY pg_total_relation_size(relid) DESC;
   ```

## 7. Backup and Recovery

If all else fails, you may need to restore from a backup:

1. **Create a Backup:**
   ```bash
   pg_dump -h localhost -U your_username -d your_database > backup.sql
   ```

2. **Restore from Backup:**
   ```bash
   psql -h localhost -U your_username -d your_database < backup.sql
   ```

## Getting Further Help

If these steps don't resolve your issue:

1. Check R2R logs for more detailed error messages.
2. Consult the [pgvector documentation](https://github.com/pgvector/pgvector) for advanced troubleshooting.
3. Reach out to the R2R community or support channels with detailed information about your setup and the steps you've tried.

Remember to always backup your data before making significant changes to your database or vector storage configuration.

---

# Troubleshooting Guide: Docker Containers Failing to Start

When Docker containers fail to start, it can be frustrating and halt your development process. This guide will help you diagnose and resolve common issues.

## 1. Check Container Status

First, check the status of your containers:

```bash
docker ps -a
```

Look for containers with a status of "Exited" or "Created" but not "Up".

## 2. View Container Logs

For containers that failed to start, view the logs:

```bash
docker logs <container_id_or_name>
```

Look for error messages that might indicate why the container failed to start.

## 3. Common Issues and Solutions

### 3.1 Port Conflicts

**Symptom:** Error message about ports already in use.

**Solution:**
- Check if the port is already in use by another process:
  ```bash
  sudo lsof -i :<port_number>
  ```
- Change the port mapping in your Docker run command or docker-compose file.

### 3.2 Missing Environment Variables

**Symptom:** Container exits immediately, logs show errors about missing environment variables.

**Solution:**
- Ensure all required environment variables are set in your Docker run command or docker-compose file.
- Double-check your .env file if you're using one.

### 3.3 Insufficient System Resources

**Symptom:** Container fails to start, logs mention memory or CPU limits.

**Solution:**
- Check your system resources:
  ```bash
  docker info
  ```
- Adjust resource limits in Docker settings or in your container configuration.

### 3.4 Image Not Found

**Symptom:** Error message about the image not being found.

**Solution:**
- Ensure the image exists locally or is accessible from the specified registry:
  ```bash
  docker images
  ```
- Pull the image manually if needed:
  ```bash
  docker pull <image_name>:<tag>
  ```

### 3.5 Volume Mount Issues

**Symptom:** Errors related to volume mounts or missing files.

**Solution:**
- Verify that the paths for volume mounts exist on the host system.
- Check permissions on the host directories.

### 3.6 Network Issues

**Symptom:** Container can't connect to other services or the internet.

**Solution:**
- Check Docker network settings:
  ```bash
  docker network ls
  docker network inspect <network_name>
  ```
- Ensure the container is connected to the correct network.

## 4. Debugging Steps

If the above solutions don't resolve the issue:

1. **Run the container in interactive mode:**
   ```bash
   docker run -it --entrypoint /bin/bash <image_name>
   ```
   This allows you to explore the container environment.

2. **Check container health:**
   If your container has a HEALTHCHECK instruction, review its status:
   ```bash
   docker inspect --format='{{json .State.Health}}' <container_name>
   ```

3. **Review Dockerfile:**
   Ensure your Dockerfile is correctly configured, especially the CMD or ENTRYPOINT instructions.

4. **Verify dependencies:**
   Make sure all required dependencies are installed and correctly configured in the image.

## 5. Advanced Troubleshooting

### 5.1 Docker Daemon Logs

Check Docker daemon logs for system-level issues:

```bash
sudo journalctl -u docker.service
```

### 5.2 Docker Events

Monitor Docker events in real-time:

```bash
docker events
```

### 5.3 Resource Constraints

Review and adjust resource constraints:

```bash
docker update --cpu-shares 512 --memory 512M <container_name>
```

## 6. Seeking Help

If you're still stuck:

1. Gather all relevant logs and configuration files.
2. Check Docker documentation and community forums.
3. If using a specific service (like R2R), consult their support channels or documentation.
4. Consider posting a detailed question on Stack Overflow or Docker community forums.

Remember to always provide:
- Docker version (`docker version`)
- Host OS and version
- Detailed error messages
- Steps to reproduce the issue

By following this guide, you should be able to diagnose and resolve most issues with Docker containers failing to start. If problems persist, don't hesitate to seek help from the Docker community or relevant support channels.

---

# Troubleshooting Guide: Unstructured.io Setup Difficulties with R2R

Unstructured.io is a crucial component in R2R for handling file ingestion. This guide addresses common issues and their solutions when setting up and using Unstructured.io within the R2R ecosystem.

## 1. Installation Issues

### 1.1 Missing Dependencies

**Problem:** Unstructured.io fails to install due to missing system dependencies.

**Solution:**
1. Ensure you have the required system libraries:
   ```bash
   sudo apt-get update
   sudo apt-get install -y python3-dev libxml2-dev libxslt1-dev antiword unrtf poppler-utils pstotext tesseract-ocr flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev swig
   ```
2. If using pip, install with extras:
   ```bash
   pip install "unstructured[all-deps]"
   ```

### 1.2 Version Compatibility

**Problem:** Incompatibility between Unstructured.io and R2R versions.

**Solution:**
1. Check the R2R documentation for the recommended Unstructured.io version.
2. Install the specific version:
   ```bash
   pip install unstructured==X.Y.Z
   ```

## 2. Configuration Issues

### 2.1 API Key Not Recognized

**Problem:** R2R fails to connect to Unstructured.io due to API key issues.

**Solution:**
1. Verify your API key is correctly set in the R2R configuration:
   ```toml
   [unstructured]
   api_key = "your-api-key-here"
   ```
2. Ensure the environment variable is set:
   ```bash
   export UNSTRUCTURED_API_KEY=your-api-key-here
   ```

### 2.2 Incorrect API Endpoint

**Problem:** R2R can't reach the Unstructured.io API.

**Solution:**
1. Check the API endpoint in your R2R configuration:
   ```toml
   [unstructured]
   api_url = "https://api.unstructured.io/general/v0/general"
   ```
2. If using a self-hosted version, ensure the URL is correct.

## 3. Runtime Errors

### 3.1 File Processing Failures

**Problem:** Unstructured.io fails to process certain file types.

**Solution:**
1. Verify the file type is supported by Unstructured.io.
2. Check file permissions and ensure R2R has access to the files.
3. For specific file types, install additional dependencies:
   ```bash
   pip install "unstructured[pdf]"  # For enhanced PDF support
   ```

### 3.2 Memory Issues

**Problem:** Unstructured.io crashes due to insufficient memory when processing large files.

**Solution:**
1. Increase the available memory for the R2R process.
2. If using Docker, adjust the container's memory limit:
   ```yaml
   services:
     r2r:
       deploy:
         resources:
           limits:
             memory: 4G
   ```

### 3.3 Slow Processing

**Problem:** File processing is exceptionally slow.

**Solution:**
1. Check system resources (CPU, RAM) and ensure they meet minimum requirements.
2. Consider using Unstructured.io's async API for large batch processing.
3. Implement a caching mechanism in R2R to store processed results.

## 4. Integration Issues

### 4.1 Data Format Mismatch

**Problem:** R2R fails to interpret the output from Unstructured.io correctly.

**Solution:**
1. Verify that R2R's parsing logic matches Unstructured.io's output format.
2. Check for any recent changes in Unstructured.io's API responses and update R2R accordingly.

### 4.2 Rate Limiting

**Problem:** Hitting API rate limits when using Unstructured.io's cloud service.

**Solution:**
1. Implement rate limiting in your R2R application.
2. Consider upgrading your Unstructured.io plan for higher limits.
3. Use local deployment of Unstructured.io for unlimited processing.

## 5. Local Deployment Issues

### 5.1 Docker Container Failures

**Problem:** Unstructured.io Docker container fails to start or crashes.

**Solution:**
1. Check Docker logs:
   ```bash
   docker logs [container_name]
   ```
2. Ensure all required environment variables are set.
3. Verify that the Docker image version is compatible with your R2R version.

### 5.2 Network Connectivity

**Problem:** R2R can't connect to locally deployed Unstructured.io.

**Solution:**
1. Ensure the Unstructured.io container is on the same Docker network as R2R.
2. Check firewall settings and ensure necessary ports are open.
3. Verify the URL in R2R configuration points to the correct local address.

## 6. Debugging Tips

1. Enable verbose logging in both R2R and Unstructured.io.
2. Use tools like `curl` to test API endpoints directly.
3. Implement proper error handling in R2R to capture and log Unstructured.io-related issues.

## 7. Seeking Help

If issues persist:
1. Check the [Unstructured.io documentation](https://unstructured-io.github.io/unstructured/).
2. Visit the [R2R GitHub repository](https://github.com/SciPhi-AI/R2R) for specific integration issues.
3. Reach out to the R2R community on Discord or other support channels.

Remember to provide detailed information when seeking help, including:
- R2R and Unstructured.io versions
- Deployment method (cloud, local, Docker)
- Specific error messages and logs
- Steps to reproduce the issue

By following this guide, you should be able to troubleshoot and resolve most Unstructured.io setup and integration issues within your R2R deployment.

---


## Welcome to the R2R API

R2R (RAG to Riches) is a powerful library that offers both methods and a REST API for document ingestion, Retrieval-Augmented Generation (RAG), evaluation, and additional features like observability, analytics, and document management. This API documentation will guide you through the various endpoints and functionalities R2R provides.

<Note>
  This API documentation is designed to help developers integrate R2R's capabilities into their applications efficiently. Whether you're building a search engine, a question-answering system, or a document management solution, the R2R API has you covered.
</Note>

## Key Features

R2R API offers a wide range of features, including:

  - Document Ingestion and Management
  - AI-Powered Search (Vector, Hybrid, and Knowledge Graph)
  - Retrieval-Augmented Generation (RAG)
  - User Auth & Management
  - Observability and Analytics

<Card
  title="R2R GitHub Repository"
  icon="github"
  href="https://github.com/SciPhi-AI/R2R"
>
  View the R2R source code and contribute
</Card>

## Getting Started

To get started with the R2R API, you'll need to:

1. Install R2R in your environment
2. Run the server with `r2r serve`, or customize your FastAPI for production settings.

For detailed installation and setup instructions, please refer to our [Installation Guide](/documentation/installation).

---

# Troubleshooting Guide: Missing or Incorrect API Keys in R2R

API keys are crucial for authenticating and accessing various services integrated with R2R. Missing or incorrect API keys can lead to connection failures and service disruptions. This guide will help you identify and resolve API key issues.

## 1. Identifying API Key Issues

Common symptoms of API key problems include:

- Error messages mentioning "unauthorized," "authentication failed," or "invalid API key"
- Specific services or integrations not working while others function correctly
- Unexpected 401 or 403 HTTP status codes in logs

## 2. Checking API Key Configuration

### 2.1 Environment Variables

R2R uses environment variables to store API keys. Check if the required environment variables are set:

```bash
env | grep API_KEY
```

Look for variables like:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `AZURE_API_KEY`
- `UNSTRUCTURED_API_KEY`
- `HATCHET_CLIENT_TOKEN`

### 2.2 Configuration Files

If you're using configuration files (e.g., `r2r.toml`), verify that API keys are correctly set:

```bash
grep -i "api_key" /path/to/your/r2r.toml
```

## 3. Common API Key Issues and Solutions

### 3.1 OpenAI API Key

**Issue:** OpenAI services not working or returning authentication errors.

**Solution:**
1. Verify the `OPENAI_API_KEY` is set:
   ```bash
   echo $OPENAI_API_KEY
   ```
2. Ensure the key starts with "sk-".
3. Check the key's validity in the OpenAI dashboard.
4. Regenerate the key if necessary and update the environment variable.

### 3.2 Anthropic API Key

**Issue:** Claude or other Anthropic models not functioning.

**Solution:**
1. Confirm the `ANTHROPIC_API_KEY` is set:
   ```bash
   echo $ANTHROPIC_API_KEY
   ```
2. Verify the key format (typically starts with "sk-ant-").
3. Test the key using Anthropic's API documentation.

### 3.3 Azure API Key

**Issue:** Azure-based services failing to authenticate.

**Solution:**
1. Check the `AZURE_API_KEY` is set:
   ```bash
   echo $AZURE_API_KEY
   ```
2. Verify additional Azure-related variables:
   - `AZURE_API_BASE`
   - `AZURE_API_VERSION`
3. Ensure the key and endpoint match your Azure resource configuration.

### 3.4 Unstructured API Key

**Issue:** File ingestion or parsing failures.

**Solution:**
1. Verify the `UNSTRUCTURED_API_KEY` is set:
   ```bash
   echo $UNSTRUCTURED_API_KEY
   ```
2. Check if the Unstructured API URL is correctly configured:
   ```bash
   echo $UNSTRUCTURED_API_URL
   ```
3. Test the key using Unstructured's API documentation.

### 3.5 Hatchet Client Token

**Issue:** Workflow orchestration failures or Hatchet connectivity issues.

**Solution:**
1. Confirm the `HATCHET_CLIENT_TOKEN` is set:
   ```bash
   echo $HATCHET_CLIENT_TOKEN
   ```
2. Verify the token was correctly generated during the R2R setup process.
3. Check Hatchet logs for any token-related errors.

## 4. Updating API Keys

If you need to update an API key:

1. Stop the R2R service:
   ```bash
   docker-compose down
   ```

2. Update the key in your environment or configuration file:
   ```bash
   export NEW_API_KEY="your-new-key-here"
   ```
   Or update the `r2r.toml` file if you're using configuration files.

3. Restart the R2R service:
   ```bash
   docker-compose up -d
   ```

## 5. Security Best Practices

- Never commit API keys to version control.
- Use environment variables or secure secret management solutions.
- Regularly rotate API keys, especially if you suspect they've been compromised.
- Use the principle of least privilege when creating API keys.

## 6. Debugging API Key Issues

If you're still having trouble:

1. Check R2R logs for detailed error messages:
   ```bash
   docker-compose logs r2r
   ```

2. Verify network connectivity to the API endpoints.

3. Ensure your account has the necessary permissions for the API keys you're using.

4. Try using the API key in a simple curl command to isolate R2R-specific issues:
   ```bash
   curl -H "Authorization: Bearer $YOUR_API_KEY" https://api.example.com/v1/test
   ```

## 7. Getting Help

If you've tried these steps and are still experiencing issues:

1. Check the R2R documentation for any recent changes or known issues with API integrations.
2. Search the R2R GitHub issues for similar problems and solutions.
3. Reach out to the R2R community on Discord or other support channels, providing:
   - R2R version
   - Relevant logs (with sensitive information redacted)
   - Steps to reproduce the issue
   - Any error messages you're seeing

Remember, never share your actual API keys when seeking help. Use placeholders or redacted versions in any logs or code snippets you share publicly.

---

# How to Check R2R Logs and Use Analytics & Observability Features

This guide covers various methods to access and analyze R2R logs, as well as leverage R2R's powerful analytics and observability features. These capabilities allow you to monitor system performance, track usage patterns, and gain valuable insights into your RAG application's behavior.

## 1. Checking R2R Logs

### 1.1 Docker Deployment

If you're running R2R using Docker:

1. List running containers:
   ```bash
   docker ps
   ```

2. View real-time logs:
   ```bash
   docker logs -f <container_id_or_name>
   ```

3. Using Docker Compose:
   ```bash
   docker-compose logs -f r2r
   ```

### 1.2 Local Deployment

For local deployments without Docker:

1. Check the R2R configuration for log file locations.
2. Use standard Unix tools to view logs:
   ```bash
   tail -f /path/to/r2r.log
   ```

### 1.3 Cloud Deployments

- For Azure: Use "Log stream" or "Diagnose and solve problems" in the Azure portal.
- For AWS: Use CloudWatch or check EC2/ECS logs directly.

## 2. Using R2R's Logging and Analytics Features

<Note>
The features described in this section are typically restricted to superusers. Ensure you have the necessary permissions before attempting to access these features.
</Note>

### 2.1 Fetching Logs

You can fetch logs using the client-server architecture:

<Tabs>
<Tab title="CLI">
```bash
r2r logs
```
</Tab>

<Tab title="Python">
```python
client.logs()
```
</Tab>

<Tab title="JavaScript">
```javascript
client.logs()
```
</Tab>

<Tab title="Curl">
```bash
curl -X POST http://localhost:7272/v2/logs \
  -H "Content-Type: application/json" \
  -d '{
    "log_type_filter": null,
    "max_runs_requested": 100
  }'
```
</Tab>
</Tabs>

Expected Output:
```python
[
    {
        'run_id': UUID('27f124ad-6f70-4641-89ab-f346dc9d1c2f'),
        'run_type': 'rag',
        'entries': [
            {'key': 'search_query', 'value': 'Who is Aristotle?'},
            {'key': 'search_latency', 'value': '0.39'},
            {'key': 'search_results', 'value': '["{\\"id\\":\\"7ed3a01c-88dc-5a58-a68b-6e5d9f292df2\\",...}"]'},
            {'key': 'rag_generation_latency', 'value': '3.79'},
            {'key': 'llm_response', 'value': 'Aristotle (Greek: Ἀριστοτέλης Aristotélēs; 384–322 BC) was...'}
        ]
    },
    # More log entries...
]
```

### 2.2 Using Analytics

R2R offers an analytics feature for aggregating and analyzing log data:

<Tabs>
<Tab title="CLI">
```bash
r2r analytics --filters '{"search_latencies": "search_latency"}' --analysis-types '{"search_latencies": ["basic_statistics", "search_latency"]}'
```
</Tab>

<Tab title="Python">
```python
client.analytics(
  {"search_latencies": "search_latency"},
  {"search_latencies": ["basic_statistics", "search_latency"]}
)
```
</Tab>

<Tab title="JavaScript">
```javascript
const filterCriteria = {
    filters: {
      search_latencies: "search_latency",
    },
  };

const analysisTypes = {
  search_latencies: ["basic_statistics", "search_latency"],
};

client.analytics(filterCriteria, analysisTypes);
```
</Tab>

<Tab title="Curl">
```bash
curl -X POST http://localhost:7272/v2/analytics \
  -H "Content-Type: application/json" \
  -d '{
    "filter_criteria": {
      "filters": {
        "search_latencies": "search_latency"
      }
    },
    "analysis_types":
    {
        "analysis_types": {
            "search_latencies": ["basic_statistics", "search_latency"]
        }
    }
  }'
```
</Tab>
</Tabs>

Expected Output:
```python
{
    'results': {
        'filtered_logs': {
            'search_latencies': [
                {
                    'timestamp': '2024-06-20 21:29:06',
                    'log_id': UUID('0f28063c-8b87-4934-90dc-4cd84dda5f5c'),
                    'key': 'search_latency',
                    'value': '0.66',
                    'rn': 3
                },
                ...
            ]
        },
        'search_latencies': {
            'Mean': 0.734,
            'Median': 0.523,
            'Mode': 0.495,
            'Standard Deviation': 0.213,
            'Variance': 0.0453
        }
    }
}
```

## 3. Advanced Analytics and Observability

### 3.1 Custom Analytics

You can specify different filters and analysis types to focus on specific aspects of your application's performance:

```python
from r2r import FilterCriteria, AnalysisTypes

# Analyze RAG latencies
rag_filter = FilterCriteria(filters={"rag_latencies": "rag_generation_latency", "rag_eval": "rag_eval_metric"})
rag_analysis = AnalysisTypes(analysis_types={"rag_latencies": ["basic_statistics", "rag_generation_latency"]})
rag_analytics = app.analytics(rag_filter, rag_analysis)

# Track usage patterns by user
user_filter = FilterCriteria(filters={"user_patterns": "user_id"})
user_analysis = AnalysisTypes(analysis_types={"user_patterns": ["bar_chart", "user_id"]})
user_analytics = app.analytics(user_filter, user_analysis)

# Monitor error rates
error_filter = FilterCriteria(filters={"error_rates": "error"})
error_analysis = AnalysisTypes(analysis_types={"error_rates": ["basic_statistics", "error"]})
error_analytics = app.analytics(error_filter, error_analysis)
```

### 3.2 Preloading Data for Analysis

To get meaningful analytics, you can preload your database with random searches:

```python
import random
from r2r import R2R, GenerationConfig

app = R2R()

queries = [
    "What is artificial intelligence?",
    "Explain machine learning.",
    "How does natural language processing work?",
    "What are neural networks?",
    "Describe deep learning.",
    # Add more queries as needed
]

for _ in range(1000):
    query = random.choice(queries)
    app.rag(query, GenerationConfig(model="openai/gpt-4o-mini"))

print("Preloading complete. You can now run analytics on this data.")
```

### 3.3 User-Level Analytics

To get analytics for a specific user:

```python
user_id = "your_user_id_here"

user_filter = FilterCriteria(filters={"user_analytics": "user_id"})
user_analysis = AnalysisTypes(analysis_types={
    "user_analytics": ["basic_statistics", "user_id"],
    "user_search_latencies": ["basic_statistics", "search_latency"]
})

user_analytics = app.analytics(user_filter, user_analysis)
print(f"Analytics for user {user_id}:")
print(user_analytics)
```

## 4. Log Analysis Tips

1. Look for ERROR or WARNING level logs first.
2. Check timestamps to correlate logs with observed issues.
3. Use tools like `grep`, `awk`, or `sed` to filter logs.
4. For large log files, use `less` with search functionality.

## 5. Log Aggregation Tools

Consider using log aggregation tools for more advanced setups:

1. ELK Stack (Elasticsearch, Logstash, Kibana)
2. Prometheus and Grafana
3. Datadog
4. Splunk

## Summary

R2R's logging, analytics, and observability features provide powerful tools for understanding and optimizing your RAG application. By leveraging these capabilities, you can:

- Monitor system performance in real-time
- Analyze trends in search and RAG operations
- Identify potential bottlenecks or areas for improvement
- Track user behavior and usage patterns
- Make data-driven decisions to enhance your application's performance and user experience

Remember to rotate logs regularly and set up log retention policies to manage disk space, especially in production environments.

For more advanced usage and customization options, consider joining the [R2R Discord community](https://discord.gg/p6KqD2kjtB) or referring to the detailed [R2R documentation](https://r2r-docs.sciphi.ai/).

---


Gets the latest settings for the application logic.

---


Web developers can easily integrate R2R into their projects using the [R2R JavaScript client](https://github.com/SciPhi-AI/r2r-js).
For more extensive reference and examples of how to use the r2r-js library, we encourage you to look at the [R2R Application](/cookbooks/application) and its source code.

## Hello R2R—JavaScript

R2R gives developers configurable vector search and RAG right out of the box, as well as direct method calls instead of the client-server architecture seen throughout the docs:
```python r2r-js/examples/hello_r2r.js

const { r2rClient } = require("r2r-js");

const client = new r2rClient("http://localhost:7272");

async function main() {
  const files = [
    { path: "examples/data/raskolnikov.txt", name: "raskolnikov.txt" },
  ];

  const EMAIL = "admin@example.com";
  const PASSWORD = "change_me_immediately";
  console.log("Logging in...");
  await client.login(EMAIL, PASSWORD);

  console.log("Ingesting file...");
  const ingestResult = await client.ingestFiles(files, {
    metadatas: [{ title: "raskolnikov.txt" }],
    skip_document_info: false,
  });
  console.log("Ingest result:", JSON.stringify(ingestResult, null, 2));

  console.log("Performing RAG...");
  const ragResponse = await client.rag({
    query: "What does the file talk about?",
    rag_generation_config: {
      model: "openai/gpt-4o",
      temperature: 0.0,
      stream: false,
    },
  });

  console.log("Search Results:");
  ragResponse.results.search_results.vector_search_results.forEach(
    (result, index) => {
      console.log(`\nResult ${index + 1}:`);
      console.log(`Text: ${result.metadata.text.substring(0, 100)}...`);
      console.log(`Score: ${result.score}`);
    },
  );

  console.log("\nCompletion:");
  console.log(ragResponse.results.completion.choices[0].message.content);
}

main();
```

## r2r-js Client
### Installing

To get started, install the R2R JavaScript client with [npm](https://www.npmjs.com/package/r2r-js):

<Tabs>
<Tab title="npm">
```bash
npm install r2r-js
```
</Tab>
</Tabs>

### Creating the Client
First, we create the R2R client and specify the base URL where the R2R server is running:

```javascript
const { r2rClient } = require("r2r-js");

// http://localhost:7272 or the address that you are running the R2R server
const client = new r2rClient("http://localhost:7272");
```

### Log into the server
Sign into the server to authenticate the session. We'll use the default superuser credentials:

```javascript
const EMAIL = "admin@example.com";
const PASSWORD = "change_me_immediately";
console.log("Logging in...");
await client.login(EMAIL, PASSWORD);
```

### Ingesting Files
Specify the files that we'll ingest:

```javascript
const files = [
    { path: "examples/data/raskolnikov.txt", name: "raskolnikov.txt" },
];
console.log("Ingesting file...");
    const ingestResult = await client.ingestFiles(files, {
        metadatas: [{ title: "raskolnikov.txt" }],
        skip_document_info: false,
    });
console.log("Ingest result:", JSON.stringify(ingestResult, null, 2));
...
/* Ingest result: {
  "results": {
    "processed_documents": [
      "Document 'raskolnikov.txt' processed successfully."
    ],
    "failed_documents": [],
    "skipped_documents": []
  }
} */
```

This command processes the ingested, splits them into chunks, embeds the chunks, and stores them into your specified Postgres database. Relational data is also stored to allow for downstream document management, which you can read about in the [quickstart](https://r2r-docs.sciphi.ai/quickstart).

### Performing RAG
We'll make a RAG request,

```javascript
console.log("Performing RAG...");
  const ragResponse = await client.rag({
    query: "What does the file talk about?",
    rag_generation_config: {
      model: "openai/gpt-4o",
      temperature: 0.0,
      stream: false,
    },
  });

console.log("Search Results:");
  ragResponse.results.search_results.vector_search_results.forEach(
    (result, index) => {
      console.log(`\nResult ${index + 1}:`);
      console.log(`Text: ${result.metadata.text.substring(0, 100)}...`);
      console.log(`Score: ${result.score}`);
    },
  );

  console.log("\nCompletion:");
  console.log(ragResponse.results.completion.choices[0].message.content);
...
/* Performing RAG...
Search Results:

Result 1:
Text: praeterire culinam eius, cuius ianua semper aperta erat, cogebatur. Et quoties praeteribat,
iuvenis ...
Score: 0.08281802143835804

Result 2:
Text: In vespera praecipue calida ineunte Iulio iuvenis e cenaculo in quo hospitabatur in
S. loco exiit et...
Score: 0.052743945852283036

Completion:
The file discusses the experiences and emotions of a young man who is staying in a small room in a tall house.
He is burdened by debt and feels anxious and ashamed whenever he passes by the kitchen of his landlady, whose
door is always open [1]. On a particularly warm evening in early July, he leaves his room and walks slowly towards
a bridge, trying to avoid encountering his landlady on the stairs. His room, which is more like a closet than a
proper room, is located under the roof of the five-story house, while the landlady lives on the floor below and
provides him with meals and services [2].
*/
```

## Connecting to a Web App
R2R can be easily integrated into web applications. We'll create a simple Next.js app that uses R2R for query answering. [We've created a template repository with this code.](https://github.com/SciPhi-AI/r2r-webdev-template)

Alternatively, you can add the code below to your own Next.js project.

![R2R Dashboard Overview](/images/r2r_webdev_template.png)

### Setting up an API Route

First, we'll create an API route to handle R2R queries. Create a file named `r2r-query.ts` in the `pages/api` directory:

<Accordion title="r2r-query.ts" icon="code">
```typescript
import { NextApiRequest, NextApiResponse } from 'next';
import { r2rClient } from 'r2r-js';

const client = new r2rClient("http://localhost:7272");

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method === 'POST') {
    const { query } = req.body;

    try {
      // Login with each request. In a production app, you'd want to manage sessions.
      await client.login("admin@example.com", "change_me_immediately");

      const response = await client.rag({
        query: query,
        rag_generation_config: {
          model: "openai/gpt-4o",
          temperature: 0.0,
          stream: false,
        }
      });

      res.status(200).json({ result: response.results.completion.choices[0].message.content });
    } catch (error) {
      res.status(500).json({ error: error instanceof Error ? error.message : 'An error occurred' });
    }
  } else {
    res.setHeader('Allow', ['POST']);
    res.status(405).end(`Method ${req.method} Not Allowed`);
  }
}
```
</Accordion>


This API route creates an R2R client, logs in, and processes the incoming query using the RAG method.

### Frontend: React Component

Next, create a React component to interact with the API. Here's an example `index.tsx` file:

<Accordion title="index.tsx" icon="code">
```tsx
import React, { useState } from 'react';
import styles from '@/styles/R2RWebDevTemplate.module.css';

const R2RQueryApp: React.FC = () => {
  const [query, setQuery] = useState('');
  const [result, setResult] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const performQuery = async () => {
    setIsLoading(true);
    setResult('');

    try {
      const response = await fetch('/api/r2r-query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      setResult(data.result);
    } catch (error) {
      setResult(`Error: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={styles.appWrapper}>
      <h1 className={styles.title}>R2R Web Dev Template</h1>
      <p>A simple template for making RAG queries with R2R.
        Make sure that your R2R server is up and running, and that you've ingested files!
      </p>
      <p>
        Check out the <a href="https://r2r-docs.sciphi.ai/" target="_blank" rel="noopener noreferrer">R2R Documentation</a> for more information.
      </p>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Enter your query here"
        className={styles.queryInput}
      />
      <button
        onClick={performQuery}
        disabled={isLoading}
        className={styles.submitButton}
      >
        Submit Query
      </button>
      {isLoading ? (
        <div className={styles.spinner} />
      ) : (
        <div className={styles.resultDisplay}>{result}</div>
      )}
    </div>
  );
};

export default R2RQueryApp;
```
</Accordion>


This component creates a simple interface with an input field for the query and a button to submit it. When the button is clicked, it sends a request to the API route we created earlier and displays the result.

### Template Repository

For a complete working example, you can check out our template repository. This repository contains a simple Next.js app with R2R integration, providing a starting point for your own R2R-powered web applications.

For more advanced examples, check out the [source code for the R2R Dashboard.](https://github.com/SciPhi-AI/R2R-Application)

[R2R Web App Template Repository](https://github.com/SciPhi-AI/r2r-webdev-template)

To use this template:

1. Clone the repository
2. Install dependencies with `pnpm install`
3. Make sure your R2R server is running
4. Start the development server with `pnpm dev`

This template provides a foundation for building more complex applications with R2R, demonstrating how to integrate R2R's powerful RAG capabilities into a web interface.

---


## Introduction to orchestration

R2R uses [Hatchet](https://docs.hatchet.run/home) for orchestrating complex workflows, particularly for ingestion and knowledge graph construction processes.

Hatchet is a distributed, fault-tolerant task queue that solves scaling problems like concurrency, fairness, and rate limiting. It allows R2R to distribute functions between workers with minimal configuration.

### Key Concepts

1. **Workflows**: Sets of functions executed in response to external triggers.
2. **Workers**: Long-running processes that execute workflow functions.
3. **Managed Queue**: Low-latency queue for handling real-time tasks.

## Orchestration in R2R


### Benefits of orchestration

1. **Scalability**: Efficiently handles large-scale tasks.
2. **Fault Tolerance**: Built-in retry mechanisms and error handling.
3. **Flexibility**: Easy to add or modify workflows as R2R's capabilities expand.

### Workflows in R2R

1. **IngestFilesWorkflow**: Handles file ingestion, parsing, chunking, and embedding.
2. **UpdateFilesWorkflow**: Manages the process of updating existing files.
3. **KgExtractAndStoreWorkflow**: Extracts and stores knowledge graph information.
4. **CreateGraphWorkflow**: Orchestrates the creation of knowledge graphs.
5. **EnrichGraphWorkflow**: Handles graph enrichment processes like node creation and clustering.


## Orchestration GUI

By default, the R2R Docker ships with with Hatchet's front-end application on port 7274. This can be accessed by navigating to `http://localhost:7274`.

You may login with the following credentials:


<Note>

**Email:** admin@example.com

**Password:** Admin123!!
</Note>

### Login

<Frame caption="Logging into hatchet at http://localhost:7274">
  <img src="../images/hatchet_login.png" />
</Frame>


### Running Tasks

The panel below shows the state of the Hatchet workflow panel at `http://localhost:7274/workflow-runs` immediately after calling `r2r ingest-sample-files`:

<Frame caption="Running workflows at http://localhost:7274/workflow-runs">
  <img src="../images/hatchet_running.png" />
</Frame>


### Inspecting a workflow

You can inspect a workflow within Hatchet and can even attempt to retry the job from directly in the GUI in the case of failure:

<Frame caption="Inspecting a workflow at http://localhost:7274/workflow-runs/274081a8-acfb-4686-84c9-9fd73bc5c7f1?tenant=707d0855-80ab-4e1f-a156-f1c4546cbf52">
  <img src="../images/hatchet_workflow.png" />
</Frame>



### Long running tasks

Hatchet supports long running tasks, which is very useful during knowledge graph construction:

<Frame caption="Worker timeout is set to 60m to support long running tasks like graph construction.">
  <img src="../images/hatchet_long_running.png" />
</Frame>



## Coming Soon

In the coming day(s) / week(s) we will further highlight the available feature set and best practices for orchestrating your ingestion workflows inside R2R.

---


R2R offers an [open-source React+Next.js application](https://github.com/SciPhi-AI/R2R-Application) designed to give developers an administrative portal for their R2R deployment, and users an application to communicate with out of the box.

## Setup

### Install PNPM

PNPM is a fast, disk space-efficient package manager. To install PNPM, visit the [official PNPM installation page](https://pnpm.io/installation) or follow these instructions:

<AccordionGroup>

<Accordion icon="terminal" title="PNPM Installation">
For Unix-based systems (Linux, macOS):

```bash
curl -fsSL https://get.pnpm.io/install.sh | sh -
```

For Windows:

```powershell
iwr https://get.pnpm.io/install.ps1 -useb | iex
```

After installation, you may need to add PNPM to your system's PATH.
</Accordion>

</AccordionGroup>

### Installing and Running the R2R Dashboard

If you're running R2R with the Docker, you already have the R2R application running! Just navigate to [http://localhost:7273](http://localhost:7273).

If you're running R2R outside of Docker, run the following commands to install the R2R Dashboard.

1. Clone the project repository and navigate to the project directory:

```bash
git clone git@github.com:SciPhi-AI/R2R-Application.git
cd R2R-Application
```

2. Install the project dependencies:

```bash
pnpm install
```

3. Build and start the application for production:

```bash
pnpm build
pnpm start
```

The dashboard will be available at [http://localhost:3000](http://localhost:3000).

## Features

### Login

To interact with R2R with the dashboard, you must first login. If it's your first time logging in, log in with the default credentials shown.

By default, an R2R instance is hosted on port 7272. The login page will include this URL by default, but be sure to update the URL if your R2R instance is deployed elsewhere. For information about deploying a local R2R application server, see the [quickstart](/documentation/quickstart).

![R2R Dashboard Overview](/images/login.png)

### Documents

The documents page provides an overview of uploaded documents and their metadata. You can upload new documents and update, download, or delete existing ones. Additionally, you can view information about each document, including the documents' chunks and previews of PDFs.

![Documents Page](/images/oss_dashboard_documents.png)

### Collections

Collections allow users to create and share sets of documents. The collections page provides a place to manage your existing collections or create new collections.

![Collections Page](/images/oss_collections_page.png)

### Chat

In the chat page, you can stream RAG responses with different models and configurable settings. You can interact with both the RAG Agent and RAG endpoints here.

![Chat Interface](/images/chat.png)

### Users

Manage your users and gain insight into their interactions.

![Users Page](/images/users.png)

### Logs

The Logs page enables tracking of user queries, search results, and LLM responses.

![Logs Page](/images/logs.png)

### Settings

The settings page allows you to view the configuration of and edit the prompts associated with your R2R deployment.

![Logs Page](/images/settings_config.png)
![Logs Page](/images/settings_prompts.png)

## Development

To develop the R2R dashboard:

1. Start the development server:

```bash
pnpm dev
```

2. Run pre-commit checks (optional but recommended):

```bash
pnpm format
pnpm lint
```

---



## Introduction

This guide demonstrates how to leverage R2R's powerful analytics and logging features. These capabilities allow you to monitor system performance, track usage patterns, and gain valuable insights into your RAG application's behavior.

<Note>
The features described in this cookbook are typically restricted to superusers. Ensure you have the necessary permissions before attempting to access these features.
</Note>

For more information on user roles and permissions, including how to set up and manage superuser accounts, please refer to our [User Auth Cookbook](/cookbooks/user-auth).

## Setup

Before diving into the authentication features, ensure you have R2R installed and configured as described in the [installation guide](/documentation/installation). For this guide, we'll use the default configuration. Further, `r2r serve` must be called to serve R2R in either your local environment or local Docker engine.

## Basic Usage

### Logging

R2R automatically logs various events and metrics during its operation.


To fetch our logs using the client-server architecture, use the following:


<Tabs>
<Tab title="CLI">
```bash
r2r logs
```
</Tab>

<Tab title="Python">
```python
client.logs()
```
</Tab>

<Tab title="JavaScript">
```javascript
client.logs()
```
</Tab>

<Tab title="Curl">

```bash
curl -X POST http://localhost:7272/v2/logs \
  -H "Content-Type: application/json" \
  -d '{
    "log_type_filter": null,
    "max_runs_requested": 100
  }'
```
</Tab>

</Tabs>

Expected Output:
```python
[
    {
        'run_id': UUID('27f124ad-6f70-4641-89ab-f346dc9d1c2f'),
        'run_type': 'rag',
        'entries': [
            {'key': 'search_query', 'value': 'Who is Aristotle?'},
            {'key': 'search_latency', 'value': '0.39'},
            {'key': 'search_results', 'value': '["{\\"id\\":\\"7ed3a01c-88dc-5a58-a68b-6e5d9f292df2\\",...}"]'},
            {'key': 'rag_generation_latency', 'value': '3.79'},
            {'key': 'llm_response', 'value': 'Aristotle (Greek: Ἀριστοτέλης Aristotélēs; 384–322 BC) was...'}
        ]
    },
    # More log entries...
]
```

These logs provide detailed information about each operation, including search results, queries, latencies, and LLM responses.


To fetch the logs directly from an instantiated R2R object:

```python
app = R2R()

# Perform some searches / RAG completions
# ...

# Get the latest logs
logs = app.logs()
print(logs)
```
### Analytics

R2R offers an analytics feature that allows you to aggregate and analyze log data:


The relevant command
<Tabs>
<Tab title="CLI">
```bash
r2r analytics --filters '{"search_latencies": "search_latency"}' --analysis-types '{"search_latencies": ["basic_statistics", "search_latency"]}'
```
</Tab>

<Tab title="Python">
```python
client.analytics(
  {"search_latencies": "search_latency"},
  {"search_latencies": ["basic_statistics", "search_latency"]}
)
```
</Tab>

<Tab title="JavaScript">
```javascript
const filterCriteria = {
    filters: {
      search_latencies: "search_latency",
    },
  };

  const analysisTypes = {
    search_latencies: ["basic_statistics", "search_latency"],
  };

  client.analytics(filterCriteria, analysisTypes);
```
</Tab>

<Tab title="Curl">

```bash
curl -X POST http://localhost:7272/v2/analytics \
  -H "Content-Type: application/json" \
  -d '{
    "filter_criteria": {
      "filters": {
        "search_latencies": "search_latency"
      }
    },
    "analysis_types":
    {
        "analysis_types": {
            "search_latencies": ["basic_statistics", "search_latency"]
        }
    }
  }'
```
</Tab>


</Tabs>


Expected Output:
```python
{
    'results': {
        'filtered_logs': {
            'search_latencies': [
                {
                    'timestamp': '2024-06-20 21:29:06',
                    'log_id': UUID('0f28063c-8b87-4934-90dc-4cd84dda5f5c'),
                    'key': 'search_latency',
                    'value': '0.66',
                    'rn': 3
                },
                ...
            ]
        },
        'search_latencies': {
            'Mean': 0.734,
            'Median': 0.523,
            'Mode': 0.495,
            'Standard Deviation': 0.213,
            'Variance': 0.0453
        }
    }
}
```


To fetch the analytics directly from an instantiated R2R object:

```python
from r2r import FilterCriteria, AnalysisTypes

filter_criteria = FilterCriteria(filters={"search_latencies": "search_latency"})
analysis_types = AnalysisTypes(analysis_types={"search_latencies": ["basic_statistics", "search_latency"]})

analytics_results = app.analytics(filter_criteria, analysis_types)
print(analytics_results)
```
The boilerplate analytics implementation allows you to:
    1. Filter logs based on specific criteria
    2. Perform statistical analysis on various metrics (e.g., search latencies)
    3. Track performance trends over time
    4. Identify potential bottlenecks or areas for optimization


## Experimental Features

<Warning>
Advanced analytics features are still in an experimental state - please reach out to the R2R team if you are interested in configuring / using these additional features.
</Warning>

### Custom Analytics

R2R's analytics system is flexible and allows for custom analysis. You can specify different filters and analysis types to focus on specific aspects of your application's performance.

```python
# Analyze RAG latencies
rag_filter = FilterCriteria(filters={"rag_latencies": "rag_generation_latency", "rag_eval": "rag_eval_metric"})
rag_analysis = AnalysisTypes(analysis_types={"rag_latencies": ["basic_statistics", "rag_generation_latency"]})
rag_analytics = app.analytics(rag_filter, rag_analysis)

# Track usage patterns by user
user_filter = FilterCriteria(filters={"user_patterns": "user_id"})
user_analysis = AnalysisTypes(analysis_types={"user_patterns": ["bar_chart", "user_id"]})
user_analytics = app.analytics(user_filter, user_analysis)

# Monitor error rates
error_filter = FilterCriteria(filters={"error_rates": "error"})
error_analysis = AnalysisTypes(analysis_types={"error_rates": ["basic_statistics", "error"]})
error_analytics = app.analytics(error_filter, error_analysis)
```

### Preloading Data for Analysis

To get meaningful analytics, you need a substantial amount of data. Here's a script to preload your database with random searches:

```python
import random
from r2r import R2R, GenerationConfig

app = R2R()

# List of sample queries
queries = [
    "What is artificial intelligence?",
    "Explain machine learning.",
    "How does natural language processing work?",
    "What are neural networks?",
    "Describe deep learning.",
    # Add more queries as needed
]

# Perform random searches
for _ in range(1000):
    query = random.choice(queries)
    app.rag(query, GenerationConfig(model="openai/gpt-4o-mini"))

print("Preloading complete. You can now run analytics on this data.")
```

After running this script, you'll have a rich dataset to analyze using the analytics features described above.

### User-Level Analytics

To get analytics for a specific user:

```python
user_id = "your_user_id_here"

user_filter = FilterCriteria(filters={"user_analytics": "user_id"})
user_analysis = AnalysisTypes(analysis_types={
    "user_analytics": ["basic_statistics", "user_id"],
    "user_search_latencies": ["basic_statistics", "search_latency"]
})

user_analytics = app.analytics(user_filter, user_analysis)
print(f"Analytics for user {user_id}:")
print(user_analytics)
```

This will give you insights into the behavior and performance of specific users in your system.

## Summary

R2R's logging and analytics features provide powerful tools for understanding and optimizing your RAG application. By leveraging these capabilities, you can:

- Monitor system performance in real-time
- Analyze trends in search and RAG operations
- Identify potential bottlenecks or areas for improvement
- Track user behavior and usage patterns
- Make data-driven decisions to enhance your application's performance and user experience

For detailed setup and basic functionality, refer back to the [R2R Quickstart](/documentation/quickstart). For more advanced usage and customization options, join the [R2R Discord community](https://discord.gg/p6KqD2kjtB).

---


## Introduction

R2R provides a powerful and flexible ingestion pipeline that allows you to efficiently process and manage various types of documents. This cookbook will guide you through the process of ingesting files, updating existing documents, and deleting documents using the R2R Python SDK.

<Note>

As of version `3.2.13`, we have expanded the options for ingesting files using multimodal foundation models. In addition to using such models by default for images, R2R can now use them on PDFs by passing the following in your ingestion configuration:

```json
"ingestion_config": {
  ...,
  "parser_overrides": {
    "pdf": "zerox"
  }
}
```

We recommend this method for achieving the highest quality ingestion results.

</Note>


## Ingesting Files

To ingest files into your R2R system, you can use the `ingest_files` method from the Python SDK:

```python
file_paths = ['path/to/file1.txt', 'path/to/file2.txt']
metadatas = [{'key1': 'value1'}, {'key2': 'value2'}]

ingest_response = client.ingest_files(
    file_paths=file_paths,
    metadatas=metadatas,
    ingestion_config={
        "provider": "unstructured_local",
        "strategy": "auto",
        "chunking_strategy": "by_title",
        "new_after_n_chars": 256,
        "max_characters": 512,
        "combine_under_n_chars": 64,
        "overlap": 100,
    }
)
```

The `ingest_files` method accepts the following parameters:

- `file_paths` (required): A list of file paths or directory paths to ingest.
- `metadatas` (optional): A list of metadata dictionaries corresponding to each file.
- `document_ids` (optional): A list of document IDs to assign to the ingested files.
- `ingestion_config` (optional): Custom ingestion settings to override the default configuration, which you can read more about [here](/documentation/configuration/ingestion/overview).

## Ingesting Chunks

If you have pre-processed chunks of text, you can directly ingest them using the `ingest_chunks` method:

```python
chunks = [
    {"text": "This is the first chunk."},
    {"text": "This is the second chunk."}
]

ingest_response = client.ingest_chunks(
    chunks=chunks,
    document_id="custom_document_id",
    metadata={"custom_metadata": "value"},
)
```

The `ingest_chunks` method accepts the following parameters:

- `chunks` (required): A list of dictionaries containing the text and metadata for each chunk.
- `document_id` (optional): A custom document ID to assign to the ingested chunks.
- `metadata` (optional): Additional metadata to associate with the ingested chunks.

## Updating Files

To update existing documents in your R2R system, you can use the `update_files` method:

```python
file_paths = ['path/to/updated_file1.txt', 'path/to/updated_file2.txt']
document_ids = ['document1_id', 'document2_id']

update_response = client.update_files(
    file_paths=file_paths,
    document_ids=document_ids,
    metadatas=[{"version": "2.0"}, {"version": "1.5"}],
)
```

The `update_files` method accepts the following parameters:

- `file_paths` (required): A list of file paths for the updated documents.
- `document_ids` (required): A list of document IDs corresponding to the files being updated.
- `metadatas` (optional): A list of metadata dictionaries to update for each document.


## Updating Chunks

To update specific chunks within existing documents in your R2R deployment, you can use the `update_chunks` method:

```python
document_id = "9fbe403b-c11c-5aae-8ade-ef22980c3ad1"
extraction_id = "aeba6400-1bd0-5ee9-8925-04732d675434"

update_response = client.update_chunks(
    document_id=document_id,
    extraction_id=extraction_id,
    text="Updated chunk content with new information...",
    metadata={
        "source": "manual_edit",
        "edited_at": "2024-10-24",
        "editor": "John Doe"
    }
)
```

The `update_chunks` method accepts the following parameters:

- `document_id` (required): The ID of the document containing the chunk you want to update.
- `extraction_id` (required): The ID of the specific chunk you want to update.
- `text` (required): The new text content that will replace the existing chunk text.
- `metadata` (optional): A metadata dictionary that will replace the existing chunk metadata.
- `run_with_orchestration` (optional): Whether to run the update through orchestration (default: true).

This method is particularly useful when you need to:
- Correct errors in specific chunks
- Update outdated information
- Add or modify metadata for individual chunks
- Make targeted changes without reprocessing entire documents

Note that updating chunks will trigger a re-vectorization of the modified content, ensuring that your vector search capabilities remain accurate with the updated information.


## Deleting Documents and Chunks

To delete documents or chunks from your R2R deployment, you can use the `delete` method:

```python
# For documents
delete_response = client.delete(
    {
        "document_id": {"$eq": "document1_id"}
    }
)

# For chunks
delete_response = client.delete(
    {
        "extraction_id": {"$eq": "extraction1_id"}
    }
)
```

The `delete` method accepts a dictionary specifying the filters to identify the documents to delete. In this example, it deletes the document with the ID "document1_id" and the chunk with the ID "extraction1_id."

## Conclusion

R2R's ingestion pipeline provides a flexible and efficient way to process, update, and manage your documents. By utilizing the `ingest_files`, `ingest_chunks`, `update_files`, and `delete` methods from the Python SDK, you can seamlessly integrate document management capabilities into your applications.

For more detailed information on the available parameters and response formats, refer to the [Python SDK Ingestion Documentation](/documentation/python-sdk/ingestion).

---



## Advanced GraphRAG Techniques

R2R supports advanced GraphRAG techniques that can be easily configured at runtime. This flexibility allows you to experiment with different SoTA strategies and optimize your RAG pipeline for specific use cases.

<Note>

Advanced GraphRAG techniques are still a beta feature in R2R.There may be limitations in observability and analytics when implementing them.

Are we missing an important technique? If so, then please let us know at founders@sciphi.ai.

</Note>


### Prompt Tuning

One way that we can improve upon GraphRAG's already impressive capabilities by tuning our prompts to a specific domain. When we create a knowledge graph, an LLM extracts the relationships between entities; but for very targeted domains, a general approach may fall short.

To demonstrate this, we can run GraphRAG over the technical papers for the 2024 Nobel Prizes in chemistry, medicine, and physics. By tuning our prompts for GraphRAG, we attempt to understand our documents at a high level, and provide the LLM with a more pointed description.

The following script, which utilizes the Python SDK, generates the tuned prompts and calls the knowledge graph creation process with these prompts at runtime:

```python
# Step 1: Tune the prompts for knowledge graph creation
# Tune the entity description prompt
entity_prompt_response = client.get_tuned_prompt(
    prompt_name="graphrag_entity_description"
)
tuned_entity_prompt = entity_prompt_response['results']['tuned_prompt']

# Tune the triples extraction prompt
triples_prompt_response = client.get_tuned_prompt(
    prompt_name="graphrag_triples_extraction_few_shot"
)
tuned_triples_prompt = triples_prompt_response['results']['tuned_prompt']

# Step 2: Create the knowledge graph
kg_settings = {
    "kg_entity_description_prompt": tuned_entity_prompt
}

# Generate the initial graph
graph_response = client.create_graph(
    run_type="run",
    kg_creation_settings=kg_settings
)

# Step 3: Clean up the graph by removing duplicate entities
client.deduplicate_entities(
    run_type="run",
    collection_id='122fdf6a-e116-546b-a8f6-e4cb2e2c0a09'
)

# Step 4: Tune and apply community reports prompt for graph enrichment
community_prompt_response = client.get_tuned_prompt(
    prompt_name="graphrag_community_reports"
)
tuned_community_prompt = community_prompt_response['results']['tuned_prompt']

# Configure enrichment settings
kg_enrichment_settings = {
    "community_reports_prompt": tuned_community_prompt
}

# Enrich the graph with additional information
client.enrich_graph(
    run_type="run",
    kg_enrichment_settings=kg_enrichment_settings
)
```

For illustrative purposes, we look can look at the `graphrag_entity_description` prompt before and after prompt tuning. It's clear that with prompt tuning, we are able to capture the intent of the documents, giving us a more targeted prompt overall.

<Tabs>
<Tab title="Prompt after Prompt Tuning">
```yaml
Provide a comprehensive yet concise summary of the given entity, incorporating its description and associated triples:

Entity Info:
{entity_info}
Triples:
{triples_txt}

Your summary should:
1. Clearly define the entity's core concept or purpose
2. Highlight key relationships or attributes from the triples
3. Integrate any relevant information from the existing description
4. Maintain a neutral, factual tone
5. Be approximately 2-3 sentences long

Ensure the summary is coherent, informative, and captures the essence of the entity within the context of the provided information.
```

</Tab>

<Tab title="Prompt after Prompt Tuning">
```yaml
Provide a comprehensive yet concise summary of the given entity, focusing on its significance in the field of scientific research, while incorporating its description and associated triples:

Entity Info:
{entity_info}
Triples:
{triples_txt}

Your summary should:
1. Clearly define the entity's core concept or purpose within computational biology, artificial intelligence, and medicine
2. Highlight key relationships or attributes from the triples that illustrate advancements in scientific understanding and reasoning
3. Integrate any relevant information from the existing description, particularly breakthroughs and methodologies
4. Maintain a neutral, factual tone
5. Be approximately 2-3 sentences long

Ensure the summary is coherent, informative, and captures the essence of the entity within the context of the provided information, emphasizing its impact on the field.
```
</Tab>

</Tabs>

After prompt tuning, we see an increase in the number of communities—after prompt tuning, these communities appear more focused and domain-specific with clearer thematic boundaries.

Prompt tuning produces:
- **More precise community separation:** GraphRAG alone produced a single `MicroRNA Research` Community, which GraphRAG with prompt tuning produced communities around `C. elegans MicroRNA Research`, `LET-7 MicroRNA`, and `miRNA-184 and EDICT Syndrome`.
- **Enhanced domain focus:** Previously, we had a single community for `AI Researchers`, but with prompt tuning we create specialized communities such as `Hinton, Hopfield, and Deep Learning`, `Hochreiter and Schmidhuber`, and `Minksy and Papert's ANN Critique.`

| Count       | GraphRAG | GraphRAG with Prompt Tuning |
|-------------|----------|-----------------------------|
| Entities    | 661      | 636                         |
| Triples     | 509      | 503                         |
| Communities | 29       | 41                          |

Prompt tuning allow for us to generate communities that better reflect the natural organization of the domain knowledge while maintaining more precise technical and thematic boundaries between related concepts.

## Contextual Chunk Enrichment

Contextual chunk enrichment is a technique that allows us to capture the semantic meaning of the entities and relationships in the knowledge graph. This is done by using a combination of the entity's textual description and its contextual embeddings. This enrichment process enhances the quality and depth of information in your knowledge graph by:

1. Analyzing the surrounding context of each entity mention
2. Incorporating semantic information from related passages
3. Preserving important contextual nuances that might be lost in simple entity extraction

You can learn more about contextual chunk enrichment [here](/cookbooks/contextual-enrichment).


### Entity Deduplication

When creating a knowledge graph across multiple documents, entities are initially created at the document level. This means that the same real-world entity (e.g., "Albert Einstein" or "CRISPR") might appear multiple times if it's mentioned in different documents. This duplication can lead to:

- Redundant information in your knowledge graph
- Fragmented relationships across duplicate entities
- Increased storage and processing overhead
- Potentially inconsistent entity descriptions

The `deduplicate-entities` endpoint addresses these issues by:
1. Identifying similar entities using name (exact match, other strategies coming soon)
2. Merging their properties and relationships
3. Maintaining the most comprehensive description
4. Removing the duplicate entries

<Tabs>
<Tab title="CLI">
```bash
r2r deduplicate-entities --collection-id=122fdf6a-e116-546b-a8f6-e4cb2e2c0a09 --run

# Example Response
[{'message': 'Deduplication task queued successfully.', 'task_id': 'd9dae1bb-5862-4a16-abaf-5297024df390'}]
```
</Tab>

<Tab title="SDK">
```python
from r2r import R2RClient

client = R2RClient("http://localhost:7272")
client.deduplicate_entities(
    collection_id="122fdf6a-e116-546b-a8f6-e4cb2e2c0a09",
    run_type="run"
)

# Example Response
[{'message': 'Deduplication task queued successfully.', 'task_id': 'd9dae1bb-5862-4a16-abaf-5297024df390'}]
```
</Tab>
</Tabs>

#### Monitoring Deduplication

You can monitor the deduplication process in two ways:

1. **Hatchet Dashboard**: Access the dashboard at http://localhost:7274 to view:
   - Task status and progress
   - Any errors or warnings
   - Completion time estimates

2. **API Endpoints**: Once deduplication is complete, verify the results using these endpoints with `entity_level = collection`:
   - [Entities API](http://localhost:7272/v2/entities?collection_id=122fdf6a-e116-546b-a8f6-e4cb2e2c0a09&entity_level=collection)
   - [Triples API](http://localhost:7272/v2/triples?collection_id=122fdf6a-e116-546b-a8f6-e4cb2e2c0a09&entity_level=collection)

#### Best Practices

When using entity deduplication:

- Run deduplication after initial graph creation but before any enrichment steps
- Monitor the number of entities before and after to ensure expected reduction
- Review a sample of merged entities to verify accuracy
- For large collections, expect the process to take longer and plan accordingly

---


## Introduction

A collection in R2R is a logical grouping of users and documents that allows for efficient access control and organization. Collections enable you to manage permissions and access to documents at a group level, rather than individually.

R2R provides robust document collection management, allowing developers to implement efficient access control and organization of users and documents. This cookbook will guide you through the collection capabilities in R2R.

For user authentication, please refer to the [User Auth Cookbook](/cookbooks/user-auth).

<Note>
Collection permissioning in R2R is still under development and as a result the is likely to API continue evolving in future releases.
</Note>


```mermaid
graph TD
    A[User] --> B(Authentication)
    B --> C{Authenticated?}
    C -->|Yes| D[Authenticated User]
    C -->|No| E[Guest User]
    D --> F[Collection Management]
    D --> G[Document Management]
    D --> H[User Profile Management]
    G --> I[CRUD Operations]
    G --> J[Search & RAG]
    D --> K[Logout]
    L[Admin] --> M[Superuser Authentication]
    M --> N[Superuser Capabilities]
    N --> O[User Management]
    N --> P[System-wide Document Access]
    N --> Q[Analytics & Observability]
    N --> R[Configuration Management]
```
_A diagram showing user and collection management across r2r_

## Basic Usage

<Info>
Collections currently follow a flat hierarchy wherein superusers are responsible for management operations. This functionality will expand as development on R2R continues.
</Info>

### Creating a Collection

Let's start by creating a new collection:

```python
from r2r import R2RClient

client = R2RClient("http://localhost:7272")  # Replace with your R2R deployment URL

# Assuming you're logged in as an admin or a user with appropriate permissions
# For testing, the default R2R implementation will grant superuser privileges to anon api calls
collection_result = client.create_collection("Marketing Team", "Collection for marketing department")

print(f"Collection creation result: {collection_result}")
# {'results': {'collection_id': '123e4567-e89b-12d3-a456-426614174000', 'name': 'Marketing Team', 'description': 'Collection for marketing department', 'created_at': '2024-07-16T22:53:47.524794Z', 'updated_at': '2024-07-16T22:53:47.524794Z'}}
```

### Getting Collection details

To retrieve details about a specific collection:

```python
collection_id = '123e4567-e89b-12d3-a456-426614174000'  # Use the collection_id from the creation result
collection_details = client.get_collection(collection_id)

print(f"Collection details: {collection_details}")
# {'results': {'collection_id': '123e4567-e89b-12d3-a456-426614174000', 'name': 'Marketing Team', 'description': 'Collection for marketing department', 'created_at': '2024-07-16T22:53:47.524794Z', 'updated_at': '2024-07-16T22:53:47.524794Z'}}
```

### Updating a Collection

You can update a collection's name or description:

```python
update_result = client.update_collection(collection_id, name="Updated Marketing Team", description="New description for marketing team")

print(f"Collection update result: {update_result}")
# {'results': {'collection_id': '123e4567-e89b-12d3-a456-426614174000', 'name': 'Updated Marketing Team', 'description': 'New description for marketing team', 'created_at': '2024-07-16T22:53:47.524794Z', 'updated_at': '2024-07-16T23:15:30.123456Z'}}
```

### Listing Collections

To get a list of all collections:

```python
collections_list = client.list_collections()

print(f"Collections list: {collections_list}")
# {'results': [{'collection_id': '123e4567-e89b-12d3-a456-426614174000', 'name': 'Updated Marketing Team', 'description': 'New description for marketing team', 'created_at': '2024-07-16T22:53:47.524794Z', 'updated_at': '2024-07-16T23:15:30.123456Z'}, ...]}
```

## User Management in Collections

### Adding a User to a Collection

To add a user to a collection, you need both the user's ID and the collections's ID:

```python
user_id = '456e789f-g01h-34i5-j678-901234567890'  # This should be a valid user ID
add_user_result = client.add_user_to_collection(user_id, collection_id)

print(f"Add user to collection result: {add_user_result}")
# {'results': {'message': 'User successfully added to the collection'}}
```

### Removing a User from a Collections

Similarly, to remove a user from a collection:

```python
remove_user_result = client.remove_user_from_collection(user_id, collection_id)

print(f"Remove user from collection result: {remove_user_result}")
# {'results': None}
```

### Listing Users in a Collection

To get a list of all users in a specific collection:

```python
users_in_collection = client.get_users_in_collection(collection_id)

print(f"Users in collection: {users_in_collection}")
# {'results': [{'user_id': '456e789f-g01h-34i5-j678-901234567890', 'email': 'user@example.com', 'name': 'John Doe', ...}, ...]}
```

### Getting Collections for a User

To get all collections that a user is a member of:

```python
user_collections = client.user_collections(user_id)

print(f"User's collections: {user_collections}")
# {'results': [{'collection_id': '123e4567-e89b-12d3-a456-426614174000', 'name': 'Updated Marketing Team', ...}, ...]}
```

## Document Management in Collections

### Assigning a Document to a Collection

To assign a document to a collection:

```python
document_id = '789g012j-k34l-56m7-n890-123456789012'  # This should be a valid document ID
assign_doc_result = client.assign_document_to_collection(document_id, collection_id)

print(f"Assign document to collection result: {assign_doc_result}")
# {'results': {'message': 'Document successfully assigned to the collection'}}
```

### Removing a Document from a Collection

To remove a document from a collection:

```python
remove_doc_result = client.remove_document_from_collection(document_id, collection_id)

print(f"Remove document from collection result: {remove_doc_result}")
# {'results': {'message': 'Document successfully removed from the collection'}}
```

### Listing Documents in a Collection

To get a list of all documents in a specific collection:

```python
docs_in_collection = client.documents_in_collection(collection_id)

print(f"Documents in collection: {docs_in_collection}")
# {'results': [{'document_id': '789g012j-k34l-56m7-n890-123456789012', 'title': 'Marketing Strategy 2024', ...}, ...]}
```

### Getting Collections for a Document

To get all collections that a document is assigned to:

```python
document_collections = client.document_collections(document_id)

print(f"Document's collections: {document_collections}")
# {'results': [{'collection_id': '123e4567-e89b-12d3-a456-426614174000', 'name': 'Updated Marketing Team', ...}, ...]}
```

## Advanced Collection Management

### Collection Overview

To get an overview of collection, including user and document counts:

```python
collections_overview = client.collections_overview()

print(f"Collections overview: {collections_overview}")
# {'results': [{'collection_id': '123e4567-e89b-12d3-a456-426614174000', 'name': 'Updated Marketing Team', 'description': 'New description for marketing team', 'user_count': 5, 'document_count': 10, ...}, ...]}
```

### Deleting a Collection

To delete a collection:

```python
delete_result = client.delete_collection(collection_id)

print(f"Delete collection result: {delete_result}")
# {'results': {'message': 'Collection successfully deleted'}}
```

## Pagination and Filtering

Many of the collection-related methods support pagination and filtering. Here are some examples:

```python
# List collections with pagination
paginated_collection = client.list_collections(offset=10, limit=20)

# Get users in a collection with pagination
paginated_users = client.get_users_in_collection(collection_id, offset=5, limit=10)

# Get documents in a collection with pagination
paginated_docs = client.documents_in_collection(collection_id, offset=0, limit=50)

# Get collections overview with specific collection IDs
specific_collections_overview = client.collections_overview(collection_ids=['id1', 'id2', 'id3'])
```

## Security Considerations

When implementing collection permissions, consider the following security best practices:

1. **Least Privilege Principle**: Assign the minimum necessary permissions to users and collections.
2. **Regular Audits**: Periodically review collection memberships and document assignments.
3. **Access Control**: Ensure that only authorized users (e.g., admins) can perform collection management operations.
4. **Logging and Monitoring**: Implement comprehensive logging for all collection-related actions.

## Customizing Collection Permissions

While R2R's current collection system follows a flat hierarchy, you can build more complex permission structures on top of it:

1. **Custom Roles**: Implement application-level roles within collections (e.g., collection admin, editor, viewer).
2. **Hierarchical Collections**: Create a hierarchy by establishing parent-child relationships between collections in your application logic.
3. **Permission Inheritance**: Implement rules for permission inheritance based on collection memberships.

## Troubleshooting

Here are some common issues and their solutions:

1. **Unable to Create/Modify Collections**: Ensure the user has superuser privileges.
2. **User Not Seeing Collection Content**: Verify that the user is correctly added to the collection and that documents are properly assigned.
3. **Performance Issues with Large Collections**: Use pagination when retrieving users or documents in large collections.

## Conclusion

R2R's collection permissioning system provides a foundation for implementing sophisticated access control in your applications. As the feature set evolves, more advanced capabilities will become available. Stay tuned to the R2R documentation for updates and new features related to collection permissions.

For user authentication and individual user management, refer to the [User Auth Cookbook](/cookbooks/user-auth). For more advanced use cases or custom implementations, consult the R2R documentation or reach out to the community for support.

---



# Advanced RAG Techniques

R2R supports advanced Retrieval-Augmented Generation (RAG) techniques that can be easily configured at runtime. This flexibility allows you to experiment with different SoTA strategies and optimize your RAG pipeline for specific use cases. **This cookbook will cover toggling between vanilla RAG, [HyDE](https://arxiv.org/abs/2212.10496) and [RAG-Fusion](https://arxiv.org/abs/2402.03367).**.

<Note>

Advanced RAG techniques are still a beta feature in R2R. They are not currently supported in agentic workflows and there may be limitations in observability and analytics when implementing them.


Are we missing an important RAG technique? If so, then please let us know at founders@sciphi.ai.

</Note>

## Advanced RAG in R2R
R2R is designed from the ground up to make it easy to implement advanced RAG techniques. Its modular architecture, based on orchestrated pipes and pipelines, allows for easy customization and extension. A generic implementation diagram of the system is shown below:

```mermaid
graph TD
    A[User Query] --> B[QueryTransformPipe]
    B --> C[MultiSearchPipe]
    C --> G[VectorSearchPipe]
    F[Postgres DB] --> G
    C --> D[GraphSearchPipe]
    H[GraphDB DB] --> D

    D --> E[RAG Generation]
    G --> E[RAG Generation]
    A --> E

```

## Supported Advanced RAG Techniques

R2R currently supports two advanced RAG techniques:

1. **HyDE (Hypothetical Document Embeddings)**: Enhances retrieval by generating and embedding hypothetical documents based on the query.
2. **RAG-Fusion**: Improves retrieval quality by combining results from multiple search iterations.

## Using Advanced RAG Techniques

You can specify which advanced RAG technique to use by setting the `search_strategy` parameter in your vector search settings. Below is a comprehensive overview of techniques supported by R2R.

### HyDE

#### What is HyDE?

HyDE is an innovative approach that supercharges dense retrieval, especially in zero-shot scenarios. Here's how it works:

1. **Query Expansion**: HyDE uses a Language Model to generate hypothetical answers or documents based on the user's query.
2. **Enhanced Embedding**: These hypothetical documents are embedded, creating a richer semantic search space.
3. **Similarity Search**: The embeddings are used to find the most relevant actual documents in your database.
4. **Informed Generation**: The retrieved documents and original query are used to generate the final response.

#### Implementation Diagram



The diagram which follows below illustrates the HyDE flow which fits neatly into the schema of our diagram above (note, the GraphRAG workflow is omitted for brevity):

```mermaid

graph TD
    A[User Query] --> B[QueryTransformPipe]
    B -->|Generate Hypothetical Documents| C[MultiSearchPipe]
    C --> D[VectorSearchPipe]
    D --> E[RAG Generation]
    A --> E
    F[Document DB] --> D

    subgraph HyDE Process
    B --> G[Hypothetical Doc 1]
    B --> H[Hypothetical Doc 2]
    B --> I[Hypothetical Doc n]
    G --> J[Embed]
    H --> J
    I --> J
    J --> C
    end

    subgraph Vector Search
    D --> K[Similarity Search]
    K --> L[Rank Results]
    L --> E
    end

    C --> |Multiple Searches| D
    K --> |Retrieved Documents| L
```

#### Using HyDE in R2R


<Tabs>

<Tab title="Python">

```python
from r2r import R2RClient

client = R2RClient()

hyde_response = client.rag(
    "What are the main themes in Shakespeare's plays?",
    vector_search_settings={
        "search_strategy": "hyde",
        "search_limit": 10
    }
)

print('hyde_response = ', hyde_response)
```
</Tab>

<Tab title="CLI">

```bash
r2r rag --query="who was aristotle" --use-hybrid-search --stream --search-strategy=hyde
```
</Tab>
</Tabs>

```bash Sample Output
'results': {
    'completion': ...
    'search_results': {
        'vector_search_results': [
            {
                ...
                'score': 0.7715058326721191,
                'text': '## Paragraph from the Chapter\n\nThe Fundamental Theorem of Calculus states that if a function is continuous on a closed interval [a, b], then the function has an antiderivative in the interval [a, b]. This theorem is a cornerstone of calculus and has far-reaching consequences in various fields, including physics, engineering, and economics. The theorem can be proved through the use of Riemann sums and the limit process, which provides a rigorous foundation for understanding the relationship between integration and differentiation. The theorem highlights the deep connection between these two branches of mathematics, offering a unified framework for analyzing functions and their derivatives.'
                'metadata': {
                        'associated_query': 'The fundamental theorem of calculus states that if a function is continuous on the interval [a, b] and F is an antiderivative of f on [a, b], then the integral of f from a to b is equal to F(b) - F(a). This theorem links the concept of differentiation with that of integration, providing a way to evaluate definite integrals without directly computing the limit of a sum.',
                        ...
                    }
                },
        ],
        ...
    }
}
```
### RAG-Fusion

#### What is RAG-Fusion?

RAG-Fusion is an advanced technique that combines Retrieval-Augmented Generation (RAG) with Reciprocal Rank Fusion (RRF) to improve the quality and relevance of retrieved information. Here's how it works:

1. **Query Expansion**: The original query is used to generate multiple related queries, providing different perspectives on the user's question.
2. **Multiple Retrievals**: Each generated query is used to retrieve relevant documents from the database.
3. **Reciprocal Rank Fusion**: The retrieved documents are re-ranked using the RRF algorithm, which combines the rankings from multiple retrieval attempts.
4. **Enhanced RAG**: The re-ranked documents, along with the original and generated queries, are used to generate the final response.

This approach helps to capture a broader context and potentially more relevant information compared to traditional RAG.

#### Implementation Diagram

Here's a diagram illustrating the RAG-Fusion workflow (again, we omit the GraphRAG pipeline for brevity):

```mermaid
graph TD
    A[User Query] --> B[QueryTransformPipe]
    B -->|Generate Multiple Queries| C[MultiSearchPipe]
    C --> D[VectorSearchPipe]
    D --> E[RRF Reranking]
    E --> F[RAG Generation]
    A --> F
    G[Document DB] --> D

    subgraph RAG-Fusion Process
    B --> H[Generated Query 1]
    B --> I[Generated Query 2]
    B --> J[Generated Query n]
    H --> C
    I --> C
    J --> C
    end

    subgraph Vector Search
    D --> K[Search Results 1]
    D --> L[Search Results 2]
    D --> M[Search Results n]
    K --> E
    L --> E
    M --> E
    end

    E --> |Re-ranked Documents| F
```

#### Using RAG-Fusion in R2R


<Tabs>

<Tab title="Python">
```python
from r2r import R2RClient

client = R2RClient()

rag_fusion_response = client.rag(
    "Explain the theory of relativity",
    vector_search_settings={
        "search_strategy": "rag_fusion",
        "search_limit": 20
    }
)

print('rag_fusion_response = ', rag_fusion_response)
```
</Tab>

<Tab title="CLI">
```bash
r2r rag --query="Explain the theory of relativity" --use-hybrid-search --stream --search-strategy=rag_fusion
```
</Tab>

</Tabs>


```bash Sample Output
'results': {
    'completion': ...
    'search_results': {
        'vector_search_results': [
            {
                ...
                'score': 0.04767399003253049,
                'text': '18. The theory of relativity, proposed by Albert Einstein in 1905, is a fundamental theory in modern physics that describes the relationships between space, time, and matter. The theory is based on two postulates, which are the principle of relativity and the invariant speed of light. The principle of relativity states that all inertial reference frames are equivalent, while the invariant speed of light refers to the constant speed of light in vacuum, independent of the motion of the emitting body.\n\n19. Through the use of space-time diagrams, we can graphically represent events and their relationships in space and time. By plotting events on a Minkowski diagram, which is a four-dimensional representation of space and time, we can visualize time dilation and length contraction, two key effects of the theory of relativity. The hyperbola of light in the Minkowski diagram illustrates the invariant speed of light, providing a clear depiction of the geometry of space and time.',
                'metadata': {
                        'associated_queries': ['What is the theory of relativity?', "What are the key principles of Einstein's theory of relativity?", 'How does the theory of relativity impact our understanding of space and time?'],
                        ...
                    }
                },
        ],
        ...
    }
}
```



### Combining with Other Settings

You can readily combine these advanced techniques with other search and RAG settings:

```python
custom_rag_response = client.rag(
    "Describe the impact of climate change on biodiversity",
    vector_search_settings={
        "search_strategy": "hyde",
        "search_limit": 15,
        "use_hybrid_search": True
    },
    rag_generation_config={
        "model": "anthropic/claude-3-opus-20240229",
        "temperature": 0.7
    }
)
```


## Customization and Server-Side Defaults

While R2R allows for runtime configuration of these advanced techniques, it's worth noting that server-side defaults can also be modified for consistent behavior across your application. This includes the ability to update and customize prompts used for techniques like HyDE and RAG-Fusion.

- For general configuration options, refer to the R2R [configuration documentation](/documentation/configuration/introduction).
- To learn about customizing prompts, including those used for HyDE and RAG-Fusion, see the [prompt configuration documentation](/documentation/configuration/prompts).

Prompts play a crucial role in shaping the behavior of these advanced RAG techniques. By modifying the HyDE and RAG-Fusion prompts, you can fine-tune the query expansion and retrieval processes to better suit your specific use case or domain.

<Note>
Currently, these advanced techniques use a hard-coded multi-search configuration in the `MultiSearchPipe`:

```python
class MultiSearchPipe(AsyncPipe):
    class PipeConfig(AsyncPipe.PipeConfig):
        name: str = "multi_search_pipe"
        use_rrf: bool = False
        rrf_k: int = 60  # RRF constant
        num_queries: int = 3
        expansion_factor: int = 3  # Factor to expand results before RRF
```

This configuration will be made user-configurable in the near future, allowing for even greater flexibility in customizing these advanced RAG techniques.
</Note>

## Conclusion

By leveraging these advanced RAG techniques and customizing their underlying prompts, you can significantly enhance the quality and relevance of your retrieval and generation processes. Experiment with different strategies, settings, and prompt variations to find the optimal configuration for your specific use case. The flexibility of R2R allows you to iteratively improve your system's performance and adapt to changing requirements.

---


This guide covers essential maintenance tasks for R2R deployments, with a focus on vector index management and system updates. Understanding when and how to build vector indices, as well as keeping your R2R installation current, is crucial for maintaining optimal performance at scale.

## Vector Indices
### Do You Need Vector Indices?

Vector indices are **not necessary for all deployments**, especially in multi-user applications where each user typically queries their own subset of documents. Consider that:

- In multi-user applications, queries are usually filtered by user_id, drastically reducing the actual number of vectors being searched
- A system with 1 million total vectors but 1000 users might only search through 1000 vectors per query
- Performance impact of not having indices is minimal when searching small per-user document sets

Only consider implementing vector indices when:
- Individual users are searching across hundreds of thousands of documents
- Query latency becomes a bottleneck even with user-specific filtering
- You need to support cross-user search functionality at scale

For development environments or smaller deployments, the overhead of maintaining vector indices often outweighs their benefits.

### Vector Index Management

R2R supports multiple indexing methods, with HNSW (Hierarchical Navigable Small World) being recommended for most use cases:

```python
# Create vector index
create_response = client.create_vector_index(
    table_name="vectors",
    index_method="hnsw",
    index_measure="cosine_distance",
    index_arguments={
        "m": 16,              # Number of connections per element
        "ef_construction": 64 # Size of dynamic candidate list
    },
    concurrently=True
)

# List existing indices
indices = client.list_vector_indices(table_name="vectors")

# Delete an index
delete_response = client.delete_vector_index(
    index_name="ix_vector_cosine_ops_hnsw__20241021211541",
    table_name="vectors",
    concurrently=True
)
```

#### Important Considerations

1. **Pre-warming Requirement**
   - New indices start "cold" and require warming for optimal performance
   - Initial queries will be slower until the index is loaded into memory
   - Consider implementing explicit pre-warming in production
   - Warming must be repeated after system restarts

2. **Resource Usage**
   - Index creation is CPU and memory intensive
   - Memory usage scales with both dataset size and `m` parameter
   - Consider creating indices during off-peak hours

3. **Performance Tuning**
   - HNSW Parameters:
     - `m`: 16-64 (higher = better quality, more memory)
     - `ef_construction`: 64-100 (higher = better quality, longer build time)
   - Distance Measures:
     - `cosine_distance`: Best for normalized vectors (most common)
     - `l2_distance`: Better for absolute distances
     - `max_inner_product`: Optimized for dot product similarity

## System Updates and Maintenance

### Version Management

Check your current R2R version:
```bash
r2r version
```

### Update Process

1. **Prepare for Update**
   ```bash
   # Check current versions
   r2r version
   r2r db current

   # Generate system report (optional)
   r2r generate-report
   ```

2. **Stop Running Services**
   ```bash
   r2r docker-down
   ```

3. **Update R2R**
   ```bash
   r2r update
   ```

4. **Update Database**
   ```bash
   r2r db upgrade
   ```

5. **Restart Services**
   ```bash
   r2r serve --docker [additional options]
   ```

### Database Migration Management

R2R uses database migrations to manage schema changes. Always check and update your database schema after updates:

```bash
# Check current migration
r2r db current

# Apply migrations
r2r db upgrade
```

### Managing Multiple Environments

Use different project names and schemas for different environments:

```bash
# Development
export R2R_PROJECT_NAME=r2r_dev
r2r serve --docker --project-name r2r-dev

# Staging
export R2R_PROJECT_NAME=r2r_staging
r2r serve --docker --project-name r2r-staging

# Production
export R2R_PROJECT_NAME=r2r_prod
r2r serve --docker --project-name r2r-prod
```

## Troubleshooting

If issues occur:

1. Generate a system report:
   ```bash
   r2r generate-report
   ```

2. Check container health:
   ```bash
   r2r docker-down
   r2r serve --docker
   ```

3. Review database state:
   ```bash
   r2r db current
   r2r db history
   ```

4. Roll back if needed:
   ```bash
   r2r db downgrade --revision <previous-working-version>
   ```


## Scaling Strategies

### Horizontal Scaling

For applications serving many users:

1. **Load Balancing**
   - Deploy multiple R2R instances behind a load balancer
   - Each instance can handle a subset of users
   - Particularly effective since most queries are user-specific

2. **Sharding**
   - Consider sharding by user_id for large multi-user deployments
   - Each shard handles a subset of users
   - Maintains performance even with millions of total documents

### Vertical Scaling

For applications requiring large single-user searches:

1. **Cloud Provider Solutions**
   - AWS RDS supports up to 1 billion vectors per instance
   - Scale up compute and memory resources as needed
   - Example instance types:
     - `db.r6g.16xlarge`: Suitable for up to 100M vectors
     - `db.r6g.metal`: Can handle 1B+ vectors

2. **Memory Optimization**
   ```python
   # Optimize for large vector collections
   client.create_vector_index(
       table_name="vectors",
       index_method="hnsw",
       index_arguments={
           "m": 32,              # Increased for better performance
           "ef_construction": 80  # Balanced for large collections
       }
   )
   ```

### Multi-User Considerations

1. **Filtering Optimization**
   ```python
   # Efficient per-user search
   response = client.search(
       "query",
       vector_search_settings={
           "search_filters": {
               "user_id": {"$eq": "current_user_id"}
           }
       }
   )
   ```

2. **Collection Management**
   - Group related documents into collections
   - Enable efficient access control
   - Optimize search scope

3. **Resource Allocation**
   - Monitor per-user resource usage
   - Implement usage quotas if needed
   - Consider dedicated instances for power users


### Performance Monitoring

Monitor these metrics to inform scaling decisions:

1. **Query Performance**
   - Average query latency per user
   - Number of vectors searched per query
   - Cache hit rates

2. **System Resources**
   - Memory usage per instance
   - CPU utilization
   - Storage growth rate

3. **User Patterns**
   - Number of active users
   - Query patterns and peak usage times
   - Document count per user


## Additional Resources

- [Python SDK Ingestion Documentation](/documentation/python-sdk/ingestion)
- [CLI Maintenance Documentation](/documentation/cli/maintenance)
- [Ingestion Configuration Documentation](/documentation/configuration/ingestion/overview)

---


This cookbook demonstrates how to use the agentic capabilities which ship by default in R2R. The R2R agent is an intelligent system that can formulate its own questions, search for information, and provide informed responses based on the retrieved context. It can be customized on the fly.

<Note> Agents in the R2R framework are still in beta. Please let us know what you like/dislike and what features you would like to see added. </Note>

## Understanding R2R's RAG Agent

R2R's RAG agent is designed to provide powerful, context-aware responses by combining large language models with a search capability over your ingested documents. When you initialize an R2R application, it automatically creates a RAG assistant that's ready to use.
R2R plans to extend its agent functionality to mirror core features supported by OpenAI and more, including:

- Multiple tool support (e.g., code interpreter, file search)
- Persistent conversation threads
- Complete end to end observability of agent interactions


R2R also provides support for local RAG capabilities, allowing you to create AI agents that can access and reason over your local document store, entirely offline.

The RAG agent is also available for use through the R2R API, specifically via the `agent` endpoint.

## Using the RAG Agent

Now, let's use the RAG agent to answer questions:

```python
def rag_agent_example():
    # Prepare messages for the chat
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Who is the greatest philospher of all time?",
        },
        {
            "role": "assistant",
            "content": "Aristotle is widely considered the greatest philospher of all time.",
        },
        {
            "role": "user",
            "content": "Can you tell me more about him?",
        },
    ]

    # Use the RAG assistant via the agent endpoint
    response = client.agent(
        messages=messages,
        vector_search_settings={"search_limit": 5, "filters": {}},
        kg_search_settings={"use_kg_search": False},
        rag_generation_config={"max_tokens": 300}
    )

    print("RAG Assistant Response:")
    # Note, response includes the full conversation, including steps taken by the assistant to produce the final result.
    print(response)

# Run the example
rag_agent_example()

# RAG Assistant Response:
# {
#    'results': [
#        {'role': 'system', 'content': '## You are a helpful assistant that can search for information.\n\nWhen asked a question, perform a search to find relevant information and provide a response.\n\nThe response should contain line-item attributions to relevent search results, and be as informative if possible.\nIf no relevant results are found, then state that no results were found.\nIf no obvious question is present, then do not carry out a search, and instead ask for clarification.', 'name': None, 'function_call': None, 'tool_calls': None},
#        {'role': 'system', 'content': 'You are a helpful assistant.', 'name': None, 'function_call': None, 'tool_calls': None},
#        {'role': 'user', 'content': 'Who is the greatest philospher of all time?', 'name': None, 'function_call': None, 'tool_calls': None},
#        {'role': 'assistant', 'content': 'Aristotle is widely considered the greatest philospher of all time.', 'name': None, 'function_call': None, 'tool_calls': None},
#        {'role': 'user', 'content': 'Can you tell me more about him?', 'name': None, 'function_call': None, 'tool_calls': None},
#        {'role': 'assistant', 'content': None, 'name': None, 'function_call': {'name': 'search', 'arguments': '{"query":"Aristotle biography"}'}, 'tool_calls': None},
#        {'role': 'function', 'content': "1. Aristotle[A] (Greek: Ἀριστοτέλης Aristotélēs, pronounced [aristotélɛːs]; 384–322 BC) was an Ancient Greek philosopher and polymath. His writings cover a broad range of subjects spanning the natural sciences, philosophy, linguistics, economics, politics, psychology, and the arts. As the founder of the Peripatetic school of philosophy in the Lyceum in Athens, he began the wider Aristotelian tradition that followed, which set the groundwork for the development of modern science.\n2. Aristotle[A] (Greek: Ἀριστοτέλης Aristotélēs, pronounced [aristotélɛːs]; 384–322 BC) was an Ancient Greek philosopher and polymath. His writings cover a broad range of subjects spanning the natural sciences, philosophy, linguistics, economics, politics, psychology, and the arts. As the founder of the Peripatetic school of philosophy in the Lyceum in Athens, he began the wider Aristotelian tradition that followed, which set the groundwork for the development of modern science.\n3. Aristotle was born in 384 BC[C] in Stagira, Chalcidice,[2] about 55 km (34 miles) east of modern-day Thessaloniki.[3][4] His father, Nicomachus, was the personal physician to King Amyntas of Macedon. While he was young, Aristotle learned about biology and medical information, which was taught by his father.[5] Both of Aristotle's parents died when he was about thirteen, and Proxenus of Atarneus became his guardian.[6] Although little information about Aristotle's childhood has survived, he probably spent\n4. Aristotle was born in 384 BC[C] in Stagira, Chalcidice,[2] about 55 km (34 miles) east of modern-day Thessaloniki.[3][4] His father, Nicomachus, was the personal physician to King Amyntas of Macedon. While he was young, Aristotle learned about biology and medical information, which was taught by his father.[5] Both of Aristotle's parents died when he was about thirteen, and Proxenus of Atarneus became his guardian.[6] Although little information about Aristotle's childhood has survived, he probably spent\n5. Life\nIn general, the details of Aristotle's life are not well-established. The biographies written in ancient times are often speculative and historians only agree on a few salient points.[B]\n", 'name': 'search', 'function_call': None, 'tool_calls': None},
#        {'role': 'assistant', 'content': "Aristotle (384–322 BC) was an Ancient Greek philosopher and polymath whose contributions have had a profound impact on various fields of knowledge. Here are some key points about his life and work:\n\n1. **Early Life**: Aristotle was born in 384 BC in Stagira, Chalcidice, which is near modern-day Thessaloniki, Greece. His father, Nicomachus, was the personal physician to King Amyntas of Macedon, which exposed Aristotle to medical and biological knowledge from a young age [C].\n\n2. **Education and Career**: After the death of his parents, Aristotle was sent to Athens to study at Plato's Academy, where he remained for about 20 years. After Plato's death, Aristotle left Athens and eventually became the tutor of Alexander the Great [C].\n\n3. **Philosophical Contributions**: Aristotle founded the Lyceum in Athens, where he established the Peripatetic school of philosophy. His works cover a wide range of subjects, including metaphysics, ethics, politics, logic, biology, and aesthetics. His writings laid the groundwork for many modern scientific and philosophical inquiries [A].\n\n4. **Legacy**: Aristotle's influence extends beyond philosophy to the natural sciences, linguistics, economics, and psychology. His method of systematic observation and analysis has been foundational to the development of modern science [A].\n\nAristotle's comprehensive approach to knowledge and his systematic methodology have earned him a lasting legacy as one of the greatest philosophers of all time.\n\nSources:\n- [A] Aristotle's broad range of writings and influence on modern science.\n- [C] Details about Aristotle's early life and education.", 'name': None, 'function_call': None, 'tool_calls': None}
#    ]
# }

```

In this example, the agent might formulate its own questions to gather more information before providing a response. For instance, it could ask the search tool about Aristotle's scientific works or his methods of inquiry.

## Streaming Responses

To see the agent's thought process in real-time, you can use streaming responses:

```python
def streaming_rag_agent_example():
    messages = [
        {"role": "system", "content": "You are a helpful agent with access to a large knowledge base. You can ask questions to gather more information if needed."},
        {"role": "user", "content": "What were Aristotle's main philosophical ideas?"},
    ]

    streaming_response = client.agent(
        messages=messages,
        vector_search_settings={"search_limit": 5, "filters": {}},
        kg_search_settings={"use_kg_search": False},
        rag_generation_config={"max_tokens": 300, "stream": True}
    )

    print("Streaming RAG Assistant Response:")
    for chunk in streaming_response:
        print(chunk, end="", flush=True)
    print()  # New line after streaming response

streaming_rag_agent_example()

# Streaming RAG Assistant Response:

# <function_call><name>search</name><arguments>{"query":"Aristotle's main philosophical ideas"}</arguments><results>1. Politics
# Main article: Politics (Aristotle)
# 2. Ethics
# Main article: Aristotelian ethics
# Aristotle considered ethics to be a practical rather than theoretical study, i.e., one aimed at becoming good and doing good rather than knowing for its own sake. He wrote several treatises on ethics, most notably including the Nicomachean Ethics.[117]
# 3. Metaphysics
# Main article: Metaphysics (Aristotle)
# The word "metaphysics" appears to have been coined by the first century AD editor who assembled various small selections of Aristotle's works to the treatise we know by the name Metaphysics.[34] Aristotle called it "first philosophy", and distinguished it from mathematics and natural science (physics) as the contemplative (theoretikē) philosophy which is "theological" and studies the divine. He wrote in his Metaphysics (1026a16):
# 4. Practical philosophy
# Aristotle's practical philosophy covers areas such as ethics, politics, economics, and rhetoric.[40]
# 5. Aristotle[A] (Greek: Ἀριστοτέλης Aristotélēs, pronounced [aristotélɛːs]; 384–322 BC) was an Ancient Greek philosopher and polymath. His writings cover a broad range of subjects spanning the natural sciences, philosophy, linguistics, economics, politics, psychology, and the arts. As the founder of the Peripatetic school of philosophy in the Lyceum in Athens, he began the wider Aristotelian tradition that followed, which set the groundwork for the development of modern science.
# </results></function_call>Aristotle, an ancient Greek philosopher, made significant contributions across various fields of philosophy. Here are some of his main philosophical ideas:
#
# 1. **Metaphysics**: Aristotle's metaphysics, often referred to as "first philosophy," deals with the nature of reality, existence, and the fundamental nature of being. He introduced the concept of substance and essence, and his work laid the groundwork for later metaphysical thought [Metaphysics (Aristotle)].
#
# 2. **Ethics**: Aristotle's ethical philosophy is primarily outlined in his work "Nicomachean Ethics." He emphasized the concept of virtue ethics, which focuses on the development of good character traits (virtues) and living a life of moral excellence. He believed that the ultimate goal of human life is eudaimonia, often translated as "happiness" or "flourishing," achieved through virtuous living [Aristotelian ethics].
#
# 3. **Politics**: In his work "Politics," Aristotle explored the nature of human communities and the role of the state. He believed that humans are naturally political animals and that the state exists to promote the good life. He analyzed different forms of government and advocated for a constitutional government that balances the interests of different social classes [Politics (Aristotle)].
#
# 4. **Practical Philosophy**: Aristotle's practical philosophy encompasses ethics, politics, economics, and rhetoric. He believed that philosophy should be practical and aimed at improving human life and society [Practical philosophy].
#
# 5. **Natural Sciences**: Aristotle made substantial contributions to the natural sciences, including biology, physics, and astronomy. He conducted empirical observations and classified various forms of life, laying the foundation for the scientific method [Aristotle's contributions to natural sciences].

# Aristotle's extensive body of work has had a profound and lasting impact on Western philosophy and science, influencing countless thinkers and shaping the development of various academic disciplines.

```

This will produce a streaming response, showing the agent's thought process, including its search queries and the gradual construction of its response.

## Customizing the RAG Agent
<Warning> This section is still under development, please proceed carefully when customizing agents.</Warning>
You can customize various aspects of the RAG agent's behavior:

```python
custom_response = client.agent(
    messages=[
        {"role": "system", "content": "You are an expert on ancient Greek philosophy. Ask clarifying questions if needed."},
        {"role": "user", "content": "Compare Aristotle's and Plato's views on the nature of reality."},
    ],
    vector_search_settings={"search_limit": 25, "use_hybrid_search": True, "filters": {"category": {"$eq": "philosophy"}}},
    kg_search_settings={"use_kg_search": False},
    rag_generation_config={
        "max_tokens": 500,
        "temperature": 0.7,
        "model": "openai/gpt-4o"  # Assuming you have access to GPT-4
    }
)

print("Customized RAG Agent Response:")
print(custom_response)
```

This example demonstrates how to:
- Set custom search filters
- Enable hybrid search (combining vector and keyword search)
- Adjust the number of search results
- Customize the generation config
- Use a specific model for the response

...

## Conclusion

The R2R RAG assistant is a powerful tool that combines large language models with advanced search capabilities. By leveraging your ingested documents and the flexible `agent` endpoint, you can create dynamic, context-aware conversational experiences.

Key takeaways:
1. R2R automatically creates a RAG assistant when initializing the application.
2. The `agent` endpoint provides easy access to the RAG assistant's capabilities.
3. You can customize various aspects of the assistant's behavior, including search settings and generation parameters.
4. Streaming responses allow for real-time interaction and visibility into the assistant's thought process.

Experiment with different settings and configurations to find the optimal setup for your specific use case. Remember to keep your ingested documents up-to-date to ensure the assistant has access to the most relevant and recent information.

---



This guide shows how to use R2R to:

   1. Ingest files into R2R
   2. Search over ingested files
   3. Use your data as input to RAG (Retrieval-Augmented Generation)
   4. Perform basic user auth
   5. Observe and analyze an R2R deployment


Be sure to complete the [installation instructions](/documentation/installation) before continuing with this guide.


## Introduction

R2R is an engine for building user-facing Retrieval-Augmented Generation (RAG) applications. At its core, R2R provides this service through an architecture of providers, services, and an integrated RESTful API. This cookbook provides a detailed walkthrough of how to interact with R2R. [Refer here](/documentation/deep-dive/main/introduction) for a deeper dive on the R2R system architecture.

## R2R Application Lifecycle

The following diagram illustrates how R2R assembles a user-facing application:

```mermaid
flowchart LR
  Developer[Developer]
  Config[R2RConfig]
  R2R[R2R Application]
  SDK[R2R SDK]
  User[User]

  Developer -->|Customizes| Config
  Config -->|Configures| R2R
  Developer -->|Deploys| R2R
  Developer -->|Implements| SDK
  SDK -->|Interfaces with| R2R
  User -->|Interacts via| SDK
```


### Hello R2R

R2R gives developers configurable vector search and RAG right out of the box, as well as direct method calls instead of the client-server architecture seen throughout the docs:
```python core/examples/hello_r2r.py
from r2r import R2RClient

client = R2RClient("http://localhost:7272")

with open("test.txt", "w") as file:
    file.write("John is a person that works at Google.")

client.ingest_files(file_paths=["test.txt"])

# Call RAG directly
rag_response = client.rag(
    query="Who is john",
    rag_generation_config={"model": "openai/gpt-4o-mini", "temperature": 0.0},
)
results = rag_response["results"]
print(f"Search Results:\n{results['search_results']}")
print(f"Completion:\n{results['completion']}")
```

### Configuring R2R
R2R is highly configurable. To customize your R2R deployment:

1. Create a local configuration file named `r2r.toml`.
2. In this file, override default settings as needed.

For example:
```toml r2r.toml
[completion]
provider = "litellm"
concurrent_request_limit = 16

  [completion.generation_config]
  model = "openai/gpt-4o"
  temperature = 0.5

[ingestion]
provider = "r2r"
chunking_strategy = "recursive"
chunk_size = 1_024
chunk_overlap = 512
excluded_parsers = ["mp4"]
```

Then, use the `config-path` argument to specify your custom configuration when launching R2R:

```bash
r2r serve --docker --config-path=r2r.toml
```

You can read more about [configuration here](/documentation/configuration).


## Document Ingestion and Management

R2R efficiently handles diverse document types using Postgres with pgvector, combining relational data management with vector search capabilities. This approach enables seamless ingestion, storage, and retrieval of multimodal data, while supporting flexible document management and user permissions.

Key features include:

- Unique `document_id` generation for each ingested file
- User and collection permissions through `user_id` and `collection_ids`
- Document versioning for tracking changes over time
- Granular access to document content through chunk retrieval
- Flexible deletion and update mechanisms


<Note> Note, all document management commands are gated at the user level, with the exception of superusers. </Note>

<AccordionGroup>

<Accordion icon="database" title="Ingest Data" defaultOpen={true}>
R2R offers a powerful data ingestion process that handles various file types including `html`, `pdf`, `png`, `mp3`, and `txt`. The full list of supported filetypes is available [here](/documentation/configuration/ingestion/overview). The ingestion process parses, chunks, embeds, and stores documents efficiently with a fully asynchronous pipeline. To demonstrate this functionality:

<Tabs>
<Tab title="CLI">
```bash
r2r ingest-sample-files
```
</Tab>

<Tab title="Python">
```python
from r2r import R2RClient
from glob import glob

client = R2RClient("http://localhost:7272")
files = glob.glob('path/to/r2r/examples/data')
client.ingest_files(files)
```
</Tab>

<Tab title="JavaScript">

```javascript
const files = [
  { path: "r2r/examples/data/aristotle.txt", name: "aristotle.txt" },
];

await client.ingestFiles(files, {
  metadatas: [{ title: "aristotle.txt" }],
  user_ids: [
    "123e4567-e89b-12d3-a456-426614174000",
  ],
  skip_document_info: false,
});
```
</Tab>
</Tabs>


This command initiates the ingestion process, producing output similar to:

```bash
[{'message': 'Ingestion task queued successfully.', 'task_id': '6e27dfca-606d-422d-b73f-2d9e138661b4', 'document_id': '28a7266e-6cee-5dd2-b7fa-e4fc8f2b49c6'}, {'message': 'Ingestion task queued successfully.', 'task_id': 'd37deef1-af08-4576-bd79-6d2a7fb6ec33', 'document_id': '2c91b66f-e960-5ff5-a482-6dd0a523d6a1'}, {'message': 'Ingestion task queued successfully.', 'task_id': '4c1240f0-0692-4b67-8d2b-1428f71ea9bc', 'document_id': '638f0ed6-e0dc-5f86-9282-1f7f5243d9fa'}, {'message': 'Ingestion task queued successfully.', 'task_id': '369abcea-79a2-480c-9ade-bbc89f5c500e', 'document_id': 'f25fd516-5cac-5c09-b120-0fc841270c7e'}, {'message': 'Ingestion task queued successfully.', 'task_id': '7c99c168-97ee-4253-8a6f-694437f3e5cb', 'document_id': '77f67c65-6406-5076-8176-3844f3ef3688'}, {'message': 'Ingestion task queued successfully.', 'task_id': '9a6f94b0-8fbc-4507-9435-53e0973aaad0', 'document_id': '9fbe403b-c11c-5aae-8ade-ef22980c3ad1'}, {'message': 'Ingestion task queued successfully.', 'task_id': '61d0e2e0-45ec-43db-9837-ff4da5166ee9', 'document_id': '0032a7a7-cb2a-5d08-bfc1-93d3b760deb4'}, {'message': 'Ingestion task queued successfully.', 'task_id': '1479390e-c295-47b0-a570-370b05b86c8b', 'document_id': 'f55616fb-7d48-53d5-89c2-15d7b8e3834c'}, {'message': 'Ingestion task queued successfully.', 'task_id': '92f73a07-2286-4c42-ac02-d3eba0f252e0', 'document_id': '916b0ed7-8440-566f-98cf-ed7c0f5dba9b'}]
```

Key features of the ingestion process:
1. Unique `document_id` generation for each file
2. Metadata association, including `user_id` and `collection_ids` for document management
3. Efficient parsing, chunking, and embedding of diverse file types
</Accordion>


<Accordion icon="folder-open" title="Get Documents Overview">
R2R allows retrieval of high-level document information stored in a relational table within the Postgres database. To fetch this information:



<Tabs>

<Tab title="CLI">
```bash
r2r documents-overview
```
</Tab>

<Tab title="Python">
```python
client.documents_overview()
```
</Tab>

<Tab title="JavaScript">
```javascript
await client.documentsOverview()
```
</Tab>

<Tab title="Curl">
```bash
curl -X POST http://localhost:7272/v2/documents_overview \
  -H "Content-Type: application/json" \
  -d '{
    "document_ids": null,
    "user_ids": null
  }'
```
</Tab>
</Tabs>


This command returns document metadata, including:

```bash
[
  {
    'id': '9fbe403b-c11c-5aae-8ade-ef22980c3ad1',
    'title': 'aristotle.txt',
    'user_id': '2acb499e-8428-543b-bd85-0d9098718220',
    'type': 'txt',
    'created_at': '2024-09-06T03:32:02.991742Z',
    'updated_at': '2024-09-06T03:32:02.991744Z',
    'ingestion_status': 'success',
    'restructuring_status': 'pending',
    'version': 'v0',
    'collection_ids': [],
    'metadata': {'title': 'aristotle.txt', 'version': 'v0'}
  }
  ...
]
```

This overview provides quick access to document versions, sizes, and associated metadata, facilitating efficient document management.
</Accordion>



<Accordion icon="file" title="Get Document Chunks">
R2R enables retrieval of specific document chunks and associated metadata. To fetch chunks for a particular document by id:


<Tabs>

<Tab title="CLI">
```bash
r2r document-chunks --document-id=9fbe403b-c11c-5aae-8ade-ef22980c3ad1
```
</Tab>

<Tab title="Python">
```python
client.document_chunks("9fbe403b-c11c-5aae-8ade-ef22980c3ad1")
```
</Tab>

<Tab title="JavaScript">
```javascript
await client.documentChunks("9fbe403b-c11c-5aae-8ade-ef22980c3ad1"),
```
</Tab>

<Tab title="Curl">
```bash
curl -X POST http://localhost:7272/v2/document_chunks \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "9fbe403b-c11c-5aae-8ade-ef22980c3ad1"
  }'
```
</Tab>

</Tabs>


This command returns detailed chunk information:

```bash
[
  {
    'text': 'Aristotle[A] (Greek: Ἀριστοτέλης Aristotélēs, pronounced [aristotélɛːs]; 384–322 BC) was an Ancient Greek philosopher and polymath. His writings cover a broad range of subjects spanning the natural sciences, philosophy, linguistics, economics, politics, psychology, and the arts. As the founder of the Peripatetic school of philosophy in the Lyceum in Athens, he began the wider Aristotelian tradition that followed, which set the groundwork for the development of modern science.',
    'title': 'aristotle.txt',
    'user_id': '2acb499e-8428-543b-bd85-0d9098718220',
    'version': 'v0',
    'chunk_order': 0,
    'document_id': '9fbe403b-c11c-5aae-8ade-ef22980c3ad1',
    'extraction_id': 'aeba6400-1bd0-5ee9-8925-04732d675434',
    'fragment_id': 'f48bcdad-4155-52a4-8c9d-8ba06e996ba3',
  },
  ...
]
```
These features allow for granular access to document content.
</Accordion>

<Accordion icon="trash" title="Delete Documents">
R2R supports flexible document deletion through a method that can run arbitrary deletion filters. To delete a document by its ID:


<Tabs>

<Tab title="CLI">
```bash
r2r delete --filter=document_id:eq:9fbe403b-c11c-5aae-8ade-ef22980c3ad1
```
</Tab>

<Tab title="Python">
```python
client.delete(
  {
    "document_id":
      {"$eq": "9fbe403b-c11c-5aae-8ade-ef22980c3ad1"}
  }
)
```
</Tab>

<Tab title="JavaScript">
```javascript
await client.delete(["document_id"], ["9fbe403b-c11c-5aae-8ade-ef22980c3ad1"]);
```
</Tab>


<Tab title="Curl">
```bash
curl -X DELETE http://localhost:7272/v2/delete \
  -H "Content-Type: application/json" \
  -d '{
    "keys": ["document_id"],
    "values": ["9fbe403b-c11c-5aae-8ade-ef22980c3ad1"]
  }'
```
</Tab>
</Tabs>



This command produces output similar to:

```bash
{results: {}}
```

Key features of the deletion process:
1. Deletion by document ID, extraction ID, or other.
2. Cascading deletion of associated chunks and metadata
3. Confirmation of successful deletion

This flexible deletion mechanism ensures precise control over document management within the R2R system.
</Accordion>

<Accordion icon="arrows-rotate" title="Update Documents">
  R2R provides robust document update capabilities through two main endpoints: `update_documents` and `update_files`. These endpoints allow for seamless updating of existing documents while maintaining version control.

  Key features of the update process:

  1. **Automatic versioning**: When updating a document, R2R automatically increments the version (e.g., from "v0" to "v1").

  2. **Metadata preservation**: The update process maintains existing metadata while allowing for updates.

  3. **Content replacement**: The new document content completely replaces the old content in the order shown below
        - Ingest the new version of the document
        - Delete the old version

  Executing the command below will update one of the sample documents ingested earlier.
  <Tabs>

    <Tab title="CLI">
    ```bash
    r2r update-files core/examples/data/aristotle_v2.txt --document-ids=9fbe403b-c11c-5aae-8ade-ef22980c3ad1
    ```
    </Tab>


    <Tab title="Python">
    ```python
    file_paths = ["/path/to/r2r/examples/data/aristotle_v2.txt"]
    document_ids = ["9fbe403b-c11c-5aae-8ade-ef22980c3ad1"]
    client.update_files(file_paths, document_ids)
    ```
    </Tab>

    <Tab title="JavaScript">
      ```javascript
      const updated_file = [
        { path: "/path/to/r2r/examples/data/aristotle_v2.txt", name: "aristotle_v2.txt" },
      ];
      await client.updateFiles(updated_file, {
        document_ids: ["9fbe403b-c11c-5aae-8ade-ef22980c3ad1"],
        metadatas: [{ title: "aristotle_v2.txt" }],
      }),
      ```
      </Tab>

    <Tab title="Curl">
    ```bash
    curl -X POST http://localhost:7272/v2/update_files \
    -H "Content-Type: multipart/form-data" \
    -F "file_paths=@/path/to/your/r2r/examples/data/aristotle_v2.txt" \
    -F 'document_ids=["9fbe403b-c11c-5aae-8ade-ef22980c3ad1"]'
    ```
    </Tab>
  </Tabs>
  **Expected Output:**

  ```bash
  {
    "results": {
      "message": "Update task queued successfully.",
      "task_id": "00fc8484-179f-47db-a474-d81b95d80cf2",
      "document_ids": [
        "9fbe403b-c11c-5aae-8ade-ef22980c3ad1"
      ]
    }
  }
  ```

  Behind the scenes, this command utilizes the `update_files` endpoint. The process involves:

  1. Reading the new file content
  2. Incrementing the document version
  3. Ingesting the new version with updated metadata
  4. Deleting the old version of the document

  For programmatic updates, you can use the RESTful API endpoint `/update_files`. This endpoint accepts a `R2RUpdateFilesRequest`, which includes:

  - `files`: List of UploadFile objects containing the new document content
  - `document_ids`: UUIDs of the documents to update
  - `metadatas`: Optional updated metadata for each document

  The update process ensures data integrity and maintains a clear history of document changes through versioning.

</Accordion>
</AccordionGroup>

For more advanced document management techniques and user authentication details, refer to [the user auth cookbook](/cookbooks/user-auth).


Certainly! I'll rewrite the AI Powered Search section without using dropdowns, presenting it as a continuous, detailed explanation of R2R's search capabilities. Here's the revised version:

## AI Powered Search

R2R offers powerful and highly configurable search capabilities, including vector search, hybrid search, and knowledge graph-enhanced search. These features allow for more accurate and contextually relevant information retrieval.


### Vector Search

Vector search inside of R2R is highly configurable, allowing you to fine-tune your search parameters for optimal results. Here's how to perform a basic vector search:

<Tabs>

<Tab title="CLI">
```python
r2r search --query="What was Uber's profit in 2020?"
```
</Tab>


<Tab title="Python">
```python
client.search("What was Uber's profit in 2020?", {
    "index_measure": "l2_distance", # default is `cosine_distance`
    "search_limit": 25,
})
```
</Tab>

<Tab title="JavaScript">
```javascript
await client.search("What was Uber's profit in 2020?", true, {}, 10, false, {
    indexMeasure: "cosine_distance",
    includeValues: true,
    includeMetadatas: true,
    probes: 10,
    efSearch: 40
});
```
</Tab>

<Tab title="Curl">
```bash
curl -X POST http://localhost:7272/v2/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was Uber'\''s profit in 2020?",
    "vector_search_settings": {
      "use_vector_search": true,
      "index_measure": "cosine_distance",
      "search_limit": 10,
      "include_values": true,
      "include_metadatas": true,
      "probes": 10,
      "ef_search": 40
    }
  }'
```
</Tab>
</Tabs>

<AccordionGroup>
  <Accordion title="Expected Output">
  ```json
  { 'results':
    {'vector_search_results':
      [
        {
          'fragment_id': 'ab6d0830-6101-51ea-921e-364984bfd177',
          'extraction_id': '429976dd-4350-5033-b06d-8ffb67d7e8c8',
          'document_id': '26e0b128-3043-5674-af22-a6f7b0e54769',
          'user_id': '2acb499e-8428-543b-bd85-0d9098718220',
          'collection_ids': [],
          'score': 0.285747126074015,
          'text': 'Net\n loss attributable to Uber Technologies, Inc. was $496 million, a 93% improvement year-over-year, driven by a $1.6 billion pre-tax gain on the sale of ourATG\n Business to Aurora, a $1.6 billion pre-tax  net benefit relating to Ubers equity investments, as  well as reductions in our fixed cost structure and increasedvariable cost effi\nciencies. Net loss attributable to Uber Technologies, Inc. also included $1.2 billion of stock-based compensation expense.Adjusted',
          'metadata': {'title': 'uber_2021.pdf', 'version': 'v0', 'chunk_order': 5, 'associatedQuery': "What was Uber's profit in 2020?"}
        },
      ...
      ]
    }
  }
  ```
  </Accordion>
</AccordionGroup>
Key configurable parameters for vector search include:

- `use_vector_search`: Enable or disable vector search.
- `index_measure`: Choose between "cosine_distance", "l2_distance", or "max_inner_product".
- `search_limit`: Set the maximum number of results to return.
- `include_values`: Include search score values in the results.
- `include_metadatas`: Include element metadata in the results.
- `probes`: Number of ivfflat index lists to query (higher increases accuracy but decreases speed).
- `ef_search`: Size of the dynamic candidate list for HNSW index search (higher increases accuracy but decreases speed).

### Hybrid Search

R2R supports hybrid search, which combines traditional keyword-based search with vector search for improved results. Here's how to perform a hybrid search:

<Tabs>

<Tab title="CLI">
```python
r2r search --query="What was Uber's profit in 2020?" --use-hybrid-search
```
</Tab>

<Tab title="Python">
```python
client.search(
    "What was Uber's profit in 2020?",
    {
        "index_measure": "l2_distance",
        "use_hybrid_search": True,
        "hybrid_search_settings": {
            "full_text_weight": 1.0,
            "semantic_weight": 5.0,
            "full_text_limit": 200,
            "rrf_k": 50,
        },
        "filters": {"title": {"$in": ["lyft_2021.pdf", "uber_2021.pdf"]}},
        "search_limit": 10,
        "probes": 25,
        "ef_search": 100,
    },
)
```
</Tab>

<Tab title="JavaScript">
```javascript
await client.search("What was Uber's profit in 2020?", {
    indexMeasure: "l2_distance",
    useHybridSearch: true,
    hybridSearchSettings: {
        fullTextWeight: 1.0,
        semanticWeight: 5.0,
        fullTextLimit: 200,
        rrfK: 50
    },
    filters: { title: { $in: ["lyft_2021.pdf", "uber_2021.pdf"] } },
    searchLimit: 10,
    probes: 25,
    efSearch: 100
});
```
</Tab>

<Tab title="Curl">
```bash
curl -X POST http://localhost:7272/v2/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was Uber'\''s profit in 2020?",
    "vector_search_settings": {
      "index_measure": "l2_distance",
      "use_hybrid_search": true,
      "hybrid_search_settings": {
        "full_text_weight": 1.0,
        "semantic_weight": 5.0,
        "full_text_limit": 200,
        "rrf_k": 50
      },
      "filters": {"title": {"$in": ["lyft_2021.pdf", "uber_2021.pdf"]}},
      "search_limit": 10,
      "probes": 25,
      "ef_search": 100
    }
  }'
```
</Tab>
</Tabs>


### Knowledge Graph Search

R2R integrates knowledge graph capabilities to enhance search results with structured relationships. Knowledge graph search can be configured to focus on specific entity types, relationships, or search levels. Here's how to utilize knowledge graph search:

<Warning>
Knowledge Graphs are not constructed by default, refer to the [cookbook here](/cookbooks/graphrag) before attempting to run the command below!
</Warning>

<Tabs>
<Tab title="CLI">
```python
r2r search --query="Who founded Airbnb?" --use-kg-search --kg-search-type=local
```
</Tab>

<Tab title="Python">
```python
client.search("Who founded Airbnb?", kg_search_settings={
    "use_kg_search": True,
    "kg_search_type": "local",
    "kg_search_level": 0, # level of community to search
    "max_community_description_length": 65536,
    "max_llm_queries_for_global_search": 250,
    "local_search_limits": {
        "__Entity__": 20,
        "__Relationship__": 20,
        "__Community__": 20
    }
})
```
</Tab>

<Tab title="JavaScript">
```javascript
await client.search("Who founded Airbnb?", true, {}, 10, false, {}, {
    useKgSearch: true,
    kgSearchType: "local",
    kgSearchLevel: "0",
    maxCommunityDescriptionLength: 65536,
    maxLlmQueriesForGlobalSearch: 250,
    localSearchLimits: {
        __Entity__: 20,
        __Relationship__: 20,
        __Community__: 20
    }
});
```
</Tab>

<Tab title="Curl">
```bash
curl -X POST http://localhost:7272/v2/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Who founded Airbnb?",
    "kg_search_settings": {
      "use_kg_search": true,
      "kg_search_type": "local",
      "kg_search_level": "0",
      "max_community_description_length": 65536,
      "max_llm_queries_for_global_search": 250,
      "local_search_limits": {
        "__Entity__": 20,
        "__Relationship__": 20,
        "__Community__": 20
      }
    }
  }'
```
</Tab>
</Tabs>

Key configurable parameters for knowledge graph search include:

- `use_kg_search`: Enable knowledge graph search.
- `kg_search_type`: "local"
- `kg_search_level`: Specify the level of community to search.
- `entity_types`: List of entity types to include in the search.
- `relationships`: List of relationship types to include in the search.
- `max_community_description_length`: Maximum length of community descriptions.
- `max_llm_queries_for_global_search`: Limit on the number of LLM queries for global search.
- `local_search_limits`: Set limits for different types of local searches.

Knowledge graph search provides structured information about entities and their relationships, complementing the text-based search results and offering a more comprehensive understanding of the data.

R2R's search functionality is highly flexible and can be tailored to specific use cases. By adjusting these parameters, you can optimize the search process for accuracy, speed, or a balance between the two, depending on your application's needs. The combination of vector search, hybrid search, and knowledge graph capabilities allows for powerful and context-aware information retrieval, enhancing the overall performance of your RAG applications.

## Retrieval-Augmented Generation (RAG)

R2R is built around a comprehensive Retrieval-Augmented Generation (RAG) engine, allowing you to generate contextually relevant responses based on your ingested documents. The RAG process combines all the search functionality shown above with Large Language Models to produce more accurate and informative answers.

<AccordionGroup>

<Accordion icon="brain" title="Basic RAG" defaultOpen={true}>
To generate a response using RAG, use the following command:


<Tabs>

<Tab title="CLI">
```bash
r2r rag --query="What was Uber's profit in 2020?"
```
</Tab>


<Tab title="Python">
```python
client.rag(query="What was Uber's profit in 2020?")
```
</Tab>

<Tab title="JavaScript">
```javascript
await client.rag({ query: "What was Uber's profit in 2020?" });
```
</Tab>

<Tab title="Curl">
```bash
curl -X POST http://localhost:7272/v2/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was Uber'\''s profit in 2020?"
  }'
```
</Tab>
</Tabs>



**Example Output:**

```bash
{'results': [
    ChatCompletion(
        id='chatcmpl-9RCB5xUbDuI1f0vPw3RUO7BWQImBN',
        choices=[
            Choice(
                finish_reason='stop',
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    content="Uber's profit in 2020 was a net loss of $6,768 million [10].",
                    role='assistant',
                    function_call=None,
                    tool_calls=None)
                )
            ],
        created=1716268695,
        model='gpt-4o-mini',
        object='chat.completion',
        system_fingerprint=None,
        usage=CompletionUsage(completion_tokens=20, prompt_tokens=1470, total_tokens=1490)
    )
]}
```

This command performs a search on the ingested documents and uses the retrieved information to generate a response.
</Accordion>
<Accordion icon="layer-group" title="RAG w/ Hybrid Search">
R2R also supports hybrid search in RAG, combining the power of vector search and keyword-based search. To use hybrid search in RAG, simply add the `use_hybrid_search` flag to your search settings input:

<Tabs>
<Tab title="CLI">
```bash
r2r rag --query="Who is Jon Snow?" --use-hybrid-search
```
</Tab>


<Tab title="Python">
```javascript
results = client.rag("Who is Jon Snow?", {"use_hybrid_search": True})
```
</Tab>

<Tab title="JavaScript">
```javascript
await client.rag({
  query: "Who is Jon Snow?",
});
```
</Tab>

<Tab title="Curl">
```bash
curl -X POST http://localhost:7272/v2/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Who is Jon Snow?",
    "vector_search_settings": {
      "use_vector_search": true,
      "filters": {},
      "search_limit": 10,
      "use_hybrid_search": true
    }
  }'
```
</Tab>
</Tabs>



**Example Output:**

```bash
{'results': [
    ChatCompletion(
        id='chatcmpl-9cbRra4MNQGEQb3BDiFujvDXIehud',
        choices=[
            Choice(
                finish_reason='stop',
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    content="Jon Snow is mentioned in the context as one of Samwell (Sam) Tarly's closest companions at the Wall [5], [6].",
                    role='assistant',
                    function_call=None,
                    tool_calls=None)
                )
            ],
        created=1718987443,
        model='openai/gpt-4o-2024-05-13',
        object='chat.completion',
        system_fingerprint=None,
        usage=CompletionUsage(completion_tokens=20, prompt_tokens=1192, total_tokens=1221)
    )
]}
```


This example demonstrates how hybrid search can enhance the RAG process by combining semantic understanding with keyword matching, potentially providing more accurate and comprehensive results.
</Accordion>

<Accordion icon="screencast" title="Streaming RAG">
R2R also supports streaming RAG responses, which can be useful for real-time applications. To use streaming RAG:

<Tabs>
<Tab title="CLI">
```bash
r2r rag --query="who was aristotle" --use-hybrid-search --stream
```
</Tab>


<Tab title="Python">
```python
response = client.rag(
  "who was aristotle",
  rag_generation_config={"stream": True},
  vector_search_settings={"use_hybrid_search": True},
)
for chunk in response:
    print(chunk, end='', flush=True)
```
</Tab>

<Tab title="JavaScript">
```javascript
await client.rag({
  query: query,
  rag_generation_config: {
    stream: true,
  }
});
```
</Tab>

</Tabs>


**Example Output:**

```bash
<search>["{\"id\":\"808c47c5-ebef-504a-a230-aa9ddcfbd87 .... </search>
<completion>Aristotle was an Ancient Greek philosopher and polymath born in 384 BC in Stagira, Chalcidice [1], [4]. He was a student of Plato and later became the tutor of Alexander the Great [2]. Aristotle founded the Peripatetic school of philosophy in the Lyceum in Athens and made significant contributions across a broad range of subjects, including natural sciences, philosophy, linguistics, economics, politics, psychology, and the arts [4]. His work laid the groundwork for the development of modern science [4]. Aristotle's influence extended well beyond his time, impacting medieval Islamic and Christian scholars, and his contributions to logic, ethics, and biology were particularly notable [8], [9], [10].</completion>```
```

Streaming allows the response to be generated and sent in real-time, chunk by chunk.
</Accordion>
<Accordion icon="gears" title="Customizing RAG">
  R2R offers extensive customization options for its Retrieval-Augmented Generation (RAG) functionality:

  1. **Search Settings**: Customize vector and knowledge graph search parameters using `VectorSearchSettings` and `KGSearchSettings`.

  2. **Generation Config**: Fine-tune the language model's behavior with `GenerationConfig`, including:
      - Temperature, top_p, top_k for controlling randomness
      - Max tokens, model selection, and streaming options
      - Advanced settings like beam search and sampling strategies

  3. **Multiple LLM Support**: Easily switch between different language models and providers:
      - OpenAI models (default)
      - Anthropic's Claude models
      - Local models via Ollama
      - Any provider supported by LiteLLM

  Example of customizing the model:


<Tabs>
<Tab title="CLI">
```bash
r2r rag --query="who was aristotle?" --rag-model="anthropic/claude-3-haiku-20240307" --stream --use-hybrid-search
```
</Tab>

<Tab title="Python">

  ```python
  # requires ANTHROPIC_API_KEY is set
  response = client.rag(
    "Who was Aristotle?",
    rag_generation_config={"model":"anthropic/claude-3-haiku-20240307", "stream": True}
  )
  for chunk in response:
      print(chunk, nl=False)
  ```
</Tab>

<Tab title="JavaScript">
```javascript
await client.rag({
  query: query,
  rag_generation_config: {
    model: 'claude-3-haiku-20240307',
    temperature: 0.7,
  }
});
```
</Tab>

<Tab title="Curl">

```bash
# requires ANTHROPIC_API_KEY is set
curl -X POST http://localhost:7272/v2/rag \
    -H "Content-Type: application/json" \
    -d '{
        "query": "Who is Jon Snow?",
        "rag_generation_config": {
            "model": "claude-3-haiku-20240307",
            "temperature": 0.7
        }
    }'
```
</Tab>


</Tabs>


  This flexibility allows you to optimize RAG performance for your specific use case and leverage the strengths of various LLM providers.
</Accordion>
</AccordionGroup>

Behind the scenes, R2R's RetrievalService handles RAG requests, combining the power of vector search, optional knowledge graph integration, and language model generation. The flexible architecture allows for easy customization and extension of the RAG pipeline to meet diverse requirements.


## User Auth

R2R provides robust user auth and management capabilities. This section briefly covers user authentication features and how they relate to document management.

<AccordionGroup>

<Accordion icon="user-plus" title="User Registration">
To register a new user:

<Tabs>
<Tab title="Python">
```python
from r2r import R2RClient

client = R2RClient("http://localhost:7272")
register_response = client.register("test@example.com", "password123")
print(f"Registration response: {register_response}")
```
</Tab>

<Tab title="Curl">
```bash
curl -X POST http://localhost:7272/v2/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "password123"
  }'
```
</Tab>

<Tab title="JavaScript">
```javascript
await client.register("test@gmail.com", "password123")
```
</Tab>
</Tabs>

Example output:

```bash
{
  'results': {
    'email': 'test@example.com',
    'id': '60af344f-7bd2-43c9-98fd-da53fe5e6d05',
    'is_superuser': False,
    'is_active': True,
    'is_verified': False,
    'verification_code_expiry': None,
    'name': None,
    'bio': None,
    'profile_picture': None,
    'created_at': '2024-07-16T21:50:57.017675Z', 'updated_at': '2024-07-16T21:50:57.017675Z'
  }
}
```


</Accordion>

<Accordion icon="envelope" title="Email Verification">
After registration, users need to verify their email:

<Tabs>
<Tab title="Python">
```python
verify_response = client.verify_email("123456")  # Verification code sent to email
print(f"Email verification response: {verify_response}")
```
</Tab>

<Tab title="Curl">
```bash
curl -X POST http://localhost:7272/v2/verify_email/123456
```
</Tab>

<Tab title="JavaScript">
```javascript
await client.verifyEmail("123456")
```
</Tab>
</Tabs>
</Accordion>

<Accordion icon="arrow-right-to-bracket" title="User Login">
To log in and obtain access tokens:

<Tabs>
<Tab title="Python">
```python
login_response = client.login("test@example.com", "password123")
print(f"Login response: {login_response}")
```
</Tab>

<Tab title="Curl">
```bash
curl -X POST http://localhost:7272/v2/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test@example.com&password=password123"
```
</Tab>

<Tab title="JavaScript">
```javascript
await client.login("test@example.com", "password123")
```
</Tab>
</Tabs>




```bash
# Note, verification is False in default settings
Registration response: {
  'results': {
    'access_token': {
      'token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0cXFAZXhhbXBsZS5jb20iLCJleHAiOjE3MjExOTU3NDQuNzQ1MTM0LCJ0b2tlbl90eXBlIjoiYWNjZXNzIn0.-HrQlguPW4EmPupOYyn5793luaDb-YhEpEsIyQ2CbLs',
      'token_type': 'access'
    },
    'refresh_token': {
      'token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0cXFAZXhhbXBsZS5jb20iLCJleHAiOjE3MjE3NzE3NDQsInRva2VuX3R5cGUiOiJyZWZyZXNoIn0.auuux_0Gg6_b5gTlUOQVCcdPuZl0eM-NFlC1OHdBqiE',
      'token_type': 'refresh'
    }
  }
}
```
</Accordion>

<Accordion icon="user" title="Get Current User Info">
To retrieve information about the currently authenticated user:

<Tabs>
<Tab title="Python">
```python
# requires client.login(...)
user_response = client.user()["results"]
print(f"Current user: {user_response}")
```
</Tab>

<Tab title="Curl">
```bash
curl -X GET http://localhost:7272/v2/user \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```
</Tab>

<Tab title="JavaScript">
```javascript
await client.usersOverview()
```
</Tab>
</Tabs>

```bash
# Note, verification is False in default settings
Current user: {
  'results': {
    'access_token': {
      'token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0cXFAZXhhbXBsZS5jb20iLCJleHAiOjE3MjExOTU3NDQuNzQ1MTM0LCJ0b2tlbl90eXBlIjoiYWNjZXNzIn0.-HrQlguPW4EmPupOYyn5793luaDb-YhEpEsIyQ2CbLs',
      'token_type': 'access'
    },
    'refresh_token': {
      'token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0cXFAZXhhbXBsZS5jb20iLCJleHAiOjE3MjE3NzE3NDQsInRva2VuX3R5cGUiOiJyZWZyZXNoIn0.auuux_0Gg6_b5gTlUOQVCcdPuZl0eM-NFlC1OHdBqiE',
      'token_type': 'refresh'
    }
  }
}
```


</Accordion>

<Accordion icon="magnifying-glass" title="User-Specific Search">
Once authenticated, search results are automatically filtered to include only documents associated with the current user:

<Tabs>
<Tab title="Python">
```python
# requires client.login(...)
search_response = client.search(query="Who was Aristotle")["results"]
print(f"Search results: {search_response}")
```
</Tab>

<Tab title="Curl">
```bash
curl -X POST http://localhost:7272/v2/search \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Who was Aristotle"
  }'
```
</Tab>

<Tab title="JavaScript">
```javascript
await client.search("Who was Aristotle")
```
</Tab>
</Tabs>
```bash
# search results are empty for a new user
Search results: {'vector_search_results': [], 'kg_search_results': []}
```

</Accordion>

<Accordion icon="arrows-rotate" title="Refresh Access Token">
To refresh an expired access token:

<Tabs>
<Tab title="Python">
```python
# requires client.login(...)
refresh_response = client.refresh_access_token()["results"]
print(f"Token refresh response: {refresh_response}")
```
</Tab>

<Tab title="Curl">
```bash
curl -X POST http://localhost:7272/v2/refresh_access_token \
  -H "Authorization: Bearer YOUR_REFRESH_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "YOUR_REFRESH_TOKEN"
  }'
```
</Tab>

<Tab title="JavaScript">
```javascript
await client.refreshAccessToken()
```
</Tab>
</Tabs>
```bash
Token refresh response:
{
  'access_token': {
    'token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0cXFAZXhhbXBsZS5jb20iLCJleHAiOjE3MjExOTU5NTYuODEzNDg0LCJ0b2tlbl90eXBlIjoiYWNjZXNzIn0.-CJy_cH7DRH5FKpZZauAFPP4mncnSa1j8NnaM7utGHo',
    'token_type': 'access'
  },
  'refresh_token': {
    'token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0cXFAZXhhbXBsZS5jb20iLCJleHAiOjE3MjE3NzE5NTYsInRva2VuX3R5cGUiOiJyZWZyZXNoIn0.uGsgTYaUd3Mn5h24uE4ydCWhOr2vFNA9ziRAAaYgnfk',
    'token_type': 'refresh'
  }
}
```
</Accordion>

<Accordion icon="arrow-right-from-bracket" title="User Logout">
To log out and invalidate the current access token:

<Tabs>
<Tab title="Python">
```python
# requires client.login(...)
logout_response = client.logout()
print(f"Logout response: {logout_response}")
```
</Tab>

<Tab title="Curl">
```bash
curl -X POST http://localhost:7272/v2/logout \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```
</Tab>

<Tab title="JavaScript">
```javascript
await client.logout()
```
</Tab>
</Tabs>
```bash
{
  'results': {'message': 'Logged out successfully'}
}
```

</Accordion>

</AccordionGroup>

These authentication features ensure that users can only access and manage their own documents. When performing operations like search, RAG, or document management, the results are automatically filtered based on the authenticated user's permissions.

Remember to replace `YOUR_ACCESS_TOKEN` and `YOUR_REFRESH_TOKEN` with actual tokens obtained during the login process.

## Observability and Analytics

R2R provides robust observability and analytics features, allowing superusers to monitor system performance, track usage patterns, and gain insights into the RAG application's behavior. These advanced features are crucial for maintaining and optimizing your R2R deployment.

<Note>
Observability and analytics features are restricted to superusers only. By default, R2R is configured to treat unauthenticated users as superusers for quick testing and development. In a production environment, you should disable this setting and properly manage superuser access.
</Note>

<AccordionGroup>



<Accordion icon="user-group" title="Users Overview" defaultOpen={true}>
R2R offers high level user observability for superusers

<Tabs>

<Tab title="CLI">
```bash
r2r users-overview
```
</Tab>


<Tab title="Python">
```python
client.users_overview()
```
</Tab>

<Tab title="JavaScript">
```javascript
await  client.users_overview()
```
</Tab>

<Tab title="Curl">

```bash
curl -X POST http://localhost:7272/v2/users_overview \
  -H "Content-Type: application/json" \
  -d '{}'
```
</Tab>

</Tabs>



This command returns detailed log user information, here's some example output:

```bash
{'results': [{'user_id': '2acb499e-8428-543b-bd85-0d9098718220', 'num_files': 9, 'total_size_in_bytes': 4027056, 'document_ids': ['9fbe403b-c11c-5aae-8ade-ef22980c3ad1', 'e0fc8bbc-95be-5a98-891f-c17a43fa2c3d', 'cafdf784-a1dc-5103-8098-5b0a97db1707', 'b21a46a4-2906-5550-9529-087697da2944', '9fbe403b-c11c-5aae-8ade-ef22980c3ad1', 'f17eac52-a22e-5c75-af8f-0b25b82d43f8', '022fdff4-f87d-5b0c-82e4-95d53bcc4e60', 'c5b31b3a-06d2-553e-ac3e-47c56139b484', 'e0c2de57-171d-5385-8081-b546a2c63ce3']}, ...]}}
```


This summary returns information for each user about their number of files ingested, the total size of user ingested files, and the corresponding document ids.
</Accordion>

<Accordion icon="chart-line" title="Logging" defaultOpen={true}>
R2R automatically logs various events and metrics during its operation. You can access these logs using the `logs` command:


<Tabs>
<Tab title="CLI">
```bash
r2r logs
```
</Tab>

<Tab title="Python">
```python
client.logs()
```
</Tab>

<Tab title="JavaScript">
```javascript
await client.logs()
```
</Tab>

<Tab title="Curl">

```bash
curl -X POST http://localhost:7272/v2/logs \
  -H "Content-Type: application/json" \
  -d '{
    "log_type_filter": null,
    "max_runs_requested": 100
  }'
```
</Tab>

</Tabs>



This command returns detailed log entries for various operations, including search and RAG requests. Here's an example of a log entry:

```python
{
    'run_id': UUID('27f124ad-6f70-4641-89ab-f346dc9d1c2f'),
    'run_type': 'rag',
    'entries': [
        {'key': 'search_results', 'value': '["{\\"id\\":\\"7ed3a01c-88dc-5a58-a68b-6e5d9f292df2\\",...}"]'},
        {'key': 'search_query', 'value': 'Who is aristotle?'},
        {'key': 'rag_generation_latency', 'value': '3.79'},
        {'key': 'llm_response', 'value': 'Aristotle (Greek: Ἀριστοτέλης Aristotélēs; 384–322 BC) was...'}
    ]
}
```

These logs provide detailed information about each operation, including search results, queries, latencies, and LLM responses.
</Accordion>

<Accordion icon="chart-pie" title="Analytics">
R2R offers an analytics feature that allows you to aggregate and analyze log data. You can use the `analytics` command to retrieve various statistics:


<Tabs>

<Tab title="Python">
```python
client.analytics(
  {"search_latencies": "search_latency"},
  {"search_latencies": ["basic_statistics", "search_latency"]}
)
```
</Tab>

<Tab title="JavaScript">
```javascript
const filterCriteria = {
    filters: {
      search_latencies: "search_latency",
    },
  };

  const analysisTypes = {
    search_latencies: ["basic_statistics", "search_latency"],
  };

  await client.analytics(filterCriteria, analysisTypes);
```
</Tab>

<Tab title="Curl">

```bash
curl -X POST http://localhost:7272/v2/analytics \
  -H "Content-Type: application/json" \
  -d '{
    "filter_criteria": {
      "filters": {
        "search_latencies": "search_latency"
      }
    },
    "analysis_types":
    {
        "analysis_types": {
            "search_latencies": ["basic_statistics", "search_latency"]
        }
    }
  }'
```
</Tab>


</Tabs>


This command returns aggregated statistics based on the specified filters and analysis types. Here's an example output:

```python
{
    'results': {
        'filtered_logs': {
            'search_latencies': [
                {
                    'timestamp': '2024-06-20 21:29:06',
                    'log_id': UUID('0f28063c-8b87-4934-90dc-4cd84dda5f5c'),
                    'key': 'search_latency',
                    'value': '0.66',
                    'rn': 3
                }
            ]
        },
        'search_latencies': {
          'Mean': 0.66,
            'Median': 0.66,
            'Mode': 0.66,
            'Standard Deviation': 0,
            'Variance': 0
        }
    }
}
```

This analytics feature allows you to:
1. Filter logs based on specific criteria
2. Perform statistical analysis on various metrics (e.g., search latencies)
3. Track performance trends over time
4. Identify potential bottlenecks or areas for optimization
</Accordion>

<Accordion icon="magnifying-glass-chart" title="Custom Analytics">
R2R's analytics system is flexible and allows for custom analysis. You can specify different filters and analysis types to focus on specific aspects of your application's performance. For example:

- Analyze RAG latencies
- Track usage patterns by user or document type
- Monitor error rates and types
- Assess the effectiveness of different LLM models or configurations

To perform custom analytics, modify the `filters` and `analysis_types` parameters in the `analytics` command to suit your specific needs.
</Accordion>

</AccordionGroup>

These observability and analytics features provide valuable insights into your R2R application's performance and usage, enabling data-driven optimization and decision-making.

---


## Introduction

R2R's hybrid search combines traditional keyword-based searching with modern semantic understanding, providing more accurate and contextually relevant results. This approach is particularly effective for complex queries where both specific terms and overall meaning are crucial.

## How R2R Hybrid Search Works

1. **Full-Text Search**: Utilizes PostgreSQL's full-text search with `ts_rank_cd` and `websearch_to_tsquery`.
2. **Semantic Search**: Performs vector similarity search using the query's embedded representation.
3. **Reciprocal Rank Fusion (RRF)**: Merges results from full-text and semantic searches using the formula:
   ```
   COALESCE(1.0 / (rrf_k + full_text.rank_ix), 0.0) * full_text_weight +
   COALESCE(1.0 / (rrf_k + semantic.rank_ix), 0.0) * semantic_weight
   ```
4. **Result Ranking**: Orders final results based on the combined RRF score.

## Key Features

### Full-Text Search

The full-text search component incorporates:

- PostgreSQL's `tsvector` for efficient text searching
- `websearch_to_tsquery` for parsing user queries
- `ts_rank_cd` for ranking full-text search results

### Semantic Search

The semantic search component uses:

- Vector embeddings for storing and querying semantic representations
- Cosine similarity for measuring the relevance of documents to the query

## Configuration

### VectorSearchSettings

Key settings for vector search configuration:

```python
class VectorSearchSettings(BaseModel):
    use_hybrid_search: bool
    search_limit: int
    filters: dict[str, Any]
    hybrid_search_settings: Optional[HybridSearchSettings]
    # ... other settings
```

### HybridSearchSettings

Specific parameters for hybrid search:

```python
class HybridSearchSettings(BaseModel):
    full_text_weight: float
    semantic_weight: float
    full_text_limit: int
    rrf_k: int
```

## Usage Example

```python
from r2r import R2RClient

client = R2RClient()

vector_settings = {
    "use_hybrid_search": True,
    "search_limit": 20,
    # Can add logical filters, as shown:
    # "filters":{"category": {"$eq": "technology"}},
    "hybrid_search_settings" : {
        "full_text_weight": 1.0,
        "semantic_weight": 5.0,
        "full_text_limit": 200,
        "rrf_k": 50
    }
}

results = client.search(
    query="Who was Aristotle?",
    vector_search_settings=vector_settings
)
print(results)
```

## Results Comparison

### Basic Vector Search

```json
{
  "results": {
    "vector_search_results": [
      {
        "score": 0.780314067545999,
        "text": "Aristotle[A] (Greek: Ἀριστοτέλης Aristotélēs, pronounced [aristotélɛːs]; 384–322 BC) was an Ancient Greek philosopher and polymath...",
        "metadata": {
          "title": "aristotle.txt",
          "version": "v0",
          "chunk_order": 0
        }
      }, ...
    ]
  }
}
```

### Hybrid Search with RRF

```json
{
  "results": {
    "vector_search_results": [
      {
        "score": 0.0185185185185185,
        "text": "Aristotle[A] (Greek: Ἀριστοτέλης Aristotélēs, pronounced [aristotélɛːs]; 384–322 BC) was an Ancient Greek philosopher and polymath...",
        "metadata": {
          "title": "aristotle.txt",
          "version": "v0",
          "chunk_order": 0,
          "semantic_rank": 1,
          "full_text_rank": 3
        }, ...
      }
    ]
  }
}
```

## Best Practices

1. Optimize PostgreSQL indexing for both full-text and vector searches
2. Regularly update search indices
3. Monitor performance and adjust weights as needed
4. Use appropriate vector dimensions and embedding models for your use case

## Conclusion

R2R's hybrid search offers a powerful solution for complex information retrieval needs, combining the strengths of keyword matching and semantic understanding. Its flexible configuration and use of Reciprocal Rank Fusion make it adaptable to a wide range of use cases, from technical documentation to broad, context-dependent queries.

---


## Introduction

R2R provides a complete set of user authentication and management features, allowing developers to implement secure and feature-rich authentication systems, or to integrate directly with their authentication provider of choice.

[Refer here](/documentation/deep-dive/providers/auth) for documentation on the available authentication provider options built into R2R, or [refer here](/api-reference/endpoint/register) for available auth API reference.


<Note>
When authentication is not required (require_authentication is set to false, which is the default in `r2r.toml`), unauthenticated requests will default to using the credentials of the default admin user.

This behavior ensures that operations can proceed smoothly in development or testing environments where authentication may not be enforced, but it should be used with caution in production settings.
</Note>


## Setup

Before diving into the authentication features, ensure you have R2R installed and configured as described in the [installation guide](/documentation/installation). For this guide, we'll use the default configuration. Further, `r2r serve` must be called to serve R2R in either your local environment or local Docker engine.

## Basic Usage

### User Registration and Login

Let's start by registering a new user and logging in:

```python core/examples/scripts/run_auth_workflow.py
from r2r import R2RClient

client = R2RClient("http://localhost:7272") # Replace with your R2R deployment URL

# Register a new user
user_result = client.register("user1@test.com", "password123")
# {'results': {'email': 'user1@test.com', 'id': 'bf417057-f104-4e75-8579-c74d26fcbed3', 'hashed_password': '$2b$12$p6a9glpAQaq.4uzi4gXQru6PN7WBpky/xMeYK9LShEe4ygBf1L.pK', 'is_superuser': False, 'is_active': True, 'is_verified': False, 'verification_code_expiry': None, 'name': None, 'bio': None, 'profile_picture': None, 'created_at': '2024-07-16T22:53:47.524794Z', 'updated_at': '2024-07-16T22:53:47.524794Z'}}

# Login immediately (assuming email verification is disabled)
login_result = client.login("user1@test.com", "password123")
# {'results': {'access_token': {'token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyMUB0ZXN0LmNvbSIsImV4cCI6MTcyMTE5OTI0Ni44Nzc2NTksInRva2VuX3R5cGUiOiJhY2Nlc3MifQ.P4RcCkCe0H5UHPHak7tRovIgyQcql4gB8NlqdDDk50Y', 'token_type': 'access'}, 'refresh_token': {'token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyMUB0ZXN0LmNvbSIsImV4cCI6MTcyMTc3NTI0NiwidG9rZW5fdHlwZSI6InJlZnJlc2gifQ.VgfZ4Lhz0f2GW41NYv6KLrMCK3CdGmGVug7eTQp0xPU', 'token_type': 'refresh'}}}
```

This code snippet demonstrates the basic user registration and login process. The `register` method creates a new user account, while the `login` method authenticates the user and returns access and refresh tokens. In the example above, it was assumed that email verification was disabled.

### Email Verification (Optional)

If email verification is enabled in your R2R configuration, you'll need to verify the user's email before they can log in:

```python
verify_result = client.verify_email("verification_code_here")
# {"results": {"message": "Email verified successfully"}}
```

### Token Refresh

After logging in, you gain immediate access to user information such as general account details, documents overview, and utility functions like token refresh:

```python
refresh_result = client.refresh_access_token()
# {'results': {'access_token': {'token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyMUB0ZXN0LmNvbSIsImV4cCI6MTcyMTIwMTk3Mi4zMzI4NTIsInRva2VuX3R5cGUiOiJhY2Nlc3MifQ.Ze9A50kefndAtu2tvcvMCiilFfAhrOV0l5A7RZgPvBY', 'token_type': 'access'}, 'refresh_token': {'token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyMUB0ZXN0LmNvbSIsImV4cCI6MTcyMTc3Nzk3MiwidG9rZW5fdHlwZSI6InJlZnJlc2gifQ.NwzFH8e2tKO0bH1Hdm_eq39VqmGPf7xSNOOhDlKFQFQ', 'token_type': 'refresh'}}}
```


## Document Management

R2R allows users to manage their documents securely. Here's how to ingest and search a given users documents:

### Ingesting Documents

```python
import os

# Ingest a sample file for the logged-in user
script_path = os.path.dirname(__file__)
sample_file = os.path.join(script_path, "..", "data", "aristotle.txt")
ingestion_result = client.ingest_files([sample_file])
# {'results': {'processed_documents': ["Document 'aristotle.txt' processed successfully."], 'failed_documents': [], 'skipped_documents': []}}
```

### User Document Overview
```python
documents_overview = client.documents_overview()
# {'results': [{'document_id': '6ab698c6-e494-5441-a740-49395f2b1881', 'version': 'v0', 'size_in_bytes': 73353, 'metadata': {}, 'status': 'success', 'user_id': 'ba0c75eb-0b21-4eb1-a902-082476e5e972', 'title': 'aristotle.txt', 'created_at': '2024-07-16T16:25:27.634411Z', 'updated_at': '2024-07-16T16:25:27.634411Z'}]}
```
### Search & RAG

```python
search_result = client.search(query="Sample search query")
# {'results': {'vector_search_results': [{ ... 'metadata': {'text': 'Aristotle[A] (Greek: Ἀριστοτέλης Aristotélēs, pronounced [aristotélɛːs]; 384–322 BC) was an Ancient Greek philosopher and polymath. His writings cover a broad range of subjects spanning the natural sciences, philosophy, linguistics, economics, politics, psychology, and the arts. As the founder of the Peripatetic school of philosophy in the Lyceum in Athens, he began the wider Aristotelian tradition that followed, which set the groundwork for the development of modern science.', 'title': 'aristotle.txt', 'user_id': 'bf417057-f104-4e75-8579-c74d26fcbed3', 'version': 'v0', 'chunk_order': 0, 'document_id': 'a2645197-d07f-558d-ba55-f7a60eb29621', 'extraction_id': 'b7bbd497-311a-5dc8-8a51-79e2208739e0', 'associatedQuery': 'Who was Aristotle'}}, {'id': '781ce9e6-9e73-5012-8445-35b7d84f161c', 'score': 0.670799394202279, 'metadata': {'text': "Aristotle was born in 384 BC[C] in Stagira, Chalcidice,[2] about 55 km (34 miles) east of modern-day Thessaloniki.[3][4] His father, Nicomachus, was the personal physician to King Amyntas of Macedon. While he was young, Aristotle learned about biology and medical information, which was taught by his father.[5] Both of Aristotle's parents died when he was about thirteen, and Proxenus of Atarneus became his guardian.[6] Although little information about Aristotle's childhood has survived, he probably spent", 'title': 'aristotle.txt', 'user_id': 'bf417057-f104-4e75-8579-c74d26fcbed3', 'version': 'v0', 'chunk_order': 8, 'document_id': 'a2645197-d07f-558d-ba55-f7a60eb29621', 'extraction_id': 'b7bbd497-311a-5dc8-8a51-79e2208739e0', 'associatedQuery': 'Who was Aristotle'}}, {'id': 'f32cda7c-2538-5248-b0b6-4d0d45cc4d60', 'score': 0.667974928858889, 'metadata': {'text': 'Aristotle was revered among medieval Muslim scholars as "The First Teacher", and among medieval Christians like Thomas Aquinas as simply "The Philosopher", while the poet Dante called him "the master of those who know". His works contain the earliest known formal study of logic, and were studied by medieval scholars such as Peter Abelard and Jean Buridan. Aristotle\'s influence on logic continued well into the 19th century. In addition, his ethics, although always influential, gained renewed interest with', 'title': 'aristotle.txt', 'user_id': 'bf417057-f104-4e75-8579-c74d26fcbed3', 'version': 'v0', 'chunk_order': 5, 'document_id': 'a2645197-d07f-558d-ba55-f7a60eb29621', 'extraction_id': 'b7bbd497-311a-5dc8-8a51-79e2208739e0', 'associatedQuery': 'Who was Aristotle'}}, {'id': 'e6592fd5-e02e-5847-b158-79bbdd8710a2', 'score': 0.6647597950983339, 'metadata': {'text': "Little is known about Aristotle's life. He was born in the city of Stagira in northern Greece during the Classical period. His father, Nicomachus, died when Aristotle was a child, and he was brought up by a guardian. At 17 or 18, he joined Plato's Academy in Athens and remained there until the age of 37 (c.\u2009347 BC). Shortly after Plato died, Aristotle left Athens and, at the request of Philip II of Macedon, tutored his son Alexander the Great beginning in 343 BC. He established a library in the Lyceum,", 'title': 'aristotle.txt', 'user_id': 'bf417057-f104-4e75-8579-c74d26fcbed3', 'version': 'v0', 'chunk_order': 1, 'document_id': 'a2645197-d07f-558d-ba55-f7a60eb29621', 'extraction_id': 'b7bbd497-311a-5dc8-8a51-79e2208739e0', 'associatedQuery': 'Who was Aristotle'}}, {'id': '8c72faca-6d98-5129-b9ee-70769272e361', 'score': 0.6476034942146001, 'metadata': {'text': 'Among countless other achievements, Aristotle was the founder of formal logic,[146] pioneered the study of zoology, and left every future scientist and philosopher in his debt through his contributions to the scientific method.[2][147][148] Taneli Kukkonen, observes that his achievement in founding two sciences is unmatched, and his reach in influencing "every branch of intellectual enterprise" including Western ethical and political theory, theology, rhetoric, and literary analysis is equally long. As a', 'title': 'aristotle.txt', 'user_id': 'bf417057-f104-4e75-8579-c74d26fcbed3', 'version': 'v0', 'chunk_order': 175, 'document_id': 'a2645197-d07f-558d-ba55-f7a60eb29621', 'extraction_id': 'b7bbd497-311a-5dc8-8a51-79e2208739e0', 'associatedQuery': 'Who was Aristotle'}}, {'id': '3ce904cc-5835-551a-a85c-f00be1a5e8dc', 'score': 0.626156434278918, 'metadata': {'text': 'Aristotle has been called the father of logic, biology, political science, zoology, embryology, natural law, scientific method, rhetoric, psychology, realism, criticism, individualism, teleology, and meteorology.[151]', 'title': 'aristotle.txt', 'user_id': 'bf417057-f104-4e75-8579-c74d26fcbed3', 'version': 'v0', 'chunk_order': 177, 'document_id': 'a2645197-d07f-558d-ba55-f7a60eb29621', 'extraction_id': 'b7bbd497-311a-5dc8-8a51-79e2208739e0', 'associatedQuery': 'Who was Aristotle'}}, {'id': '6a15b09b-4bf1-5c1f-af24-fe659c8a011d', 'score': 0.624521989361129, 'metadata': {'text': 'after friends and relatives, and to deal with the latter as with beasts or plants".[13] By 335 BC, Aristotle had returned to Athens, establishing his own school there known as the Lyceum. Aristotle conducted courses at the school for the next twelve years. While in Athens, his wife Pythias died and Aristotle became involved with Herpyllis of Stagira. They had a son whom Aristotle named after his father, Nicomachus. If the Suda – an uncritical compilation from the Middle Ages – is accurate, he may also have', 'title': 'aristotle.txt', 'user_id': 'bf417057-f104-4e75-8579-c74d26fcbed3', 'version': 'v0', 'chunk_order': 16, 'document_id': 'a2645197-d07f-558d-ba55-f7a60eb29621', 'extraction_id': 'b7bbd497-311a-5dc8-8a51-79e2208739e0', 'associatedQuery': 'Who was Aristotle'}}, {'id': '19a755d0-770f-5c6f-991e-ca191a40c8d6', 'score': 0.614493374720815, 'metadata': {'text': "passed to Plato's nephew Speusippus, although it is possible that he feared the anti-Macedonian sentiments in Athens at that time and left before Plato died.[10] Aristotle then accompanied Xenocrates to the court of his friend Hermias of Atarneus in Asia Minor. After the death of Hermias, Aristotle travelled with his pupil Theophrastus to the island of Lesbos, where together they researched the botany and zoology of the island and its sheltered lagoon. While in Lesbos, Aristotle married Pythias, either", 'title': 'aristotle.txt', 'user_id': 'bf417057-f104-4e75-8579-c74d26fcbed3', 'version': 'v0', 'chunk_order': 12, 'document_id': 'a2645197-d07f-558d-ba55-f7a60eb29621', 'extraction_id': 'b7bbd497-311a-5dc8-8a51-79e2208739e0', 'associatedQuery': 'Who was Aristotle'}}, {'id': '33b2dbd7-2f3a-5450-9618-976a996bde2a', 'score': 0.6117302824500019, 'metadata': {'text': 'Transmission\nFurther information: List of writers influenced by Aristotle\nMore than 2300 years after his death, Aristotle remains one of the most influential people who ever lived.[142][143][144] He contributed to almost every field of human knowledge then in existence, and he was the founder of many new fields. According to the philosopher Bryan Magee, "it is doubtful whether any human being has ever known as much as he did".[145]', 'title': 'aristotle.txt', 'user_id': 'bf417057-f104-4e75-8579-c74d26fcbed3', 'version': 'v0', 'chunk_order': 174, 'document_id': 'a2645197-d07f-558d-ba55-f7a60eb29621', 'extraction_id': 'b7bbd497-311a-5dc8-8a51-79e2208739e0', 'associatedQuery': 'Who was Aristotle'}}, {'id': '2d101d42-6317-5d8c-85c3-fb9b6d947c68', 'score': 0.610827455968717, 'metadata': {'text': "The immediate influence of Aristotle's work was felt as the Lyceum grew into the Peripatetic school. Aristotle's students included Aristoxenus, Dicaearchus, Demetrius of Phalerum, Eudemos of Rhodes, Harpalus, Hephaestion, Mnason of Phocis, Nicomachus, and Theophrastus. Aristotle's influence over Alexander the Great is seen in the latter's bringing with him on his expedition a host of zoologists, botanists, and researchers. He had also learned a great deal about Persian customs and traditions from his", 'title': 'aristotle.txt', 'user_id': 'bf417057-f104-4e75-8579-c74d26fcbed3', 'version': 'v0', 'chunk_order': 181, 'document_id': 'a2645197-d07f-558d-ba55-f7a60eb29621', 'extraction_id': 'b7bbd497-311a-5dc8-8a51-79e2208739e0', 'associatedQuery': 'Who was Aristotle'}}], 'kg_search_results': []}}

rag_result = client.rag(query="Sample search query")
# {'results': {'completion': {'id': 'chatcmpl-9llkGYsrG1YZaWkqYvzXr1eQNl0gA', 'choices': [{'finish_reason': 'stop', 'index': 0, 'message': {'content': 'The search results for the query "Sample search query" include various topics and excerpts related to Aristotle\'s works and other subjects. Here are the relevant references:\n\n1. **Categories of Aristotle\'s Works**:\n   - On Interpretation [1], [2]\n   - Prior Analytics [1], [2]\n   - Posterior Analytics [1], [2]\n   - Topics [1], [2]\n   - On Sophistical Refutations [1], [2]\n\n2. **Aristotle\'s Theory on Sense Perceptions and Memory**:\n   - Aristotle\'s belief that people receive sense perceptions and perceive them as impressions, leading to the weaving together of new experiences. The search for these impressions involves searching the memory itself, where recollection occurs when one retrieved experience naturally follows another [3], [4].\n\n3. **Medieval Judaism**:\n   - References to Medieval Judaism [5], [6].\n\n4. **Scientific Style**:\n   - References to Scientific Style [7], [8].\n\n5. **Recovery of Texts by Apellicon**:\n   - Apellicon\'s efforts to recover degraded texts by copying them into new manuscripts and using guesswork to fill in unreadable gaps [9], [10].\n\nThese references provide a broad overview of the topics related to the query, including Aristotle\'s works, his theories on memory, Medieval Judaism, scientific style, and the recovery of ancient texts.', 'role': 'assistant'}}], 'created': 1721171976, 'model': 'gpt-4o-2024-05-13', 'object': 'chat.completion', 'system_fingerprint': 'fp_5e997b69d8', 'usage': {'completion_tokens': 286, 'prompt_tokens': 513, 'total_tokens': 799}}, 'search_results': {'vector_search_results': [{'id': 'd70e2776-befa-5b67-9da7-b76aedb7c101', 'score': 0.270276627830369, 'metadata': {'text': 'Categories\nOn Interpretation\nPrior Analytics\nPosterior Analytics\nTopics\nOn Sophistical Refutations', 'title': 'aristotle.txt', 'user_id': '76eea168-9f98-4672-af3b-2c26ec92d7f8', 'version': 'v0', 'chunk_order': 26, 'document_id': '4bb1e5e0-3bb3-54e0-bc71-69e68bce30c7', 'extraction_id': '9401dfe6-10dd-5eb1-8b88-de1927a6c556', 'associatedQuery': 'Sample search query'}}, {'id': 'f54c9cda-0053-5ea2-a22b-aaba6437518c', 'score': 0.270276627830369, 'metadata': {'text': 'Categories\nOn Interpretation\nPrior Analytics\nPosterior Analytics\nTopics\nOn Sophistical Refutations', 'title': 'aristotle.txt', 'user_id': '2acb499e-8428-543b-bd85-0d9098718220', 'version': 'v0', 'chunk_order': 26, 'document_id': '9fbe403b-c11c-5aae-8ade-ef22980c3ad1', 'extraction_id': 'bc497a0c-4b17-5e86-97d4-aa06474e0e5b', 'associatedQuery': 'Sample search query'}}, {'id': 'd0675bcd-23d1-5982-8114-1a6459faec3f', 'score': 0.242980153623792, 'metadata': {'text': 'Because Aristotle believes people receive all kinds of sense perceptions and perceive them as impressions, people are continually weaving together new impressions of experiences. To search for these impressions, people search the memory itself.[105] Within the memory, if one experience is offered instead of a specific memory, that person will reject this experience until they find what they are looking for. Recollection occurs when one retrieved experience naturally follows another. If the chain of', 'title': 'aristotle.txt', 'user_id': '76eea168-9f98-4672-af3b-2c26ec92d7f8', 'version': 'v0', 'chunk_order': 119, 'document_id': '4bb1e5e0-3bb3-54e0-bc71-69e68bce30c7', 'extraction_id': '9401dfe6-10dd-5eb1-8b88-de1927a6c556', 'associatedQuery': 'Sample search query'}}, {'id': '69aed771-061f-5360-90f1-0ce395601b98', 'score': 0.242980153623792, 'metadata': {'text': 'Because Aristotle believes people receive all kinds of sense perceptions and perceive them as impressions, people are continually weaving together new impressions of experiences. To search for these impressions, people search the memory itself.[105] Within the memory, if one experience is offered instead of a specific memory, that person will reject this experience until they find what they are looking for. Recollection occurs when one retrieved experience naturally follows another. If the chain of', 'title': 'aristotle.txt', 'user_id': '2acb499e-8428-543b-bd85-0d9098718220', 'version': 'v0', 'chunk_order': 119, 'document_id': '9fbe403b-c11c-5aae-8ade-ef22980c3ad1', 'extraction_id': 'bc497a0c-4b17-5e86-97d4-aa06474e0e5b', 'associatedQuery': 'Sample search query'}}, {'id': 'dadd2d48-a2b7-5e55-9a8c-1030712c5ca0', 'score': 0.20218510005651702, 'metadata': {'text': 'Medieval Judaism', 'title': 'aristotle.txt', 'user_id': '76eea168-9f98-4672-af3b-2c26ec92d7f8', 'version': 'v0', 'chunk_order': 202, 'document_id': '4bb1e5e0-3bb3-54e0-bc71-69e68bce30c7', 'extraction_id': '9401dfe6-10dd-5eb1-8b88-de1927a6c556', 'associatedQuery': 'Sample search query'}}, {'id': 'da81f692-40d9-599b-a69b-25b6a5179b47', 'score': 0.20218510005651702, 'metadata': {'text': 'Medieval Judaism', 'title': 'aristotle.txt', 'user_id': '2acb499e-8428-543b-bd85-0d9098718220', 'version': 'v0', 'chunk_order': 202, 'document_id': '9fbe403b-c11c-5aae-8ade-ef22980c3ad1', 'extraction_id': 'bc497a0c-4b17-5e86-97d4-aa06474e0e5b', 'associatedQuery': 'Sample search query'}}, {'id': '0c4fea20-f7ee-520f-ae1f-155ecb398e1f', 'score': 0.19056136124594703, 'metadata': {'text': 'Scientific style', 'title': 'aristotle.txt', 'user_id': '2acb499e-8428-543b-bd85-0d9098718220', 'version': 'v0', 'chunk_order': 92, 'document_id': '9fbe403b-c11c-5aae-8ade-ef22980c3ad1', 'extraction_id': 'bc497a0c-4b17-5e86-97d4-aa06474e0e5b', 'associatedQuery': 'Sample search query'}}, {'id': 'c3c3145a-5d9d-5362-9629-f9159a027a9d', 'score': 0.19051768949311598, 'metadata': {'text': 'Scientific style', 'title': 'aristotle.txt', 'user_id': '76eea168-9f98-4672-af3b-2c26ec92d7f8', 'version': 'v0', 'chunk_order': 92, 'document_id': '4bb1e5e0-3bb3-54e0-bc71-69e68bce30c7', 'extraction_id': '9401dfe6-10dd-5eb1-8b88-de1927a6c556', 'associatedQuery': 'Sample search query'}}, {'id': '63e3a252-90bd-5494-9f9f-aee772f4db54', 'score': 0.18900877964391904, 'metadata': {'text': 'Apellicon sought to recover the texts, many of which were seriously degraded at this point due to the conditions in which they were stored. He had them copied out into new manuscripts, and used his best guesswork to fill in the gaps where the originals were unreadable.[216]:\u200a5–6', 'title': 'aristotle.txt', 'user_id': '76eea168-9f98-4672-af3b-2c26ec92d7f8', 'version': 'v0', 'chunk_order': 228, 'document_id': '4bb1e5e0-3bb3-54e0-bc71-69e68bce30c7', 'extraction_id': '9401dfe6-10dd-5eb1-8b88-de1927a6c556', 'associatedQuery': 'Sample search query'}}, {'id': '2c1183a8-e130-5432-a311-ee1f0f194562', 'score': 0.18894388145542895, 'metadata': {'text': 'Apellicon sought to recover the texts, many of which were seriously degraded at this point due to the conditions in which they were stored. He had them copied out into new manuscripts, and used his best guesswork to fill in the gaps where the originals were unreadable.[216]:\u200a5–6', 'title': 'aristotle.txt', 'user_id': '2acb499e-8428-543b-bd85-0d9098718220', 'version': 'v0', 'chunk_order': 228, 'document_id': '9fbe403b-c11c-5aae-8ade-ef22980c3ad1', 'extraction_id': 'bc497a0c-4b17-5e86-97d4-aa06474e0e5b', 'associatedQuery': 'Sample search query'}}], 'kg_search_results': None}}}
```

## Advanced Authentication Features

R2R offers several advanced authentication features to enhance security and user experience:

### Password Management

Users can change their passwords and request password resets:

```python
# Change password
change_password_result = client.change_password("password123", "new_password")
# {"result": {"message": "Password changed successfully"}}

# Request password reset
reset_request_result = client.request_password_reset("user@example.com")
# {"result": {"message": "If the email exists, a reset link has been sent"}}

# Confirm password reset (after user receives reset token)
reset_confirm_result = client.confirm_password_reset("reset_token_here", "new_password")
# {"result": {"message": "Password reset successfully"}}
```

### User Profile Management

Users can view and update their profiles:

```python
# Get user profile
profile = client.user()
# {'results': {'email': 'user1@test.com', 'id': '76eea168-9f98-4672-af3b-2c26ec92d7f8', 'hashed_password': 'null', 'is_superuser': False, 'is_active': True, 'is_verified': True, 'verification_code_expiry': None, 'name': None, 'bio': None, 'profile_picture': None, 'created_at': '2024-07-16T23:06:42.123303Z', 'updated_at': '2024-07-16T23:22:48.256239Z'}}

# Update user profile
update_result = client.update_user(name="John Doe", bio="R2R enthusiast")
# {'results': {'email': 'user1@test.com', 'id': '76eea168-9f98-4672-af3b-2c26ec92d7f8', 'hashed_password': 'null', 'is_superuser': False, 'is_active': True, 'is_verified': True, 'verification_code_expiry': None, 'name': 'John Doe', 'bio': 'R2R enthusiast', 'profile_picture': None, 'created_at': '2024-07-16T23:06:42.123303Z', 'updated_at': '2024-07-16T23:22:48.256239Z'}}
```

### Account Deletion

Users can delete their accounts:

```python
# Delete account (requires password confirmation)
user_id = register_response["results"]["id"] # input unique id here
delete_result = client.delete_user(user_id, "password123")
# {'results': {'message': 'User account deleted successfully'}}
```

### Logout

To end a user session:

```python
logout_result = client.logout()
print(f"Logout Result:\n{logout_result}")
# {'results': {'message': 'Logged out successfully'}}
```

## Superuser Capabilities and Default Admin Creation

R2R includes powerful superuser capabilities and a mechanism for default admin creation, which are crucial for system management and initial setup. Let's explore these features:

### Superuser Capabilities

Superusers in R2R have elevated privileges that allow them to perform system-wide operations and access sensitive information. Some key superuser capabilities include:

1. **User Management**: Superusers can view, modify, and delete user accounts.
2. **System-wide Document Access**: They can access and manage documents across all users.
3. **Analytics and Observability**: Superusers have access to system-wide analytics and logs.
4. **Configuration Management**: They can modify system configurations and settings.

To use superuser capabilities, you need to authenticate as a superuser. The methods for accessing these features are the same as regular user methods, but with expanded scope and permissions.

### Default Admin Creation

R2R automatically creates a default admin user during initialization. This process is handled by the `R2RAuthProvider` class. Here's how it works:

1. During system initialization, R2R attempts to create a default admin user.
2. The admin email and password are typically set through environment variables or configuration files.
3. If the admin user already exists, R2R logs this information and continues without creating a duplicate.

The relevant part of the configuration that affects this process is:

```toml
[auth]
provider = "r2r"
access_token_lifetime_in_minutes = 60
refresh_token_lifetime_in_days = 7
require_authentication = true
require_email_verification = false
default_admin_email = "admin@example.com"
default_admin_password = "change_me_immediately"
```

- With `"require_authentication": false`, the system allows unauthenticated access for testing and development. In a production environment, this should be set to `true`.
- `"require_email_verification": false` means that email verification is not required for new users, including the default admin. For increased security in production, consider enabling this.

### Accessing Superuser Features

To access superuser features, you need to authenticate as the default admin or another user with superuser privileges. Here's an example of how to do this:

```python

from r2r import R2RClient

client = R2RClient("http://localhost:7272")

# Login as admin
login_result = client.login("admin@example.com", "change_me_immediately")

# Now you can access superuser features, for example:
users_overview = client.users_overview()
# {'results': [{'user_id': '2acb499e-8428-543b-bd85-0d9098718220', 'num_files': 2, 'total_size_in_bytes': 73672, 'document_ids': ['c4967f03-1780-5161-8e1d-57b55aa65076', '9fbe403b-c11c-5aae-8ade-ef22980c3ad1']}, {'user_id': 'ac730ec3-7d3d-451a-a166-e7ac7c57b198', 'num_files': 1, 'total_size_in_bytes': 73353, 'document_ids': ['d4861e78-cf02-5184-9b6a-d5bdbddd39b2']}, {'user_id': 'e0514342-e51a-43e5-8aaa-665468102dce', 'num_files': 1, 'total_size_in_bytes': 73353, 'document_ids': ['f4fbe534-b7d6-5fec-9d41-9093b2112732']}]}

# Access system-wide logs
logs = client.logs()
# {'results': [{'run_id': '645cb90f-c281-4188-b93b-94fb383170f6', 'run_type': 'search', 'entries': [{'key': 'search_latency', 'value': '0.43'}, { ...

# Perform analytics
analytics_result = client.analytics(
    {"all_latencies": "search_latency"},
    {"search_latencies": ["basic_statistics", "search_latency"]}
)
# {'results': {'filtered_logs': {'search_latencies': [{'timestamp': '2024-07-18 18:10:26', 'log_id': '645cb90f-c281-4188-b93b-94fb383170f6', 'key': 'search_latency', 'value': '0.43', 'rn': 3}, {'timestamp': '2024-07-18 18:04:54', 'log_id': 'a22d6a4c-3e68-4f01-b129-c9cbb5ae0b86', 'key': 'search_latency', 'value': '0.76', 'rn': 3}, {'timestamp': '2024-07-18 17:45:04', 'log_id': '253aa7b2-5abc-46d4-9bc3-f1ee47598bc5', 'key': 'search_latency', 'value': '0.43', 'rn': 3}, {'timestamp': '2024-07-18 17:44:47', 'log_id': 'add3d166-392c-44e0-aec2-d00d97a584f9', 'key': 'search_latency', 'value': '0.71', 'rn': 3}, {'timestamp': '2024-07-18 17:43:40', 'log_id': '9b64d038-aa56-44c9-af5e-96091fff62f2', 'key': 'search_latency', 'value': '0.44', 'rn': 3}, {'timestamp': '2024-07-18 17:43:20', 'log_id': '2a433a26-dc5b-4460-823a-58f01070e44d', 'key': 'search_latency', 'value': '0.37', 'rn': 3}, {'timestamp': '2024-07-18 17:43:16', 'log_id': '71a05fb2-5993-45c3-af3d-5019a745a33d', 'key': 'search_latency', 'value': '0.72', 'rn': 3}, {'timestamp': '2024-07-16 23:19:35', 'log_id': 'fada5559-ccd1-42f3-81a1-c96dcbc2ff08', 'key': 'search_latency', 'value': '0.34', 'rn': 3}, {'timestamp': '2024-07-16 23:07:32', 'log_id': '530bc25c-efc9-4b10-b46f-529c64c89fdf', 'key': 'search_latency', 'value': '0.62', 'rn': 3}, {'timestamp': '2024-07-16 23:07:14', 'log_id': '5c977046-bd71-4d95-80ba-dcf16e33cfd5', 'key': 'search_latency', 'value': '0.72', 'rn': 3}, {'timestamp': '2024-07-16 23:06:44', 'log_id': 'a2bef456-7370-4977-83cf-a60de5abd0cf', 'key': 'search_latency', 'value': '0.44', 'rn': 3}, {'timestamp': '2024-07-16 23:02:10', 'log_id': 'ee8d8bb4-7bc5-4c92-a9a2-c3ddd9f6bcd9', 'key': 'search_latency', 'value': '0.53', 'rn': 3}, {'timestamp': '2024-07-16 22:00:48', 'log_id': '13701f93-a2ab-4182-abd0-a401aa8e720a', 'key': 'search_latency', 'value': '0.52', 'rn': 3}, {'timestamp': '2024-07-16 21:59:17', 'log_id': '17bd24d7-37e2-4838-9830-c92110f492c7', 'key': 'search_latency', 'value': '0.30', 'rn': 3}, {'timestamp': '2024-07-16 21:59:16', 'log_id': 'd1d1551b-80cc-4f93-878a-7d7579aa3b9e', 'key': 'search_latency', 'value': '0.43', 'rn': 3}, {'timestamp': '2024-07-16 21:55:46', 'log_id': '0291254e-262f-4d9d-ace2-b98c2eaae547', 'key': 'search_latency', 'value': '0.42', 'rn': 3}, {'timestamp': '2024-07-16 21:55:45', 'log_id': '74df032f-e9d6-4ba9-ae0e-1be58927c2b1', 'key': 'search_latency', 'value': '0.57', 'rn': 3}, {'timestamp': '2024-07-16 21:54:36', 'log_id': '413982fd-3588-42cc-8009-8c686a85f27e', 'key': 'search_latency', 'value': '0.55', 'rn': 3}, {'timestamp': '2024-07-16 03:35:48', 'log_id': 'ae79062e-f4f0-4fb1-90f4-4ddcb8ae0cc4', 'key': 'search_latency', 'value': '0.50', 'rn': 3}, {'timestamp': '2024-07-16 03:26:10', 'log_id': '9fd51a36-9fdf-4a70-89cb-cb0d43dd0b63', 'key': 'search_latency', 'value': '0.41', 'rn': 3}, {'timestamp': '2024-07-16 03:01:18', 'log_id': '6cb79d2e-c431-447e-bbbb-99f96d56784e', 'key': 'search_latency', 'value': '0.67', 'rn': 3}, {'timestamp': '2024-07-16 01:29:44', 'log_id': '34eea2d3-dd98-47fb-ae3b-05e1001850a5', 'key': 'search_latency', 'value': '0.42', 'rn': 3}, {'timestamp': '2024-07-16 01:29:25', 'log_id': '5204d260-4ad3-49ce-9b38-1043ceae65ac', 'key': 'search_latency', 'value': '0.58', 'rn': 3}]}, 'search_latencies': {'Mean': 0.517, 'Median': 0.5, 'Mode': 0.43, 'Standard Deviation': 0.132, 'Variance': 0.018}}}

```

### Security Considerations for Superusers

When using superuser capabilities, keep the following security considerations in mind:

1. **Limit Superuser Access**: Only grant superuser privileges to trusted individuals who require full system access.
2. **Use Strong Passwords**: Ensure that superuser accounts, especially the default admin, use strong, unique passwords.
3. **Enable Authentication and Verification**: In production, set `"require_authentication": true` and `"require_email_verification": true` for enhanced security.
4. **Audit Superuser Actions**: Regularly review logs of superuser activities to detect any unusual or unauthorized actions.
5. **Rotate Credentials**: Periodically update superuser credentials, including the default admin password.

By understanding and properly managing superuser capabilities and default admin creation, you can ensure secure and effective administration of your R2R deployment.

## Security Considerations

When implementing user authentication, consider the following security best practices:

1. **Use HTTPS**: Always use HTTPS in production to encrypt data in transit.
2. **Implement rate limiting**: Protect against brute-force attacks by limiting login attempts.
3. **Use secure password hashing**: R2R uses bcrypt for password hashing by default, which is a secure choice.
4. **Implement multi-factor authentication (MFA)**: Consider adding MFA for an extra layer of security.
5. **Regular security audits**: Conduct regular security audits of your authentication system.

## Customizing Authentication

R2R's authentication system is flexible and can be customized to fit your specific needs:

1. **Custom user fields**: Extend the User model to include additional fields.
2. **OAuth integration**: Integrate with third-party OAuth providers for social login.
3. **Custom password policies**: Implement custom password strength requirements.
4. **User roles and permissions**: Implement a role-based access control system.

## Troubleshooting

Here are some common issues and their solutions:

1. **Login fails after registration**: Ensure email verification is completed if enabled.
2. **Token refresh fails**: Check if the refresh token has expired; the user may need to log in again.
3. **Unable to change password**: Verify that the current password is correct.

## Conclusion

R2R provides a comprehensive set of user authentication and management features, allowing developers to create secure and user-friendly applications. By leveraging these capabilities, you can implement robust user authentication, document management, and access control in your R2R-based projects.

For more advanced use cases or custom implementations, refer to the R2R documentation or reach out to the community for support.

---


## Introduction

GraphRAG is a powerful feature of R2R that allows you to perform graph-based search and retrieval. This guide will walk you through the process of setting it up and running your first queries.



<Frame caption="An example knowledge graph constructed from companies in the YC directory.">
  <img src="../images/yc_s24.png" />
</Frame>

<Note>
Note that graph construction may take long for local LLMs, we recommend using cloud LLMs for faster results.
</Note>


## Start server

We provide three configurations for R2R: Light, Light with Local LLMs, and Full with Docker+Hatchet. If you want to get started quickly, we recommend using R2R Light. If you want to run large graph workloads, we recommend using R2R Full with Docker+Hatchet.

<Tabs>
<Tab title="R2R Light">
```bash
r2r serve
```

<Accordion icon="gear" title="Configuration: r2r.toml">
``` toml
[app]
# app settings are global available like `r2r_config.agent.app`
# project_name = "r2r_default" # optional, can also set with `R2R_PROJECT_NAME` env var

[agent]
system_instruction_name = "rag_agent"
tool_names = ["search"]

[auth]
provider = "r2r"
access_token_lifetime_in_minutes = 60
refresh_token_lifetime_in_days = 7
require_authentication = false
require_email_verification = false
default_admin_email = "admin@example.com"
default_admin_password = "change_me_immediately"


[completion]
provider = "litellm"
concurrent_request_limit = 256

  [completion.generation_config]
  model = "openai/gpt-4o"
  temperature = 0.1
  top_p = 1
  max_tokens_to_sample = 1_024
  stream = false
  add_generation_kwargs = { }

[crypto]
provider = "bcrypt"

[database]
provider = "postgres"
default_collection_name = "Default"
default_collection_description = "Your default collection."
enable_fts = true # whether or not to enable full-text search, e.g `hybrid search`

[embedding]
provider = "litellm"
base_model = "openai/text-embedding-3-small"
base_dimension = 512
batch_size = 128
add_title_as_prefix = false
rerank_model = "None"
concurrent_request_limit = 256
quantization_settings = { quantization_type = "FP32" }

[file]
provider = "postgres"

[ingestion]
provider = "r2r"
chunking_strategy = "recursive"
chunk_size = 1_024
chunk_overlap = 512
excluded_parsers = ["mp4"]

  [ingestion.chunk_enrichment_settings]
    enable_chunk_enrichment = false # disabled by default
    strategies = ["semantic", "neighborhood"]
    forward_chunks = 3
    backward_chunks = 3
    semantic_neighbors = 10
    semantic_similarity_threshold = 0.7
    generation_config = { model = "openai/gpt-4o-mini" }

  [ingestion.extra_parsers]
    pdf = "zerox"

[database]
provider = "postgres"
batch_size = 256

  [database.kg_creation_settings]
    kg_entity_description_prompt = "graphrag_entity_description"
    kg_triples_extraction_prompt = "graphrag_triples_extraction_few_shot"
    entity_types = [] # if empty, all entities are extracted
    relation_types = [] # if empty, all relations are extracted
    fragment_merge_count = 4 # number of fragments to merge into a single extraction
    max_knowledge_triples = 100
    max_description_input_length = 65536
    generation_config = { model = "openai/gpt-4o-mini" } # and other params, model used for triplet extraction

  [database.kg_entity_deduplication_settings]
    kg_entity_deduplication_type = "by_name"
    kg_entity_deduplication_prompt = "graphrag_entity_deduplication"
    max_description_input_length = 65536
    generation_config = { model = "openai/gpt-4o-mini" } # and other params, model used for deduplication

  [database.kg_enrichment_settings]
    community_reports_prompt = "graphrag_community_reports"
    max_summary_input_length = 65536
    generation_config = { model = "openai/gpt-4o-mini" } # and other params, model used for node description and graph clustering
    leiden_params = {}

  [database.kg_search_settings]
    entities_level = "document" # set to collection if you've run deduplication
    map_system_prompt = "graphrag_map_system"
    reduce_system_prompt = "graphrag_reduce_system"
    generation_config = { model = "openai/gpt-4o-mini" }

[logging]
provider = "r2r"
log_table = "logs"
log_info_table = "log_info"

[orchestration]
provider = "simple"


[prompt]
provider = "r2r"
```
</Accordion>
</Tab>

<Tab title="R2R Light with Local LLMs">
```bash
r2r serve --config-name=local_llm
```

<Accordion icon="gear" title="Configuration: local_llm">
``` toml
[agent]
system_instruction_name = "rag_agent"
tool_names = ["search"]

  [agent.generation_config]
  model = "ollama/llama3.1"

[completion]
provider = "litellm"
concurrent_request_limit = 1

  [completion.generation_config]
  model = "ollama/llama3.1"
  temperature = 0.1
  top_p = 1
  max_tokens_to_sample = 1_024
  stream = false
  add_generation_kwargs = { }

[embedding]
provider = "ollama"
base_model = "mxbai-embed-large"
base_dimension = 1_024
batch_size = 128
add_title_as_prefix = true
concurrent_request_limit = 2

[orchestration]
provider = "simple"
```
</Accordion>
</Tab>

<Tab title="R2R Full with Docker+Hatchet">
```bash
r2r serve --full --docker --config-name=full
```
<Accordion icon="gear" title="Configuration: full.toml">
``` toml
[ingestion]
provider = "unstructured_local"
strategy = "auto"
chunking_strategy = "by_title"
new_after_n_chars = 512
max_characters = 1_024
combine_under_n_chars = 128
overlap = 256

    [ingestion.extra_parsers]
    pdf = "zerox"

[orchestration]
provider = "hatchet"
kg_creation_concurrency_lipmit = 32
ingestion_concurrency_limit = 128
kg_enrichment_concurrency_limit = 8
```
</Accordion>
</Tab>
</Tabs>


## Ingesting files

We begin the cookbook by ingesting the default sample file `aristotle.txt` used across R2R tutorials and cookbooks:

<Tabs>
<Tab title="CLI">
```bash
r2r ingest-sample-file
# or
r2r ingest-files /path/to/your/files_or_directory

# Example Response
[{'message': 'Ingestion task queued successfully.', 'task_id': '2b16bb55-4f47-4e66-a6bd-da9e215b9793', 'document_id': '9fbe403b-c11c-5aae-8ade-ef22980c3ad1'}]
```
</Tab>

<Tab title="SDK">
```python
from r2r import R2RClient

import requests
import tempfile
import os

# URL of the raw content on GitHub
url = "https://raw.githubusercontent.com/SciPhi-AI/R2R/main/py/core/examples/data/aristotle.txt"

# Fetch the content
response = requests.get(url)
content = response.text

# Create a temporary file
with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', prefix='aristotle_') as temp_file:
    # Write the content to the temporary file
    temp_file.write(content)
    temp_file_path = temp_file.name

client = R2RClient("http://localhost:7272")
client.ingest_files([temp_file_path])

# Example Response
[{'message': 'Ingestion task queued successfully.', 'task_id': '2b16bb55-4f47-4e66-a6bd-da9e215b9793', 'document_id': '9fbe403b-c11c-5aae-8ade-ef22980c3ad1'}]
```
</Tab>
</Tabs>

The initial ingestion step adds parses the given documents and inserts them into R2R's relational and vector databases, enabling document management and semantic search over them. The `aristotle.txt` example file is typically ingested in under 10s. You can confirm ingestion is complete by querying the documents overview table:

```bash
r2r documents-overview

# Example Response
{'id': '9fbe403b-c11c-5aae-8ade-ef22980c3ad1', 'title': 'aristotle.txt', 'user_id': '2acb499e-8428-543b-bd85-0d9098718220', 'type': 'txt', 'created_at': '2024-09-05T18:20:47.921933Z', 'updated_at': '2024-09-05T18:20:47.921938Z', 'ingestion_status': 'success', 'restructuring_status': 'pending', 'version': 'v0', 'collection_ids': [], 'metadata': {'version': 'v0'}}
```

When ingestion completes successfully for a given file we will find that `ingestion_status` reads `success` in the corresponding output. You can also view in R2R's dashboard on http://localhost:7273 that the file has been ingested.

![Ingested File](../images/kg_ingestion_status.png)

## Create Knowledge Graph

Knowledge graph creation is done in two steps:

1. `create-graph`: Extracts nodes and relationships from your input document collection.
2. `enrich-graph`: Enhances the graph structure through clustering and explaining entities (commonly referred to as `GraphRAG`).


<Tabs>
<Tab title="CLI">
```bash
# Cost Estimation step.
# collection ID is optional. If you don't specify one, the default collection will be used.
r2r create-graph --collection-id=122fdf6a-e116-546b-a8f6-e4cb2e2c0a09

This will run a cost estimation step to give you an estimate of the cost of the graph creation process.

# Example Response
Time taken: 0.21 seconds
{
  "results": {
    "message": "Ran Graph Creation Estimate (not the actual run). Note that these are estimated ranges, actual values may vary. To run the KG creation process, run `create-graph` with `--run` in the cli, or `run_type=\"run\"` in the client.",
    "document_count": 2,
    "number_of_jobs_created": 3,
    "total_chunks": 29,
    "estimated_entities": "290 - 580",
    "estimated_triples": "362 - 870",
    "estimated_llm_calls": "348 - 638",
    "estimated_total_in_out_tokens_in_millions": "0 - 1",
    "estimated_total_time_in_minutes": "Depends on your API key tier. Accurate estimate coming soon. Rough estimate: 0.0 - 0.17",
    "estimated_cost_in_usd": "0.0 - 0.06"
  }
}

# Then, you can run the graph creation process with:
r2r create-graph --collection-id=<optional> --run

# Example response for R2R Light
[{'message': 'Graph created successfully, please run enrich-graph to enrich the graph for GraphRAG.'}]

# Example Response for R2R Full. This call is non-blocking and returns immediately. We can check the status using the hatchet dashboard on http://localhost:7274. Details below:
[{'message': 'Graph creation task queued successfully.', 'task_id': 'd9dae1bb-5862-4a16-abaf-5297024df390'}]
```
</Tab>

<Tab title="SDK">
```python
from r2r import R2RClient

client = R2RClient("http://localhost:7272")
kg_search_settings = { 'run_mode': 'estimate' }
estimate = client.create_graph(collection_id=collection_id, kg_search_settings=kg_search_settings)
print(estimate)
# This will run a cost estimation step to give you an estimate of the cost of the graph creation process.

# Example Response
Time taken: 0.21 seconds
{
  "results": {
    "message": "These are estimated ranges, actual values may vary. To run the KG creation process, run `create-graph` with `--run` in the cli, or `run_mode=\"run\"` in the client.",
    "document_count": 2,
    "number_of_jobs_created": 3,
    "total_chunks": 29,
    "estimated_entities": "290 - 580",
    "estimated_triples": "362 - 870",
    "estimated_llm_calls": "348 - 638",
    "estimated_total_in_out_tokens_in_millions": "0 - 1",
    "estimated_total_time_in_minutes": "Depends on your API key tier. Accurate estimate coming soon. Rough estimate: 0.0 - 0.17",
    "estimated_cost_in_usd": "0.0 - 0.06"
  }
}

# Then, you can run the graph creation process with:
kg_search_settings = { 'run_mode': 'run' }
client.create_graph(collection_id=collection_id, kg_search_settings=kg_search_settings)

# Example response for R2R Light
[{'message': 'Graph created successfully, please run enrich-graph to enrich the graph for GraphRAG.'}]

# Example Response for R2R Full. This call is non-blocking and returns immediately. We can check the status using the hatchet dashboard on http://localhost:7274. Details below:
[{'message': 'Graph creation task queued successfully.', 'task_id': 'd9dae1bb-5862-4a16-abaf-5297024df390'}]


```
</Tab>

</Tabs>

If you are using R2R Full, you can log into the hatchet dashboard on http://localhost:7274 (admin@example.com / Admin123!!) to check the status of the graph creation process. Please make sure all the `kg-extract-*` tasks are completed before running the enrich-graph step.

![Hatchet Dashboard](../images/kg_extraction_progress.png)


This step will create a knowledge graph with nodes and relationships. You can get the entities and relationships in the graph using our dashboard on http://localhost:7273 or by calling the following API endpoints. These hit the /v2/entities and /v2/triples endpoints respectively. This will by default use the `entity_level=document` query parameter to get the entities and triples at the document level. We will set the default collection id to `122fdf6a-e116-546b-a8f6-e4cb2e2c0a09` when submitting requests to the endpoints below.

- Entities: [Entities](/api-reference/endpoint/entities)
- Triples: [Triples](/api-reference/endpoint/entities)

## Graph Enrichment

Now we have a searchable graph, but this graph is not enriched yet. It does not have any community level information. We will now run the enrichment step.

The graph enrichment step performs hierarchical leiden clustering to create communities, and embeds the descriptions. These embeddings will be used later in the local search stage of the pipeline. If you are more interested in the algorithm, please refer to the blog post [here](https://www.sciphi.ai/blog/graphrag).

<Tabs>
<Tab title="CLI">
```bash
# collection ID is optional. If you don't specify one, the default collection will be used.
r2r enrich-graph --collection-id=122fdf6a-e116-546b-a8f6-e4cb2e2c0a09

# Similar to the graph creation step, this will run a cost estimation step to give you an estimate of the cost of the graph enrichment process.
Time taken: 0.22 seconds
{
  "results": {
    "total_entities": 269,
    "total_triples": 345,
    "estimated_llm_calls": "26 - 53",
    "estimated_total_in_out_tokens_in_millions": "0.05 - 0.11",
    "estimated_total_time_in_minutes": "Depends on your API key tier. Accurate estimate coming soon. Rough estimate: 0.01 - 0.02",
    "estimated_cost_in_usd": "0.0 - 0.01"
  }
}

# Now, you can run the graph enrichment process with:
r2r enrich-graph --collection-id=122fdf6a-e116-546b-a8f6-e4cb2e2c0a09 --run

# Example Response with R2R Light
[{'message': 'Graph enriched successfully.'}]

# Example Response with R2R Full, you can check the status using the hatchet dashboard on http://localhost:7274.
[{'message': 'Graph enrichment task queued successfully.', 'task_id': 'd9dae1bb-5862-4a16-abaf-5297024df390'}]
```
</Tab>

<Tab title="SDK">
```python
from r2r import R2RClient

client = R2RClient("http://localhost:7272")
kg_search_settings = { 'run_mode': 'estimate' }
collection_id = "122fdf6a-e116-546b-a8f6-e4cb2e2c0a09" # optional, if not specified, the default collection will be used.
estimate = client.enrich_graph(collection_id=collection_id, kg_search_settings=kg_search_settings)
print(estimate)

# This will run a cost estimation step to give you an estimate of the cost of the graph enrichment process.

# Example Response
Time taken: 0.22 seconds
{
  "results": {
    "total_entities": 269,
    "total_triples": 345,
    "estimated_llm_calls": "26 - 53",
    "estimated_total_in_out_tokens_in_millions": "0.05 - 0.11",
    "estimated_total_time_in_minutes": "Depends on your API key tier. Accurate estimate coming soon. Rough estimate: 0.01 - 0.02",
    "estimated_cost_in_usd": "0.0 - 0.01"
  }
}

# Now, you can run the graph enrichment process with:
kg_search_settings = { 'run_mode': 'run' }
client.enrich_graph(collection_id=collection_id, kg_search_settings=kg_search_settings)

# Example Response with R2R Light
[{'message': 'Graph enriched successfully.'}]

# Example Response with R2R Full, you can check the status using the hatchet dashboard on http://localhost:7274.
[{'message': 'Graph enrichment task queued successfully.', 'task_id': 'd9dae1bb-5862-4a16-abaf-5297024df390'}]
```
</Tab>
</Tabs>

If you're using R2R Full, you can similarly check that all `community-summary-*` tasks are completed before proceeding.


Now you can see that the graph is enriched with the following information. We have added descriptions and embeddings to the nodes and relationships. Also, each node is mapped to a community. Following is a visualization of the enriched graph (deprecated as of now. We are working on a new visualization tool):

You can see the list of communities in the graph using the following API endpoint:

- Communities: [Communities](/api-reference/endpoint/communities)

## Search

A knowledge graph search performs similarity search on the entity and community description embeddings.

```bash

r2r search --query="Who is Aristotle?" --use-kg-search

# The answer will be returned in JSON format and contains results from entities, relationships and communities. Following is a snippet of the output:

Vector search results:
[
  {
    'fragment_id': 'ecc754cd-380d-585f-84ac-021542ef3c1d',
    'extraction_id': '92d78034-8447-5046-bf4d-e019932fbc20',
    'document_id': '9fbe403b-c11c-5aae-8ade-ef22980c3ad1',
    'user_id': '2acb499e-8428-543b-bd85-0d9098718220',
    'collection_ids': [],
    'score': 0.7393344796100582,
    'text': 'Aristotle[A] (Greek: Ἀριστοτέλης Aristotélēs, pronounced [aristotélɛːs]; 384–322 BC) was an Ancient Greek philosopher and polymath. His writings cover a broad range of subjects spanning the natural sciences, philosophy, linguistics, economics, politics, psychology, and the arts. As the founder of the Peripatetic school of philosophy in the Lyceum in Athens, he began the wider Aristotelian tradition that followed, which set the groundwork for the development of modern science.\n\nLittle is known about Aristotle's life. He was born in the city of Stagira in northern Greece during the Classical period. His father, Nicomachus, died when Aristotle was a child, and he was brought up by a guardian. At 17 or 18, he joined Plato's Academy in Athens and remained there until the age of 37 (c.\u2009347 BC). Shortly after Plato died, Aristotle left Athens and, at the request of Philip II of Macedon, tutored his son Alexander the Great beginning in 343 BC. He established a library in the Lyceum, which helped him to produce many of his hundreds of books on papyrus scrolls.\n\nThough Aristotle wrote many elegant treatises and dia ...",
    'metadata': {'title': 'aristotle.txt', 'version': 'v0', 'file_name': 'tmpm3ceiqs__aristotle.txt', 'chunk_order': 0, 'document_type': 'txt', 'size_in_bytes': 73353, 'unstructured_filetype': 'text/plain', 'unstructured_languages': ['eng'], 'partitioned_by_unstructured': True, 'associatedQuery': 'Who is Aristotle?'}}
  }, ...
]

KG search results:
{
  'local_result': {
    'query': 'Who is Aristotle?',
    'entities': {'0': {'name': 'Aristotle', 'description': 'Aristotle was an ancient Greek philosopher and polymath, recognized as the father of various fields including logic, biology, and political science. He authored significant works such as the *Nicomachean Ethics* and *Politics*, where he explored concepts of virtue, governance, and the nature of reality, while also critiquing Platos ideas. His teachings and observations laid the groundwork for numerous disciplines, influencing thinkers ...'}},
    'relationships': {},
    'communities': {'0': {'summary': '```json\n{\n    "title": "Aristotle and His Contributions",\n    "summary": "The community revolves around Aristotle, an ancient Greek philosopher and polymath, who made significant contributions to various fields including logic, biology, political science, and economics. His works, such as 'Politics' and 'Nicomachean Ethics', have influenced numerous disciplines and thinkers from antiquity through the Middle Ages and beyond. The relationships between his various works and the fields he contributed to highlight his profound impact on Western thought.",\n    "rating": 9.5,\n    "rating_explanation": "The impact severity rating is high due to Aristotle's foundational influence on multiple disciplines and his enduring legacy in Western philosophy and science.",\n    "findings": [\n        {\n            "summary": "Aristotle's Foundational Role in Logic",\n            "explanation": "Aristotle is credited with the earliest study of formal logic, and his conception of it was the dominant form of Western logic until the 19th-century advances in mathematical logic. His works compiled into a set of six bo ...}}}}
  },
  'global_result': None
}
Time taken: 2.39 seconds
```

# Conclusion

In conclusion, integrating R2R with GraphRAG significantly enhances the capabilities of your RAG applications. By leveraging the power of graph-based knowledge representations, GraphRAG allows for more nuanced and context-aware information retrieval. This is evident in the example query we ran using R2R, which not only retrieved relevant information but also provided a structured analysis of the key contributions of Aristotle to modern society.

In essence, combining R2R with GraphRAG empowers your RAG applications to deliver more intelligent, context-aware, and insightful responses, making it a powerful tool for advanced information retrieval and analysis tasks.

Feel free to reach out to us at founders@sciphi.ai if you have any questions or need further assistance.


# Advanced GraphRAG Techniques

If you want to learn more about the advanced techniques that we use in GraphRAG, please refer to the [Advanced GraphRAG Techniques](/cookbooks/advanced-graphrag) page.

---


1. [**Deploy with Azure**](/documentation/deployment/azure)
2. [**Deploy with GCP**](/documentation/deployment/gcp)
3. [**Deploy with AWS**](/documentation/deployment/aws)

---


# Understanding Chunk Enrichment in RAG Systems

In modern Retrieval-Augmented Generation (RAG) systems, documents are systematically broken down into smaller, manageable pieces called chunks. While chunking is essential for efficient vector search operations, these individual chunks sometimes lack the broader context needed for comprehensive question answering or analysis tasks.

## The Challenge of Context Loss

Let's examine a real-world example using Lyft's 2021 annual report (Form 10-K) from their [public filing](https://github.com/SciPhi-AI/R2R/blob/main/py/core/examples/data/lyft_2021.pdf).

During ingestion, this 200+ page document is broken into 1,223 distinct chunks. Consider this isolated chunk:

```plaintext
storing unrented and returned vehicles. These impacts to the demand for and operations of the different rental programs have and may continue to adversely affect our business, financial condition and results of operation.
```

Reading this chunk in isolation raises several questions:
- What specific impacts are being discussed?
- Which rental programs are affected?
- What's the broader context of these business challenges?

This is where contextual enrichment becomes invaluable.

## Introducing Contextual Enrichment

Contextual enrichment is an advanced technique that enhances chunks with relevant information from surrounding or semantically related content. Think of it as giving each chunk its own "memory" of related information.

### Enabling Enrichment

To activate this feature, configure your `r2r.toml` file with the following settings:

```toml
[ingestion.chunk_enrichment_settings]
    enable_chunk_enrichment = true # disabled by default
    strategies = ["semantic", "neighborhood"]
    forward_chunks = 3            # Look ahead 3 chunks
    backward_chunks = 3           # Look behind 3 chunks
    semantic_neighbors = 10       # Find 10 semantically similar chunks
    semantic_similarity_threshold = 0.7  # Minimum similarity score
    generation_config = { model = "openai/gpt-4o-mini" }
```

## Enrichment Strategies Explained

R2R implements two sophisticated strategies for chunk enrichment:

### 1. Neighborhood Strategy
This approach looks at the document's natural flow by examining chunks that come before and after the target chunk:
- **Forward Looking**: Captures upcoming context (configurable, default: 3 chunks)
- **Backward Looking**: Incorporates previous context (configurable, default: 3 chunks)
- **Use Case**: Particularly effective for narrative documents where context flows linearly

### 2. Semantic Strategy
This method uses advanced embedding similarity to find related content throughout the document:
- **Vector Similarity**: Identifies chunks with similar meaning regardless of location
- **Configurable Neighbors**: Customizable number of similar chunks to consider
- **Similarity Threshold**: Set minimum similarity scores to ensure relevance
- **Use Case**: Excellent for documents with themes repeated across different sections

## The Enrichment Process

When enriching chunks, R2R uses a carefully crafted prompt to guide the LLM:

```plaintext
## Task:

Enrich and refine the given chunk of text using information from the provided context chunks. The goal is to make the chunk more precise and self-contained.

## Context Chunks:
{context_chunks}

## Chunk to Enrich:
{chunk}

## Instructions:
1. Rewrite the chunk in third person.
2. Replace all common nouns with appropriate proper nouns.
3. Use information from the context chunks to enhance clarity.
4. Ensure the enriched chunk remains independent and self-contained.
5. Maintain original scope without bleeding information.
6. Focus on precision and informativeness.
7. Preserve original meaning while improving clarity.
8. Output only the enriched chunk.

## Enriched Chunk:
```

## Implementation and Results

To process your documents with enrichment:

```bash
r2r ingest-files --file_paths path/to/lyft_2021.pdf
```

### Viewing Enriched Results

Access your enriched chunks through the API:
```
http://localhost:7272/v2/document_chunks/{document_id}
```

Let's compare the before and after of our example chunk:

**Before Enrichment:**
```plaintext
storing unrented and returned vehicles. These impacts to the demand for and operations of the different rental programs have and may continue to adversely affect our business, financial condition and results of operation.
```

**After Enrichment:**
```plaintext
The impacts of the COVID-19 pandemic on the demand for and operations of the various vehicle rental programs, including Lyft Rentals and the Express Drive program, have resulted in challenges regarding the storage of unrented and returned vehicles. These adverse conditions are anticipated to continue affecting Lyft's overall business performance, financial condition, and operational results.
```

Notice how the enriched version:
- Specifies the cause (COVID-19 pandemic)
- Names specific programs (Lyft Rentals, Express Drive)
- Provides clearer context about the business impact
- Maintains professional, third-person tone

## Metadata and Storage

The system maintains both enriched and original versions:

```json
{
    "results": [
        {
            "text": "enriched_version",
            "metadata": {
                "original_text": "original_version",
                "chunk_enrichment_status": "success",
                // ... additional metadata ...
            }
        }
    ]
}
```

This dual storage ensures transparency and allows for version comparison when needed.

## Best Practices

1. **Tune Your Parameters**: Adjust `forward_chunks`, `backward_chunks`, and `semantic_neighbors` based on your document structure
2. **Monitor Enrichment Quality**: Regularly review enriched chunks to ensure they maintain accuracy
3. **Consider Document Type**: Different documents may benefit from different enrichment strategies
4. **Balance Context Size**: More context isn't always better - find the sweet spot for your use case