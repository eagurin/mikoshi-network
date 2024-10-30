

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