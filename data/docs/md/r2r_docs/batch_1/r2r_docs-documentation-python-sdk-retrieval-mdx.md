
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