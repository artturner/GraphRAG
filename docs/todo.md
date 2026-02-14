# Grounded GraphRAG Tutor - Implementation Checklist

Use this checklist to track progress through the implementation. Each item corresponds to a prompt in `docs/GRAPH_RAG_TUTOR_PROMPTS.md`.

---

## CHUNK 1: Foundation & Project Structure

### Step 1.1: Project Skeleton
- [ ] Create `src/` directory structure
  - [ ] `src/connectors/` with `__init__.py`
  - [ ] `src/ingestion/` with `__init__.py`
  - [ ] `src/embeddings/` with `__init__.py`
  - [ ] `src/store/` with `__init__.py`
  - [ ] `src/retrieval/` with `__init__.py`
  - [ ] `src/graphs/` with `__init__.py`
  - [ ] `src/app/` with `__init__.py`
  - [ ] `src/eval/` with `__init__.py`
- [ ] Create `tests/` directory structure (mirrors src)
- [ ] Create `configs/` directory
- [ ] Create `scripts/` directory
- [ ] Create `pyproject.toml` with all dependencies
- [ ] Create `.env.example`
- [ ] Create `.gitignore` (Python-specific)
- [ ] Create `README.md`
- [ ] Write tests verifying project structure exists
- [ ] Run: `pytest tests/ -v` - all tests pass

### Step 1.2: Configuration System
- [ ] Create `src/config.py`
  - [ ] Settings class using pydantic-settings
  - [ ] CorpusConfig nested model
  - [ ] VectorStoreConfig nested model
  - [ ] EmbeddingsConfig nested model
  - [ ] LLMConfig nested model
  - [ ] GraphConfig nested model
- [ ] Create `configs/default.yaml` with default settings
- [ ] Update `.env.example` with all environment variables
- [ ] Create `tests/test_config.py`
  - [ ] Test loading from YAML
  - [ ] Test environment variable override
  - [ ] Test validation errors
  - [ ] Test default values
- [ ] Run: `pytest tests/test_config.py -v` - all tests pass

### Step 1.3: Logging Infrastructure
- [ ] Create `src/logging_config.py`
  - [ ] Configure structured JSON logging
  - [ ] Support LOG_LEVEL from environment
  - [ ] Add correlation ID support
  - [ ] Create `get_logger(name)` function
- [ ] Update `src/__init__.py` to initialize logging
- [ ] Create `tests/test_logging.py`
  - [ ] Test logger creation
  - [ ] Test log level configuration
  - [ ] Test JSON format output
  - [ ] Test correlation ID propagation
- [ ] Run: `pytest tests/test_logging.py -v` - all tests pass

### Step 1.4: Type Definitions
- [ ] Create `src/types.py`
  - [ ] Document model (id, content, source, metadata, created_at)
  - [ ] Chunk model (id, document_id, content, start_idx, end_idx, metadata)
  - [ ] Citation model (source, chunk_id, text, score)
  - [ ] Answer model (text, citations, confidence, refusal_reason)
  - [ ] QueryResult model (question, answer, mode, latency_ms)
  - [ ] IngestResult model (documents_count, chunks_count, errors)
  - [ ] EmbeddingVector type alias
  - [ ] ProviderConfig base model
- [ ] Create `src/exceptions.py`
  - [ ] RAGError base exception
  - [ ] ConnectorError
  - [ ] IngestionError
  - [ ] EmbeddingError
  - [ ] VectorStoreError
  - [ ] RetrievalError
  - [ ] LLMError
  - [ ] ConfigurationError
- [ ] Create `tests/test_types.py`
  - [ ] Test Document creation and validation
  - [ ] Test Chunk creation with metadata
  - [ ] Test Citation with score validation
  - [ ] Test Answer with optional refusal
  - [ ] Test QueryResult serialization
  - [ ] Test exception hierarchy
- [ ] Run: `pytest tests/test_types.py -v` - all tests pass

---

## CHUNK 2: Document Connectors

### Step 2.1: Base Connector Interface
- [ ] Create `src/connectors/base.py`
  - [ ] Abstract `BaseConnector` class
  - [ ] Abstract `load() -> list[Document]` method
  - [ ] Abstract `list_documents() -> list[str]` method
  - [ ] Concrete `validate_source() -> bool` method
- [ ] Update `src/connectors/__init__.py`
  - [ ] Export `BaseConnector`
  - [ ] Create `ConnectorRegistry`
- [ ] Create `tests/connectors/test_base.py`
  - [ ] Test BaseConnector cannot be instantiated
  - [ ] Test subclass must implement all methods
  - [ ] Test ConnectorRegistry registration
- [ ] Run: `pytest tests/connectors/test_base.py -v` - all tests pass

### Step 2.2: Document Model Enhancement
- [ ] Update `src/types.py`
  - [ ] Add `DocumentType` enum (PDF, TXT, MD, HTML, JSON)
  - [ ] Add `DocumentMetadata` model
  - [ ] Update Document to include document_type and full_metadata
- [ ] Create `src/connectors/document.py`
  - [ ] `create_document_id()` function
  - [ ] `detect_document_type()` function
  - [ ] `extract_metadata()` function
- [ ] Create `tests/connectors/test_document.py`
  - [ ] Test document ID generation is deterministic
  - [ ] Test document type detection
  - [ ] Test metadata extraction
- [ ] Run: `pytest tests/connectors/test_document.py -v` - all tests pass

### Step 2.3: Local File Connector
- [ ] Create `src/connectors/local.py`
  - [ ] `LocalConnector` class extending `BaseConnector`
  - [ ] Constructor with `source_path` parameter
  - [ ] Support file types: .pdf, .txt, .md, .html
  - [ ] Implement `load()` method
  - [ ] Implement `list_documents()` method
  - [ ] Implement `validate_source()` method
  - [ ] Handle encoding detection
- [ ] Create test fixtures in `tests/fixtures/documents/`
  - [ ] `sample.txt`
  - [ ] `sample.md`
  - [ ] `sample.html`
- [ ] Create `tests/connectors/test_local.py`
  - [ ] Test loading PDF files
  - [ ] Test loading text files
  - [ ] Test loading markdown files
  - [ ] Test loading HTML files
  - [ ] Test handling unsupported file types
  - [ ] Test list_documents returns correct paths
  - [ ] Test validate_source with missing directory
- [ ] Run: `pytest tests/connectors/test_local.py -v` - all tests pass

### Step 2.4: Connector Factory
- [ ] Create `src/connectors/factory.py`
  - [ ] `ConnectorFactory` class
  - [ ] `get_connector(config: CorpusConfig) -> BaseConnector` method
  - [ ] Support connector types: "local", "s3", "web"
  - [ ] Raise ConfigurationError for unknown types
- [ ] Create `src/connectors/s3.py` (stub)
  - [ ] `S3Connector` stub class with `NotImplementedError`
- [ ] Create `src/connectors/web.py` (stub)
  - [ ] `WebConnector` stub class with `NotImplementedError`
- [ ] Update `src/connectors/__init__.py`
  - [ ] Export `ConnectorFactory`
  - [ ] Export all connector classes
- [ ] Create `tests/connectors/test_factory.py`
  - [ ] Test factory returns correct connector type
  - [ ] Test factory raises error for unknown type
  - [ ] Test factory with local connector config
  - [ ] Test stub connectors raise NotImplementedError
- [ ] Run: `pytest tests/connectors/ -v` - all tests pass

---

## CHUNK 3: Ingestion Pipeline

### Step 3.1: Text Cleaning
- [ ] Create `src/ingestion/cleaning.py`
  - [ ] `TextCleaner` class
  - [ ] `normalize_whitespace(text: str) -> str` method
  - [ ] `remove_html_tags(text: str) -> str` method
  - [ ] `normalize_unicode(text: str) -> str` method
  - [ ] `remove_boilerplate(text: str) -> str` method
  - [ ] `clean(text: str) -> str` method (applies all)
  - [ ] Configurable cleaning options
- [ ] Create `tests/ingestion/test_cleaning.py`
  - [ ] Test whitespace normalization
  - [ ] Test HTML tag removal
  - [ ] Test unicode normalization
  - [ ] Test boilerplate removal
  - [ ] Test full cleaning pipeline
  - [ ] Test that cleaning preserves important content
- [ ] Run: `pytest tests/ingestion/test_cleaning.py -v` - all tests pass

### Step 3.2: Chunking Strategies
- [ ] Create `src/ingestion/chunking.py`
  - [ ] `BaseChunker` abstract class
  - [ ] `FixedSizeChunker` class
    - [ ] Chunks by character count
    - [ ] Supports overlap
    - [ ] Respects word boundaries
  - [ ] `SentenceChunker` class
    - [ ] Chunks by sentences
    - [ ] Supports min/max chunk size
    - [ ] Groups sentences to meet size constraints
  - [ ] `ChunkMetadata` model for tracking position
- [ ] Create `tests/ingestion/test_chunking.py`
  - [ ] Test FixedSizeChunker with various sizes
  - [ ] Test overlap functionality
  - [ ] Test word boundary respect
  - [ ] Test SentenceChunker sentence detection
  - [ ] Test chunk size constraints
  - [ ] Test metadata is correctly attached
  - [ ] Test edge cases (empty text, very long words)
- [ ] Run: `pytest tests/ingestion/test_chunking.py -v` - all tests pass

### Step 3.3: Metadata Extraction
- [ ] Create `src/ingestion/metadata.py`
  - [ ] `ChunkMetadataExtractor` class
  - [ ] `extract(chunk: Chunk, document: Document) -> dict` method
  - [ ] `extract_position_info(chunk: Chunk) -> dict` function
  - [ ] `extract_context(chunk: Chunk, document: Document) -> dict` function
  - [ ] `generate_chunk_id(document_id: str, chunk_index: int) -> str` function
- [ ] Create `tests/ingestion/test_metadata.py`
  - [ ] Test position info extraction
  - [ ] Test context extraction
  - [ ] Test chunk ID generation (deterministic)
  - [ ] Test metadata is serializable
- [ ] Run: `pytest tests/ingestion/test_metadata.py -v` - all tests pass

### Step 3.4: Ingestion Orchestrator
- [ ] Create `src/ingestion/pipeline.py`
  - [ ] `IngestionPipeline` class
  - [ ] `__init__(connector, cleaner, chunker)` constructor
  - [ ] `run() -> IngestResult` method
  - [ ] `run_with_progress() -> Iterator[IngestProgress]` method
  - [ ] `IngestProgress` model
- [ ] Update `src/ingestion/__init__.py`
  - [ ] Export `IngestionPipeline`
  - [ ] Export `TextCleaner`
  - [ ] Export `FixedSizeChunker`
  - [ ] Export `SentenceChunker`
- [ ] Create `tests/ingestion/test_pipeline.py`
  - [ ] Test full pipeline with local connector
  - [ ] Test documents are cleaned before chunking
  - [ ] Test chunks have correct metadata
  - [ ] Test error handling for corrupted files
  - [ ] Test progress reporting
- [ ] Create `scripts/ingest.py`
  - [ ] CLI script to run ingestion
  - [ ] Arguments: --corpus, --config
  - [ ] Progress output and summary
- [ ] Run: `pytest tests/ingestion/ -v` - all tests pass

---

## CHUNK 4: Embedding Providers

### Step 4.1: Base Embedding Interface
- [ ] Create `src/embeddings/base.py`
  - [ ] `BaseEmbeddings` abstract class
  - [ ] `embed_documents(texts: list[str]) -> list[EmbeddingVector]` method
  - [ ] `embed_query(text: str) -> EmbeddingVector` method
  - [ ] `dimension` property
  - [ ] `EmbeddingCache` protocol
- [ ] Update `src/embeddings/__init__.py`
  - [ ] Export `BaseEmbeddings`
- [ ] Create `tests/embeddings/test_base.py`
  - [ ] Test BaseEmbeddings cannot be instantiated
  - [ ] Test subclass must implement all methods
  - [ ] Test dimension property is required
- [ ] Run: `pytest tests/embeddings/test_base.py -v` - all tests pass

### Step 4.2: Local Sentence Transformers
- [ ] Create `src/embeddings/local.py`
  - [ ] `LocalEmbeddings` class extending `BaseEmbeddings`
  - [ ] Uses sentence-transformers library
  - [ ] Default model: "all-MiniLM-L6-v2"
  - [ ] Configurable model name
  - [ ] Support batch embedding
  - [ ] GPU detection and usage
- [ ] Create `tests/embeddings/test_local.py`
  - [ ] Test embedding single text
  - [ ] Test embedding multiple texts
  - [ ] Test dimension matches model
  - [ ] Test embeddings are normalized
  - [ ] Test with different model names (marked slow)
- [ ] Run: `pytest tests/embeddings/test_local.py -v` - all tests pass

### Step 4.3: Bedrock Titan Embeddings
- [ ] Create `src/embeddings/bedrock.py`
  - [ ] `BedrockEmbeddings` class extending `BaseEmbeddings`
  - [ ] Uses boto3 Bedrock client
  - [ ] Model: amazon.titan-embed-text-v1
  - [ ] Handle AWS credentials from environment
  - [ ] Implement retry logic with exponential backoff
  - [ ] Handle rate limiting
- [ ] Create `tests/embeddings/test_bedrock.py`
  - [ ] Mock boto3 client for unit tests
  - [ ] Test embedding single text
  - [ ] Test embedding multiple texts
  - [ ] Test retry logic
  - [ ] Test error handling
  - [ ] Skip integration tests if no AWS credentials
- [ ] Run: `pytest tests/embeddings/test_bedrock.py -v` - all tests pass

### Step 4.4: OpenAI Embeddings
- [ ] Create `src/embeddings/openai_emb.py`
  - [ ] `OpenAIEmbeddings` class extending `BaseEmbeddings`
  - [ ] Uses openai library
  - [ ] Model: text-embedding-3-small (default)
  - [ ] Handle API key from environment
  - [ ] Implement retry logic
  - [ ] Handle rate limiting
- [ ] Create `tests/embeddings/test_openai.py`
  - [ ] Mock OpenAI client for unit tests
  - [ ] Test embedding single text
  - [ ] Test embedding multiple texts
  - [ ] Test retry logic
  - [ ] Test error handling
  - [ ] Skip integration tests if no OPENAI_API_KEY
- [ ] Run: `pytest tests/embeddings/test_openai.py -v` - all tests pass

### Step 4.5: Embedding Factory and Cache
- [ ] Create `src/embeddings/factory.py`
  - [ ] `EmbeddingsFactory` class
  - [ ] `get_embeddings(config: EmbeddingsConfig) -> BaseEmbeddings` method
  - [ ] Support providers: "local_st", "bedrock_titan", "openai"
- [ ] Create `src/embeddings/cache.py`
  - [ ] `DiskEmbeddingCache` class
  - [ ] Cache key: hash of text + model name
  - [ ] `get(text: str, model: str) -> EmbeddingVector | None` method
  - [ ] `set(text: str, model: str, vector: EmbeddingVector)` method
  - [ ] Clear cache functionality
- [ ] Create `src/embeddings/cached.py`
  - [ ] `CachedEmbeddings` wrapper class
  - [ ] Wraps any BaseEmbeddings with cache
- [ ] Update `src/embeddings/__init__.py`
  - [ ] Export all embedding classes and factory
- [ ] Create `tests/embeddings/test_factory.py`
  - [ ] Test factory returns correct type
  - [ ] Test factory with different configs
- [ ] Create `tests/embeddings/test_cache.py`
  - [ ] Test cache hit and miss
  - [ ] Test cache persistence
  - [ ] Test cache key generation
- [ ] Run: `pytest tests/embeddings/ -v` - all tests pass

---

## CHUNK 5: Vector Store

### Step 5.1: Base Vector Store Interface
- [ ] Create `src/store/base.py`
  - [ ] `BaseVectorStore` abstract class
  - [ ] `add_embeddings(chunks, embeddings) -> int` method
  - [ ] `similarity_search(query_embedding, k) -> list[SearchResult]` method
  - [ ] `delete(chunk_ids) -> int` method
  - [ ] `count() -> int` method
  - [ ] `clear() -> None` method
  - [ ] `SearchResult` model (chunk_id, score, chunk, metadata)
- [ ] Update `src/store/__init__.py`
  - [ ] Export `BaseVectorStore`
  - [ ] Export `SearchResult`
- [ ] Create `tests/store/test_base.py`
  - [ ] Test BaseVectorStore cannot be instantiated
  - [ ] Test subclass must implement all methods
- [ ] Run: `pytest tests/store/test_base.py -v` - all tests pass

### Step 5.2: FAISS Implementation
- [ ] Create `src/store/faiss_store.py`
  - [ ] `FAISSVectorStore` class extending `BaseVectorStore`
  - [ ] Use `faiss.IndexFlatIP` for inner product
  - [ ] Store metadata separately
  - [ ] Persist index to disk
  - [ ] Load index from disk
  - [ ] Support incremental additions
- [ ] Create `tests/store/test_faiss.py`
  - [ ] Test adding embeddings
  - [ ] Test similarity search returns correct results
  - [ ] Test search with different k values
  - [ ] Test persistence (save and load)
  - [ ] Test count and clear
  - [ ] Test empty store behavior
- [ ] Run: `pytest tests/store/test_faiss.py -v` - all tests pass

### Step 5.3: Chroma Implementation
- [ ] Create `src/store/chroma_store.py`
  - [ ] `ChromaVectorStore` class extending `BaseVectorStore`
  - [ ] Use chromadb library
  - [ ] Support persistent client
  - [ ] Support metadata filtering
  - [ ] Collection management
- [ ] Create `tests/store/test_chroma.py`
  - [ ] Test adding embeddings
  - [ ] Test similarity search
  - [ ] Test metadata filtering
  - [ ] Test persistence
  - [ ] Test count and clear
- [ ] Run: `pytest tests/store/test_chroma.py -v` - all tests pass

### Step 5.4: Vector Store Factory
- [ ] Create `src/store/factory.py`
  - [ ] `VectorStoreFactory` class
  - [ ] `get_store(config: VectorStoreConfig, dimension: int) -> BaseVectorStore` method
  - [ ] Support types: "faiss", "chroma"
- [ ] Update `src/store/__init__.py`
  - [ ] Export all store classes and factory
- [ ] Create `tests/store/test_factory.py`
  - [ ] Test factory returns correct type
  - [ ] Test factory with different configs
  - [ ] Test dimension is passed correctly
- [ ] Run: `pytest tests/store/ -v` - all tests pass

---

## CHUNK 6: Retrieval Service

### Step 6.1: Similarity Search Service
- [ ] Create `src/retrieval/service.py`
  - [ ] `RetrievalService` class
  - [ ] `__init__(embeddings: BaseEmbeddings, store: BaseVectorStore)` constructor
  - [ ] `index_documents(chunks: list[Chunk]) -> int` method
  - [ ] `search(query: str, k: int = 5) -> list[SearchResult]` method
  - [ ] `search_with_threshold(query: str, k: int, min_score: float) -> list[SearchResult]` method
- [ ] Update `src/retrieval/__init__.py`
  - [ ] Export `RetrievalService`
- [ ] Create `tests/retrieval/test_service.py`
  - [ ] Test document indexing
  - [ ] Test search returns results
  - [ ] Test search with threshold filters low scores
  - [ ] Test empty query handling
  - [ ] Test with real embeddings (marked slow)
- [ ] Run: `pytest tests/retrieval/test_service.py -v` - all tests pass

### Step 6.2: Citation Extraction
- [ ] Create `src/retrieval/citations.py`
  - [ ] `CitationBuilder` class
  - [ ] `build_citations(results: list[SearchResult]) -> list[Citation]` method
  - [ ] `format_citation(citation: Citation, style: str) -> str` method
  - [ ] Support styles: default, mla, apa
- [ ] Create `tests/retrieval/test_citations.py`
  - [ ] Test citation building from search results
  - [ ] Test citation formatting
  - [ ] Test different citation styles
  - [ ] Test citation with missing metadata
- [ ] Run: `pytest tests/retrieval/test_citations.py -v` - all tests pass

### Step 6.3: Reranking Support
- [ ] Create `src/retrieval/reranker.py`
  - [ ] `BaseReranker` abstract class
  - [ ] `rerank(query: str, results: list[SearchResult]) -> list[SearchResult]` method
  - [ ] `CrossEncoderReranker` class
  - [ ] `IdentityReranker` class (no-op)
- [ ] Create `tests/retrieval/test_reranker.py`
  - [ ] Test identity reranker returns same order
  - [ ] Test cross-encoder reranker changes order
  - [ ] Test reranking with empty results
- [ ] Run: `pytest tests/retrieval/test_reranker.py -v` - all tests pass

### Step 6.4: Retrieval Integration
- [ ] Update `src/retrieval/service.py`
  - [ ] Add reranker parameter to constructor
  - [ ] Add build_citations parameter to search
  - [ ] Return citations with search results
- [ ] Update `src/retrieval/__init__.py`
  - [ ] Export all retrieval classes
- [ ] Create `tests/retrieval/test_integration.py`
  - [ ] Test full retrieval pipeline
  - [ ] Test with reranking enabled
  - [ ] Test citation generation
- [ ] Run: `pytest tests/retrieval/ -v` - all tests pass

---

## CHUNK 7: LangGraph Q&A Workflow

### Step 7.1: Graph State Definition
- [ ] Create `src/graphs/state.py`
  - [ ] `GraphState` TypedDict with all fields
  - [ ] `StateBuilder` helper class
  - [ ] `StateValidator` for validating transitions
- [ ] Update `src/graphs/__init__.py`
  - [ ] Export `GraphState`
- [ ] Create `tests/graphs/test_state.py`
  - [ ] Test state creation
  - [ ] Test state validation
  - [ ] Test state immutability
- [ ] Run: `pytest tests/graphs/test_state.py -v` - all tests pass

### Step 7.2: Route Node
- [ ] Create `src/graphs/nodes/route.py`
  - [ ] `route_node(state: GraphState) -> dict` function
  - [ ] Classifies query type: "factual", "procedural", "unsupported"
  - [ ] Returns routing decision
- [ ] Create `src/graphs/nodes/__init__.py`
  - [ ] Export `route_node`
- [ ] Create `tests/graphs/nodes/test_route.py`
  - [ ] Test factual query routing
  - [ ] Test procedural query routing
  - [ ] Test unsupported query detection
  - [ ] Test edge cases
- [ ] Run: `pytest tests/graphs/nodes/test_route.py -v` - all tests pass

### Step 7.3: Retrieve Node
- [ ] Create `src/graphs/nodes/retrieve.py`
  - [ ] `retrieve_node(state: GraphState, retrieval: RetrievalService) -> dict` function
  - [ ] Calls retrieval service with question
  - [ ] Updates state with chunks and search results
  - [ ] Handles empty results
- [ ] Create `tests/graphs/nodes/test_retrieve.py`
  - [ ] Test retrieval updates state correctly
  - [ ] Test handling of empty results
  - [ ] Test with mocked retrieval service
- [ ] Run: `pytest tests/graphs/nodes/test_retrieve.py -v` - all tests pass

### Step 7.4: Answer Node
- [ ] Create `src/graphs/prompts.py`
  - [ ] `RAG_PROMPT` template
  - [ ] `REFUSAL_PROMPT` template
  - [ ] Prompt builder utilities
- [ ] Create `src/graphs/nodes/answer.py`
  - [ ] `answer_node(state: GraphState, llm: BaseLLM) -> dict` function
  - [ ] Builds prompt with context from chunks
  - [ ] Calls LLM for generation
  - [ ] Extracts answer text
- [ ] Create `tests/graphs/nodes/test_answer.py`
  - [ ] Test answer generation with mocked LLM
  - [ ] Test prompt building
  - [ ] Test handling of empty context
- [ ] Run: `pytest tests/graphs/nodes/test_answer.py -v` - all tests pass

### Step 7.5: Verify Grounding Node
- [ ] Create `src/graphs/grounding.py`
  - [ ] `GroundingChecker` class
  - [ ] `check_grounding(answer: str, chunks: list[Chunk]) -> GroundingResult` method
- [ ] Create `src/graphs/nodes/verify.py`
  - [ ] `verify_grounding_node(state: GraphState) -> dict` function
  - [ ] Checks if answer is supported by chunks
  - [ ] Calculates confidence score
  - [ ] Identifies unsupported claims
- [ ] Create `tests/graphs/nodes/test_verify.py`
  - [ ] Test grounding check with supported answer
  - [ ] Test grounding check with unsupported answer
  - [ ] Test confidence calculation
  - [ ] Test with mocked grounding checker
- [ ] Run: `pytest tests/graphs/nodes/test_verify.py -v` - all tests pass

### Step 7.6: Retry/Refuse Logic
- [ ] Create `src/graphs/nodes/retry.py`
  - [ ] `retry_node(state: GraphState) -> dict` function
  - [ ] Increments retry count
  - [ ] Decides whether to retry or refuse
- [ ] Create `src/graphs/nodes/refuse.py`
  - [ ] `refuse_node(state: GraphState) -> dict` function
  - [ ] Generates refusal response
  - [ ] Provides helpful refusal reason
- [ ] Create `tests/graphs/nodes/test_retry.py`
  - [ ] Test retry count increment
  - [ ] Test retry decision logic
  - [ ] Test max retries reached
- [ ] Create `tests/graphs/nodes/test_refuse.py`
  - [ ] Test refusal response generation
  - [ ] Test refusal reason is helpful
- [ ] Run: `pytest tests/graphs/nodes/test_retry.py tests/graphs/nodes/test_refuse.py -v` - all tests pass

### Step 7.7: Graph Assembly
- [ ] Create `src/graphs/qna_graph.py`
  - [ ] `create_qna_graph(retrieval, llm, config) -> StateGraph` function
  - [ ] Wire all nodes together
  - [ ] Add conditional edges for retry/refuse
  - [ ] Compile graph
- [ ] Update `src/graphs/__init__.py`
  - [ ] Export `create_qna_graph`
- [ ] Create `tests/graphs/test_qna_graph.py`
  - [ ] Test graph creation
  - [ ] Test graph execution with sample query
  - [ ] Test retry path
  - [ ] Test refusal path
  - [ ] Test successful path
- [ ] Run: `pytest tests/graphs/ -v` - all tests pass

---

## CHUNK 8: LLM Providers

### Step 8.1: Base LLM Interface
- [ ] Create `src/llm/base.py`
  - [ ] `BaseLLM` abstract class
  - [ ] `generate(prompt: str, **kwargs) -> str` method
  - [ ] `generate_with_context(prompt: str, context: list[str], **kwargs) -> str` method
  - [ ] `count_tokens(text: str) -> int` method
  - [ ] `model_name` property
- [ ] Create `src/llm/__init__.py`
  - [ ] Export `BaseLLM`
- [ ] Create `tests/llm/test_base.py`
  - [ ] Test BaseLLM cannot be instantiated
  - [ ] Test subclass must implement all methods
- [ ] Run: `pytest tests/llm/test_base.py -v` - all tests pass

### Step 8.2: Bedrock Claude
- [ ] Create `src/llm/bedrock.py`
  - [ ] `BedrockLLM` class extending `BaseLLM`
  - [ ] Uses boto3 Bedrock client
  - [ ] Support Claude 3 models
  - [ ] Handle streaming (optional)
  - [ ] Implement retry logic
- [ ] Create `tests/llm/test_bedrock.py`
  - [ ] Mock boto3 client for unit tests
  - [ ] Test generate method
  - [ ] Test token counting
  - [ ] Test error handling
- [ ] Run: `pytest tests/llm/test_bedrock.py -v` - all tests pass

### Step 8.3: OpenAI GPT-4
- [ ] Create `src/llm/openai_llm.py`
  - [ ] `OpenAILLM` class extending `BaseLLM`
  - [ ] Uses openai library
  - [ ] Support GPT-4 and GPT-3.5-turbo
  - [ ] Handle streaming (optional)
  - [ ] Implement retry logic
- [ ] Create `tests/llm/test_openai.py`
  - [ ] Mock OpenAI client for unit tests
  - [ ] Test generate method
  - [ ] Test token counting
  - [ ] Test error handling
- [ ] Run: `pytest tests/llm/test_openai.py -v` - all tests pass

### Step 8.4: Local Ollama
- [ ] Create `src/llm/ollama.py`
  - [ ] `OllamaLLM` class extending `BaseLLM`
  - [ ] Uses ollama library or HTTP API
  - [ ] Support various local models
  - [ ] Handle connection errors
- [ ] Create `tests/llm/test_ollama.py`
  - [ ] Mock Ollama for unit tests
  - [ ] Test generate method
  - [ ] Test connection error handling
- [ ] Run: `pytest tests/llm/test_ollama.py -v` - all tests pass

### Step 8.5: LLM Factory
- [ ] Create `src/llm/factory.py`
  - [ ] `LLMFactory` class
  - [ ] `get_llm(config: LLMConfig) -> BaseLLM` method
  - [ ] Support providers: "bedrock_claude", "openai", "ollama"
- [ ] Update `src/llm/__init__.py`
  - [ ] Export all LLM classes and factory
- [ ] Create `tests/llm/test_factory.py`
  - [ ] Test factory returns correct type
  - [ ] Test factory with different configs
- [ ] Run: `pytest tests/llm/ -v` - all tests pass

---

## CHUNK 9: FastAPI Application

### Step 9.1: FastAPI Setup
- [ ] Create `src/app/main.py`
  - [ ] FastAPI app instance
  - [ ] CORS middleware
  - [ ] Exception handlers
  - [ ] OpenAPI configuration
  - [ ] Lifespan context manager for initialization
- [ ] Create `src/app/__init__.py`
  - [ ] Export app
- [ ] Create `tests/app/test_main.py`
  - [ ] Test app creation
  - [ ] Test CORS configuration
  - [ ] Test OpenAPI docs are available
- [ ] Run: `pytest tests/app/test_main.py -v` - all tests pass

### Step 9.2: Query Endpoint
- [ ] Create `src/app/schemas.py`
  - [ ] `QueryRequest` model (question, mode, k)
  - [ ] `QueryResponse` model (answer, citations, confidence, refusal_reason, latency_ms)
- [ ] Create `src/app/routes/query.py`
  - [ ] POST `/query` endpoint
  - [ ] Wire to LangGraph workflow
  - [ ] Handle errors gracefully
  - [ ] Return structured response
- [ ] Update `src/app/main.py`
  - [ ] Include query router
- [ ] Create `tests/app/test_query.py`
  - [ ] Test query endpoint with valid request
  - [ ] Test query endpoint with invalid request
  - [ ] Test error handling
  - [ ] Test response format
- [ ] Run: `pytest tests/app/test_query.py -v` - all tests pass

### Step 9.3: Health Endpoint
- [ ] Create `src/app/routes/health.py`
  - [ ] GET `/health` endpoint
  - [ ] Check component health
  - [ ] Return detailed status
- [ ] Update `src/app/main.py`
  - [ ] Include health router
- [ ] Create `tests/app/test_health.py`
  - [ ] Test health endpoint returns 200
  - [ ] Test health check with unhealthy component
- [ ] Run: `pytest tests/app/test_health.py -v` - all tests pass

### Step 9.4: Ingestion Endpoint
- [ ] Create `src/app/routes/ingest.py`
  - [ ] POST `/ingest` endpoint
  - [ ] Trigger ingestion pipeline
  - [ ] Return ingestion status
  - [ ] Support async ingestion (optional)
- [ ] Update `src/app/main.py`
  - [ ] Include ingest router
- [ ] Create `tests/app/test_ingest.py`
  - [ ] Test ingest endpoint
  - [ ] Test ingestion status
  - [ ] Test error handling
- [ ] Run: `pytest tests/app/test_ingest.py -v` - all tests pass

### Step 9.5: Error Handling
- [ ] Create `src/app/exceptions.py`
  - [ ] HTTPException subclasses
  - [ ] Error response models
- [ ] Create `src/app/middleware.py`
  - [ ] Error handling middleware
  - [ ] Request logging middleware
  - [ ] Correlation ID middleware
- [ ] Update `src/app/main.py`
  - [ ] Add middleware
- [ ] Create `tests/app/test_errors.py`
  - [ ] Test various error scenarios
  - [ ] Test error response format
  - [ ] Test correlation ID propagation
- [ ] Run: `pytest tests/app/ -v` - all tests pass

---

## CHUNK 10: Evaluation Harness

### Step 10.1: Eval Dataset Format
- [ ] Create `src/eval/dataset.py`
  - [ ] `EvalDataset` model
  - [ ] `EvalQuestion` model
  - [ ] Dataset loader from JSON/YAML
- [ ] Create `src/eval/__init__.py`
  - [ ] Export dataset classes
- [ ] Create `src/eval/datasets/sample_qna.yaml`
  - [ ] Sample evaluation dataset with 10 questions
- [ ] Create `tests/eval/test_dataset.py`
  - [ ] Test dataset loading
  - [ ] Test dataset validation
- [ ] Run: `pytest tests/eval/test_dataset.py -v` - all tests pass

### Step 10.2: Groundedness Scorer
- [ ] Create `src/eval/scorers/groundedness.py`
  - [ ] `GroundednessScorer` class
  - [ ] `score(answer: str, chunks: list[Chunk]) -> float` method
- [ ] Create `tests/eval/scorers/test_groundedness.py`
  - [ ] Test scoring with grounded answer
  - [ ] Test scoring with ungrounded answer
  - [ ] Test scoring with partial grounding
- [ ] Run: `pytest tests/eval/scorers/test_groundedness.py -v` - all tests pass

### Step 10.3: Relevance Scorer
- [ ] Create `src/eval/scorers/relevance.py`
  - [ ] `RelevanceScorer` class
  - [ ] `score(answer: str, question: str) -> float` method
- [ ] Create `tests/eval/scorers/test_relevance.py`
  - [ ] Test scoring with relevant answer
  - [ ] Test scoring with irrelevant answer
  - [ ] Test scoring with partial relevance
- [ ] Run: `pytest tests/eval/scorers/test_relevance.py -v` - all tests pass

### Step 10.4: Refusal Correctness
- [ ] Create `src/eval/scorers/refusal.py`
  - [ ] `RefusalScorer` class
  - [ ] `score(result: QueryResult, expected: EvalQuestion) -> bool` method
- [ ] Create `tests/eval/scorers/test_refusal.py`
  - [ ] Test correct refusal
  - [ ] Test incorrect refusal
  - [ ] Test incorrect answer (should have refused)
- [ ] Run: `pytest tests/eval/scorers/test_refusal.py -v` - all tests pass

### Step 10.5: Evaluation Runner and Reports
- [ ] Create `src/eval/runner.py`
  - [ ] `EvalRunner` class
  - [ ] `run(dataset: EvalDataset, graph: StateGraph) -> EvalReport` method
- [ ] Create `src/eval/report.py`
  - [ ] `EvalReport` model
  - [ ] Aggregate metrics
  - [ ] Per-question results
  - [ ] `to_json()`, `to_html()` methods
- [ ] Create `scripts/eval.py`
  - [ ] CLI script to run evaluation
  - [ ] Arguments: --suite, --output
  - [ ] Print summary and save report
- [ ] Create `tests/eval/test_runner.py`
  - [ ] Test evaluation runner
  - [ ] Test report generation
- [ ] Run: `pytest tests/eval/ -v` - all tests pass

---

## CHUNK 11: Integration & Polish

### Step 11.1: CLI Scripts
- [ ] Update `scripts/ingest.py`
  - [ ] Full implementation
  - [ ] Progress bar support
  - [ ] Error handling
  - [ ] Summary output
- [ ] Update `scripts/eval.py`
  - [ ] Full implementation
  - [ ] Multiple output formats
  - [ ] Comparison with baseline
- [ ] Create `scripts/query.py`
  - [ ] Interactive query CLI
  - [ ] Debug mode with intermediate steps
- [ ] Create `tests/test_scripts.py`
  - [ ] Test CLI scripts work correctly
- [ ] Run: `pytest tests/test_scripts.py -v` - all tests pass

### Step 11.2: Docker Setup
- [ ] Create `Dockerfile`
  - [ ] Multi-stage build
  - [ ] Python 3.11 base
  - [ ] Install dependencies
  - [ ] Copy application code
  - [ ] Set up entry point
- [ ] Create `docker-compose.yml`
  - [ ] API service
  - [ ] Optional: local LLM service
- [ ] Create `.dockerignore`
- [ ] Test Docker build: `docker build -t graphrag-tutor .`

### Step 11.3: Documentation
- [ ] Update `README.md`
  - [ ] Full installation instructions
  - [ ] Configuration guide
  - [ ] API documentation
  - [ ] Usage examples
- [ ] Create `docs/API.md`
  - [ ] Detailed API documentation
  - [ ] Request/response examples
  - [ ] Error codes
- [ ] Create `docs/CONFIGURATION.md`
  - [ ] All configuration options
  - [ ] Environment variables
  - [ ] Provider setup
- [ ] Create `docs/DEPLOYMENT.md`
  - [ ] Docker deployment
  - [ ] Cloud deployment options
  - [ ] Scaling considerations
- [ ] Create `docs/EVALUATION.md`
  - [ ] How to run evaluations
  - [ ] Interpreting results
  - [ ] Creating custom datasets

### Step 11.4: End-to-End Testing
- [ ] Create `tests/e2e/test_full_pipeline.py`
  - [ ] Test full ingestion to query flow
  - [ ] Test with sample documents
  - [ ] Verify answer quality
- [ ] Create `tests/e2e/test_api.py`
  - [ ] Test API endpoints with real components
  - [ ] Test error scenarios
  - [ ] Test concurrent requests
- [ ] Create `tests/e2e/test_eval.py`
  - [ ] Run evaluation and verify scores meet thresholds
- [ ] Create `pytest.ini`
  - [ ] Configure test markers
  - [ ] Configure slow tests
  - [ ] Configure e2e tests
- [ ] Run: `pytest tests/e2e/ -v --run-e2e` - all tests pass

### Step 11.5: Final Integration
- [ ] Update `src/__init__.py`
  - [ ] Export main classes
  - [ ] Version info
- [ ] Update `src/app/main.py`
  - [ ] Wire all components together
  - [ ] Initialize on startup
  - [ ] Graceful shutdown
- [ ] Update `configs/default.yaml`
  - [ ] Production-ready defaults
  - [ ] Comments explaining options
- [ ] Add sample corpus in `data/sample_docs/`
  - [ ] A few sample documents for testing
- [ ] Final test run: `pytest tests/ -v --cov=src`
  - [ ] All tests pass
  - [ ] Coverage > 80%

---

## Summary Statistics

- **Total Chunks**: 11
- **Total Steps**: 45
- **Total Checklist Items**: ~350+

## Progress Tracking

| Chunk | Steps | Status |
|-------|-------|--------|
| 1. Foundation | 4 | 🔲 Not Started |
| 2. Connectors | 4 | 🔲 Not Started |
| 3. Ingestion | 4 | 🔲 Not Started |
| 4. Embeddings | 5 | 🔲 Not Started |
| 5. Vector Store | 4 | 🔲 Not Started |
| 6. Retrieval | 4 | 🔲 Not Started |
| 7. LangGraph | 7 | 🔲 Not Started |
| 8. LLMs | 5 | 🔲 Not Started |
| 9. FastAPI | 5 | 🔲 Not Started |
| 10. Evaluation | 5 | 🔲 Not Started |
| 11. Integration | 5 | 🔲 Not Started |

**Legend**: 🔲 Not Started | 🔄 In Progress | ✅ Complete
