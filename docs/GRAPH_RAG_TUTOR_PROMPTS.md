# Grounded GraphRAG Tutor - LLM Implementation Prompts

This document contains a series of prompts for implementing the Grounded GraphRAG Tutor project in a test-driven, incremental manner. Each prompt builds on the previous work and ends with integration.

---

## CHUNK 1: Foundation & Project Structure

### Prompt 1.1: Project Skeleton

```
Create the project skeleton for a Grounded GraphRAG Tutor service. This is a Python project that will use LangGraph for RAG workflow orchestration.

Create the following directory structure:
- src/
  - connectors/     (document source adapters)
  - ingestion/      (clean + chunk pipeline)
  - embeddings/     (embedding providers)
  - store/          (vector store adapters)
  - retrieval/      (retrieval + citations)
  - graphs/         (LangGraph workflows)
  - app/            (FastAPI API layer)
  - eval/           (eval datasets + runner)
  - __init__.py files in each directory
- tests/
  - mirrors src structure
  - __init__.py
- configs/
  - default.yaml
- scripts/
  - ingest.py
  - eval.py
- pyproject.toml (with project metadata, use setuptools)
- .env.example
- .gitignore (Python-specific)
- README.md (copy from spec)

For pyproject.toml, include these dependencies:
- langchain>=0.1.0
- langgraph>=0.0.20
- fastapi>=0.109.0
- uvicorn>=0.27.0
- pydantic>=2.0.0
- pydantic-settings>=2.0.0
- pyyaml>=6.0
- python-dotenv>=1.0.0
- faiss-cpu>=1.7.4
- chromadb>=0.4.0
- sentence-transformers>=2.2.0
- boto3>=1.34.0 (for Bedrock)
- openai>=1.10.0
- pypdf>=4.0.0
- beautifulsoup4>=4.12.0
- pytest>=7.4.0
- pytest-asyncio>=0.23.0
- httpx>=0.26.0 (for testing)

Create placeholder __init__.py files that export nothing for now.

Write tests that verify:
1. The project structure exists
2. Required directories are present
3. Config file exists and is valid YAML

Run: pytest tests/ -v
All tests should pass.
```

---

### Prompt 1.2: Configuration System

```
Build on the project skeleton by implementing a configuration system that loads settings from YAML files with environment variable overrides.

Create the following files:

1. src/config.py:
   - Settings class using pydantic-settings
   - Supports loading from configs/default.yaml
   - Environment variable overrides (e.g., CORPUS, VECTORSTORE, EMBEDDINGS, LLM, GRAPH)
   - Nested settings for each component:
     - CorpusConfig (name, path, connector_type)
     - VectorStoreConfig (type, persist_directory, collection_name)
     - EmbeddingsConfig (provider, model_name, dimension)
     - LLMConfig (provider, model_name, temperature, max_tokens)
     - GraphConfig (type, max_retries, refusal_threshold)

2. configs/default.yaml:
   - Default configuration for local development
   - Use local_st embeddings, faiss vectorstore
   - Sample corpus config pointing to a test directory

3. .env.example:
   - All supported environment variables with example values
   - Grouped by component

4. tests/test_config.py:
   - Test loading from YAML
   - Test environment variable override
   - Test validation errors for invalid config
   - Test default values

The config should be a singleton that can be imported anywhere:
```python
from src.config import settings
print(settings.llm.model_name)
```

Run: pytest tests/test_config.py -v
All tests should pass.
```

---

### Prompt 1.3: Logging Infrastructure

```
Add structured logging infrastructure to the project. This will be used for debugging and monitoring the RAG pipeline.

Create the following files:

1. src/logging_config.py:
   - Configure structured JSON logging
   - Support log level from config/environment (LOG_LEVEL)
   - Add correlation ID support for request tracing
   - Create a get_logger(name) function for consistent logger creation
   - Include timestamp, level, module, message, and extra fields

2. src/__init__.py:
   - Initialize logging on import
   - Export get_logger

3. tests/test_logging.py:
   - Test logger creation
   - Test log level configuration
   - Test JSON format output
   - Test correlation ID propagation

Example usage should be:
```python
from src import get_logger
logger = get_logger(__name__)
logger.info("Processing document", extra={"doc_id": "123", "source": "s3"})
```

The log output should be JSON like:
```json
{"timestamp": "2024-01-15T10:30:00Z", "level": "INFO", "module": "connectors.local", "message": "Processing document", "doc_id": "123", "source": "s3", "correlation_id": "abc-123"}
```

Run: pytest tests/test_logging.py -v
All tests should pass.
```

---

### Prompt 1.4: Type Definitions

```
Create the core type definitions that will be used throughout the project. These are Pydantic models that define the data structures.

Create the following files:

1. src/types.py:
   - Document (id, content, source, metadata, created_at)
   - Chunk (id, document_id, content, start_idx, end_idx, metadata)
   - Citation (source, chunk_id, text, score)
   - Answer (text, citations, confidence, refusal_reason)
   - QueryResult (question, answer, mode, latency_ms)
   - IngestResult (documents_count, chunks_count, errors)
   - EmbeddingVector (alias for list[float])
   - ProviderConfig (base for provider configs)

2. src/exceptions.py:
   - RAGError (base exception)
   - ConnectorError
   - IngestionError
   - EmbeddingError
   - VectorStoreError
   - RetrievalError
   - LLMError
   - ConfigurationError

3. tests/test_types.py:
   - Test Document creation and validation
   - Test Chunk creation with metadata
   - Test Citation with score validation
   - Test Answer with optional refusal
   - Test QueryResult serialization
   - Test exception hierarchy

All models should:
- Use Pydantic v2
- Have proper type hints
- Include docstrings
- Be immutable where appropriate (frozen=True for read-only models)

Run: pytest tests/test_types.py -v
All tests should pass.
```

---

## CHUNK 2: Document Connectors

### Prompt 2.1: Base Connector Interface

```
Create the abstract base class for document connectors. This defines the interface that all connector implementations must follow.

Create the following files:

1. src/connectors/base.py:
   - Abstract base class BaseConnector
   - Abstract methods:
     - load() -> list[Document]
     - list_documents() -> list[str] (returns available document IDs/names)
   - Concrete methods:
     - validate_source() -> bool (check if source is accessible)
   - Include proper docstrings

2. src/connectors/__init__.py:
   - Export BaseConnector
   - Create ConnectorRegistry for registering implementations

3. tests/connectors/test_base.py:
   - Test that BaseConnector cannot be instantiated directly
   - Test that subclass must implement all abstract methods
   - Test ConnectorRegistry registration

Example implementation pattern:
```python
from src.connectors import BaseConnector

class MyConnector(BaseConnector):
    def load(self) -> list[Document]:
        # Implementation
        pass
    
    def list_documents(self) -> list[str]:
        # Implementation
        pass
```

Run: pytest tests/connectors/test_base.py -v
All tests should pass.
```

---

### Prompt 2.2: Document Model Enhancement

```
Enhance the Document model and add document processing utilities.

Update/Create the following files:

1. src/types.py (update):
   - Add DocumentType enum (PDF, TXT, MD, HTML, JSON)
   - Add DocumentMetadata model (author, title, created, modified, tags)
   - Update Document to include document_type and full_metadata

2. src/connectors/document.py:
   - create_document_id() function (generates deterministic ID from source)
   - detect_document_type() function (from file extension or content)
   - extract_metadata() function (basic metadata extraction)

3. tests/connectors/test_document.py:
   - Test document ID generation is deterministic
   - Test document type detection for various extensions
   - Test metadata extraction

Example:
```python
from src.connectors.document import create_document_id, detect_document_type

doc_id = create_document_id("s3://bucket/doc.pdf")  # Returns consistent hash
doc_type = detect_document_type("report.pdf")  # Returns DocumentType.PDF
```

Run: pytest tests/connectors/test_document.py -v
All tests should pass.
```

---

### Prompt 2.3: Local File Connector

```
Implement a connector for loading documents from the local filesystem.

Create the following files:

1. src/connectors/local.py:
   - LocalConnector class extending BaseConnector
   - Constructor takes source_path (str or Path)
   - Supports file types: .pdf, .txt, .md, .html
   - load() reads all supported files from directory
   - list_documents() returns relative file paths
   - validate_source() checks directory exists
   - Handle encoding detection for text files
   - Use pypdf for PDF extraction

2. tests/connectors/test_local.py:
   - Create fixture with sample documents in temp directory
   - Test loading PDF files
   - Test loading text files
   - Test loading markdown files
   - Test loading HTML files
   - Test handling of unsupported file types (skip with warning)
   - Test list_documents returns correct paths
   - Test validate_source with missing directory

3. tests/fixtures/documents/:
   - Create sample test documents:
     - sample.txt (plain text)
     - sample.md (markdown)
     - sample.html (simple HTML)

Example usage:
```python
from src.connectors.local import LocalConnector

connector = LocalConnector(source_path="./data/documents")
documents = connector.load()
for doc in documents:
    print(f"Loaded: {doc.source} ({len(doc.content)} chars)")
```

Run: pytest tests/connectors/test_local.py -v
All tests should pass.
```

---

### Prompt 2.4: Connector Factory

```
Create a factory for instantiating connectors based on configuration.

Create the following files:

1. src/connectors/factory.py:
   - ConnectorFactory class
   - get_connector(config: CorpusConfig) -> BaseConnector
   - Support connector types: "local", "s3", "web"
   - Raise ConfigurationError for unknown types
   - Register connectors dynamically

2. src/connectors/__init__.py (update):
   - Export ConnectorFactory
   - Export all connector classes

3. src/connectors/s3.py (stub):
   - S3Connector stub class (raises NotImplementedError)
   - Document that this is a placeholder for future implementation

4. src/connectors/web.py (stub):
   - WebConnector stub class (raises NotImplementedError)
   - Document that this is a placeholder for future implementation

5. tests/connectors/test_factory.py:
   - Test factory returns correct connector type
   - Test factory raises error for unknown type
   - Test factory with local connector config
   - Test that stub connectors raise NotImplementedError when load() is called

Example usage:
```python
from src.connectors import ConnectorFactory
from src.config import CorpusConfig

config = CorpusConfig(connector_type="local", path="./data")
connector = ConnectorFactory.get_connector(config)
documents = connector.load()
```

Run: pytest tests/connectors/ -v
All tests should pass.
```

---

## CHUNK 3: Ingestion Pipeline

### Prompt 3.1: Text Cleaning

```
Create text cleaning utilities for normalizing document content.

Create the following files:

1. src/ingestion/cleaning.py:
   - TextCleaner class with methods:
     - normalize_whitespace(text: str) -> str
     - remove_html_tags(text: str) -> str
     - normalize_unicode(text: str) -> str
     - remove_boilerplate(text: str) -> str (remove common headers/footers)
     - clean(text: str) -> str (applies all cleaning steps)
   - Configurable cleaning options

2. tests/ingestion/test_cleaning.py:
   - Test whitespace normalization
   - Test HTML tag removal
   - Test unicode normalization
   - Test boilerplate removal
   - Test full cleaning pipeline
   - Test that cleaning preserves important content

Example:
```python
from src.ingestion.cleaning import TextCleaner

cleaner = TextCleaner()
dirty = "<p>  Hello   World  </p>"
clean = cleaner.clean(dirty)  # "Hello World"
```

Run: pytest tests/ingestion/test_cleaning.py -v
All tests should pass.
```

---

### Prompt 3.2: Chunking Strategies

```
Implement text chunking strategies for breaking documents into processable pieces.

Create the following files:

1. src/ingestion/chunking.py:
   - BaseChunker abstract class:
     - chunk(text: str, metadata: dict) -> list[Chunk]
   - FixedSizeChunker:
     - Chunks by character count
     - Supports overlap
     - Respects word boundaries
   - SentenceChunker:
     - Chunks by sentences
     - Supports min/max chunk size
     - Groups sentences to meet size constraints
   - ChunkMetadata model for tracking chunk position

2. tests/ingestion/test_chunking.py:
   - Test FixedSizeChunker with various sizes
   - Test overlap functionality
   - Test word boundary respect
   - Test SentenceChunker sentence detection
   - Test chunk size constraints
   - Test metadata is correctly attached
   - Test edge cases (empty text, very long words)

Example:
```python
from src.ingestion.chunking import FixedSizeChunker, SentenceChunker

fixed_chunker = FixedSizeChunker(chunk_size=500, overlap=50)
chunks = fixed_chunker.chunk(long_text, {"doc_id": "123"})

sentence_chunker = SentenceChunker(min_size=200, max_size=1000)
chunks = sentence_chunker.chunk(long_text, {"doc_id": "123"})
```

Run: pytest tests/ingestion/test_chunking.py -v
All tests should pass.
```

---

### Prompt 3.3: Metadata Extraction

```
Implement metadata extraction for document chunks.

Create the following files:

1. src/ingestion/metadata.py:
   - ChunkMetadataExtractor class:
     - extract(chunk: Chunk, document: Document) -> dict
   - Functions:
     - extract_position_info(chunk: Chunk) -> dict (page, paragraph, position)
     - extract_context(chunk: Chunk, document: Document) -> dict (surrounding context)
     - generate_chunk_id(document_id: str, chunk_index: int) -> str

2. tests/ingestion/test_metadata.py:
   - Test position info extraction
   - Test context extraction
   - Test chunk ID generation (deterministic)
   - Test metadata is serializable

Example:
```python
from src.ingestion.metadata import ChunkMetadataExtractor, generate_chunk_id

extractor = ChunkMetadataExtractor()
chunk_id = generate_chunk_id("doc_123", 0)  # "doc_123_chunk_0"
metadata = extractor.extract(chunk, document)
```

Run: pytest tests/ingestion/test_metadata.py -v
All tests should pass.
```

---

### Prompt 3.4: Ingestion Orchestrator

```
Create the ingestion pipeline that wires together connectors, cleaning, and chunking.

Create the following files:

1. src/ingestion/pipeline.py:
   - IngestionPipeline class:
     - __init__(connector, cleaner, chunker)
     - run() -> IngestResult
     - run_with_progress() -> Iterator[IngestProgress] (for large datasets)
   - IngestProgress model (documents_processed, chunks_created, current_file)

2. src/ingestion/__init__.py:
   - Export IngestionPipeline, TextCleaner, FixedSizeChunker, SentenceChunker

3. tests/ingestion/test_pipeline.py:
   - Test full pipeline with local connector
   - Test that documents are cleaned before chunking
   - Test that chunks have correct metadata
   - Test error handling for corrupted files
   - Test progress reporting

4. scripts/ingest.py:
   - CLI script to run ingestion
   - Arguments: --corpus, --config
   - Outputs progress and summary

Example:
```python
from src.ingestion import IngestionPipeline, TextCleaner, FixedSizeChunker
from src.connectors import LocalConnector

connector = LocalConnector("./data")
pipeline = IngestionPipeline(
    connector=connector,
    cleaner=TextCleaner(),
    chunker=FixedSizeChunker(chunk_size=500)
)
result = pipeline.run()
print(f"Ingested {result.documents_count} docs, {result.chunks_count} chunks")
```

Run: pytest tests/ingestion/ -v
All tests should pass.
```

---

## CHUNK 4: Embedding Providers

### Prompt 4.1: Base Embedding Interface

```
Create the abstract base class for embedding providers.

Create the following files:

1. src/embeddings/base.py:
   - BaseEmbeddings abstract class:
     - embed_documents(texts: list[str]) -> list[EmbeddingVector]
     - embed_query(text: str) -> EmbeddingVector
     - dimension property (int)
   - EmbeddingCache protocol (for caching implementations)

2. src/embeddings/__init__.py:
   - Export BaseEmbeddings

3. tests/embeddings/test_base.py:
   - Test that BaseEmbeddings cannot be instantiated
   - Test that subclass must implement all methods
   - Test dimension property is required

Example:
```python
from src.embeddings import BaseEmbeddings

class MyEmbeddings(BaseEmbeddings):
    @property
    def dimension(self) -> int:
        return 768
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 768 for _ in texts]
    
    def embed_query(self, text: str) -> list[float]:
        return [0.1] * 768
```

Run: pytest tests/embeddings/test_base.py -v
All tests should pass.
```

---

### Prompt 4.2: Local Sentence Transformers

```
Implement local embeddings using sentence-transformers library.

Create the following files:

1. src/embeddings/local.py:
   - LocalEmbeddings class extending BaseEmbeddings
   - Uses sentence-transformers library
   - Default model: "all-MiniLM-L6-v2" (fast, good quality)
   - Configurable model name
   - Support for batch embedding
   - GPU detection and usage

2. tests/embeddings/test_local.py:
   - Test embedding single text
   - Test embedding multiple texts
   - Test dimension matches model
   - Test embeddings are normalized (if applicable)
   - Test with different model names

Note: These tests may be slow. Use pytest markers:
```python
@pytest.mark.slow
def test_local_embeddings():
    ...
```

Example:
```python
from src.embeddings.local import LocalEmbeddings

embeddings = LocalEmbeddings(model_name="all-MiniLM-L6-v2")
vectors = embeddings.embed_documents(["Hello world", "Test sentence"])
print(f"Dimension: {embeddings.dimension}")  # 384
```

Run: pytest tests/embeddings/test_local.py -v -m "not slow" (for fast tests)
Run: pytest tests/embeddings/test_local.py -v (for all tests)
```

---

### Prompt 4.3: Bedrock Titan Embeddings

```
Implement embeddings using AWS Bedrock Titan model.

Create the following files:

1. src/embeddings/bedrock.py:
   - BedrockEmbeddings class extending BaseEmbeddings
   - Uses boto3 Bedrock client
   - Model: amazon.titan-embed-text-v1
   - Handle AWS credentials from environment
   - Implement retry logic with exponential backoff
   - Handle rate limiting

2. tests/embeddings/test_bedrock.py:
   - Mock boto3 client for unit tests
   - Test embedding single text
   - Test embedding multiple texts (batch)
   - Test retry logic
   - Test error handling
   - Skip integration tests if AWS credentials not available

Example:
```python
from src.embeddings.bedrock import BedrockEmbeddings

embeddings = BedrockEmbeddings()
vector = embeddings.embed_query("What is federalism?")
print(f"Dimension: {embeddings.dimension}")  # 1536
```

Run: pytest tests/embeddings/test_bedrock.py -v
All tests should pass.
```

---

### Prompt 4.4: OpenAI Embeddings

```
Implement embeddings using OpenAI API.

Create the following files:

1. src/embeddings/openai_emb.py:
   - OpenAIEmbeddings class extending BaseEmbeddings
   - Uses openai library
   - Model: text-embedding-3-small (default) or text-embedding-3-large
   - Handle API key from environment
   - Implement retry logic
   - Handle rate limiting

2. tests/embeddings/test_openai.py:
   - Mock OpenAI client for unit tests
   - Test embedding single text
   - Test embedding multiple texts
   - Test retry logic
   - Test error handling
   - Skip integration tests if OPENAI_API_KEY not available

Example:
```python
from src.embeddings.openai_emb import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectors = embeddings.embed_documents(["Hello", "World"])
print(f"Dimension: {embeddings.dimension}")  # 1536
```

Run: pytest tests/embeddings/test_openai.py -v
All tests should pass.
```

---

### Prompt 4.5: Embedding Factory and Cache

```
Create factory for embedding providers and implement a disk-based cache.

Create the following files:

1. src/embeddings/factory.py:
   - EmbeddingsFactory class
   - get_embeddings(config: EmbeddingsConfig) -> BaseEmbeddings
   - Support providers: "local_st", "bedrock_titan", "openai"

2. src/embeddings/cache.py:
   - DiskEmbeddingCache class
   - Cache key: hash of text + model name
   - Store on disk in configurable directory
   - get(text: str, model: str) -> EmbeddingVector | None
   - set(text: str, model: str, vector: EmbeddingVector)
   - Clear cache functionality

3. src/embeddings/cached.py:
   - CachedEmbeddings wrapper class
   - Wraps any BaseEmbeddings with cache
   - Transparent caching

4. src/embeddings/__init__.py (update):
   - Export all embedding classes and factory

5. tests/embeddings/test_factory.py:
   - Test factory returns correct type
   - Test factory with different configs

6. tests/embeddings/test_cache.py:
   - Test cache hit and miss
   - Test cache persistence
   - Test cache key generation

Example:
```python
from src.embeddings import EmbeddingsFactory, CachedEmbeddings, DiskEmbeddingCache
from src.config import EmbeddingsConfig

config = EmbeddingsConfig(provider="local_st", model_name="all-MiniLM-L6-v2")
embeddings = EmbeddingsFactory.get_embeddings(config)

# With cache
cache = DiskEmbeddingCache("./cache/embeddings")
cached_embeddings = CachedEmbeddings(embeddings, cache)
```

Run: pytest tests/embeddings/ -v
All tests should pass.
```

---

## CHUNK 5: Vector Store

### Prompt 5.1: Base Vector Store Interface

```
Create the abstract base class for vector stores.

Create the following files:

1. src/store/base.py:
   - BaseVectorStore abstract class:
     - add_embeddings(chunks: list[Chunk], embeddings: list[EmbeddingVector]) -> int
     - similarity_search(query_embedding: EmbeddingVector, k: int) -> list[SearchResult]
     - delete(chunk_ids: list[str]) -> int
     - count() -> int
     - clear() -> None
   - SearchResult model (chunk_id, score, chunk, metadata)

2. src/store/__init__.py:
   - Export BaseVectorStore, SearchResult

3. tests/store/test_base.py:
   - Test that BaseVectorStore cannot be instantiated
   - Test that subclass must implement all methods

Example:
```python
from src.store import BaseVectorStore, SearchResult

class MyStore(BaseVectorStore):
    def add_embeddings(self, chunks, embeddings) -> int:
        return len(chunks)
    
    def similarity_search(self, query_embedding, k) -> list[SearchResult]:
        return []
    
    # ... other methods
```

Run: pytest tests/store/test_base.py -v
All tests should pass.
```

---

### Prompt 5.2: FAISS Implementation

```
Implement vector store using FAISS library.

Create the following files:

1. src/store/faiss_store.py:
   - FAISSVectorStore class extending BaseVectorStore
   - Use faiss.IndexFlatIP (inner product for cosine similarity with normalized vectors)
   - Store metadata separately (FAISS only stores vectors)
   - Persist index to disk
   - Load index from disk
   - Support incremental additions

2. tests/store/test_faiss.py:
   - Test adding embeddings
   - Test similarity search returns correct results
   - Test search with different k values
   - Test persistence (save and load)
   - Test count and clear
   - Test empty store behavior

Example:
```python
from src.store.faiss_store import FAISSVectorStore
from src.types import Chunk

store = FAISSVectorStore(dimension=384, persist_dir="./data/index")
store.add_embeddings(chunks, embeddings)
results = store.similarity_search(query_embedding, k=5)
for result in results:
    print(f"Score: {result.score}, Text: {result.chunk.content[:50]}")
```

Run: pytest tests/store/test_faiss.py -v
All tests should pass.
```

---

### Prompt 5.3: Chroma Implementation

```
Implement vector store using ChromaDB.

Create the following files:

1. src/store/chroma_store.py:
   - ChromaVectorStore class extending BaseVectorStore
   - Use chromadb library
   - Support persistent client
   - Support metadata filtering
   - Collection management

2. tests/store/test_chroma.py:
   - Test adding embeddings
   - Test similarity search
   - Test metadata filtering
   - Test persistence
   - Test count and clear

Example:
```python
from src.store.chroma_store import ChromaVectorStore

store = ChromaVectorStore(
    collection_name="documents",
    persist_directory="./data/chroma"
)
store.add_embeddings(chunks, embeddings)
results = store.similarity_search(query_embedding, k=5)
```

Run: pytest tests/store/test_chroma.py -v
All tests should pass.
```

---

### Prompt 5.4: Vector Store Factory

```
Create factory for vector store instantiation.

Create the following files:

1. src/store/factory.py:
   - VectorStoreFactory class
   - get_store(config: VectorStoreConfig, dimension: int) -> BaseVectorStore
   - Support types: "faiss", "chroma"

2. src/store/__init__.py (update):
   - Export all store classes and factory

3. tests/store/test_factory.py:
   - Test factory returns correct type
   - Test factory with different configs
   - Test dimension is passed correctly

Example:
```python
from src.store import VectorStoreFactory
from src.config import VectorStoreConfig

config = VectorStoreConfig(type="faiss", persist_directory="./data/index")
store = VectorStoreFactory.get_store(config, dimension=384)
```

Run: pytest tests/store/ -v
All tests should pass.
```

---

## CHUNK 6: Retrieval Service

### Prompt 6.1: Similarity Search Service

```
Create the retrieval service that combines embeddings and vector store.

Create the following files:

1. src/retrieval/service.py:
   - RetrievalService class:
     - __init__(embeddings: BaseEmbeddings, store: BaseVectorStore)
     - index_documents(chunks: list[Chunk]) -> int
     - search(query: str, k: int = 5) -> list[SearchResult]
     - search_with_threshold(query: str, k: int, min_score: float) -> list[SearchResult]

2. src/retrieval/__init__.py:
   - Export RetrievalService

3. tests/retrieval/test_service.py:
   - Test document indexing
   - Test search returns results
   - Test search with threshold filters low scores
   - Test empty query handling
   - Test with real embeddings (slow test)

Example:
```python
from src.retrieval import RetrievalService
from src.embeddings import LocalEmbeddings
from src.store import FAISSVectorStore

embeddings = LocalEmbeddings()
store = FAISSVectorStore(dimension=embeddings.dimension)
retrieval = RetrievalService(embeddings, store)

retrieval.index_documents(chunks)
results = retrieval.search("What is federalism?", k=5)
```

Run: pytest tests/retrieval/test_service.py -v
All tests should pass.
```

---

### Prompt 6.2: Citation Extraction

```
Implement citation extraction from search results.

Create the following files:

1. src/retrieval/citations.py:
   - CitationBuilder class:
     - build_citations(results: list[SearchResult]) -> list[Citation]
     - format_citation(citation: Citation, style: str = "default") -> str
   - Support multiple citation styles: default, mla, apa

2. tests/retrieval/test_citations.py:
   - Test citation building from search results
   - Test citation formatting
   - Test different citation styles
   - Test citation with missing metadata

Example:
```python
from src.retrieval.citations import CitationBuilder

builder = CitationBuilder()
citations = builder.build_citations(search_results)
for citation in citations:
    print(builder.format_citation(citation, style="default"))
```

Run: pytest tests/retrieval/test_citations.py -v
All tests should pass.
```

---

### Prompt 6.3: Reranking Support

```
Add optional reranking to improve retrieval quality.

Create the following files:

1. src/retrieval/reranker.py:
   - BaseReranker abstract class:
     - rerank(query: str, results: list[SearchResult]) -> list[SearchResult]
   - CrossEncoderReranker:
     - Uses cross-encoder model for reranking
     - Configurable model name
   - IdentityReranker (no-op, returns same order)

2. tests/retrieval/test_reranker.py:
   - Test identity reranker returns same order
   - Test cross-encoder reranker changes order
   - Test reranking with empty results

Example:
```python
from src.retrieval.reranker import CrossEncoderReranker

reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranked = reranker.rerank("What is federalism?", search_results)
```

Run: pytest tests/retrieval/test_reranker.py -v
All tests should pass.
```

---

### Prompt 6.4: Retrieval Integration

```
Wire together retrieval service with citations and reranking.

Update/Create the following files:

1. src/retrieval/service.py (update):
   - Add reranker parameter to constructor
   - Add build_citations parameter to search
   - Return citations with search results

2. src/retrieval/__init__.py (update):
   - Export all retrieval classes

3. tests/retrieval/test_integration.py:
   - Test full retrieval pipeline
   - Test with reranking enabled
   - Test citation generation

Example:
```python
from src.retrieval import RetrievalService, CitationBuilder, CrossEncoderReranker

retrieval = RetrievalService(
    embeddings=embeddings,
    store=store,
    reranker=CrossEncoderReranker()
)
results = retrieval.search("What is federalism?", k=5)
citations = CitationBuilder().build_citations(results)
```

Run: pytest tests/retrieval/ -v
All tests should pass.
```

---

## CHUNK 7: LangGraph Q&A Workflow

### Prompt 7.1: Graph State Definition

```
Define the state that flows through the LangGraph workflow.

Create the following files:

1. src/graphs/state.py:
   - GraphState TypedDict:
     - question: str
     - chunks: list[Chunk]
     - search_results: list[SearchResult]
     - answer: str | None
     - citations: list[Citation]
     - confidence: float
     - retry_count: int
     - refusal_reason: str | None
     - error: str | None
   - StateBuilder helper class for creating initial state
   - StateValidator for validating state transitions

2. src/graphs/__init__.py:
   - Export GraphState

3. tests/graphs/test_state.py:
   - Test state creation
   - Test state validation
   - Test state immutability where appropriate

Example:
```python
from src.graphs import GraphState, StateBuilder

initial_state = StateBuilder().with_question("What is federalism?").build()
```

Run: pytest tests/graphs/test_state.py -v
All tests should pass.
```

---

### Prompt 7.2: Route Node

```
Implement the route node that classifies queries.

Create the following files:

1. src/graphs/nodes/route.py:
   - route_node(state: GraphState) -> dict
   - Classifies query type: "factual", "procedural", "unsupported"
   - Returns routing decision
   - Uses simple heuristics (can be enhanced with LLM later)

2. src/graphs/nodes/__init__.py:
   - Export route_node

3. tests/graphs/nodes/test_route.py:
   - Test factual query routing
   - Test procedural query routing
   - Test unsupported query detection
   - Test edge cases

Example:
```python
from src.graphs.nodes import route_node

result = route_node({"question": "What is federalism?"})
# Returns {"query_type": "factual"}
```

Run: pytest tests/graphs/nodes/test_route.py -v
All tests should pass.
```

---

### Prompt 7.3: Retrieve Node

```
Implement the retrieve node that fetches relevant chunks.

Create the following files:

1. src/graphs/nodes/retrieve.py:
   - retrieve_node(state: GraphState, retrieval: RetrievalService) -> dict
   - Calls retrieval service with question
   - Updates state with chunks and search results
   - Handles empty results

2. tests/graphs/nodes/test_retrieve.py:
   - Test retrieval updates state correctly
   - Test handling of empty results
   - Test with mocked retrieval service

Example:
```python
from src.graphs.nodes import retrieve_node

result = retrieve_node(
    {"question": "What is federalism?"},
    retrieval_service
)
# Returns {"chunks": [...], "search_results": [...]}
```

Run: pytest tests/graphs/nodes/test_retrieve.py -v
All tests should pass.
```

---

### Prompt 7.4: Answer Node

```
Implement the answer node that generates responses using LLM.

Create the following files:

1. src/graphs/nodes/answer.py:
   - answer_node(state: GraphState, llm: BaseLLM) -> dict
   - Builds prompt with context from chunks
   - Calls LLM for generation
   - Extracts answer text
   - Note: BaseLLM will be defined in Chunk 8

2. src/graphs/prompts.py:
   - RAG_PROMPT template
   - REFUSAL_PROMPT template
   - Prompt builder utilities

3. tests/graphs/nodes/test_answer.py:
   - Test answer generation with mocked LLM
   - Test prompt building
   - Test handling of empty context

Example:
```python
from src.graphs.nodes import answer_node
from src.graphs.prompts import RAG_PROMPT

prompt = RAG_PROMPT.format(
    context="Federalism is...",
    question="What is federalism?"
)
```

Run: pytest tests/graphs/nodes/test_answer.py -v
All tests should pass.
```

---

### Prompt 7.5: Verify Grounding Node

```
Implement the grounding verification node.

Create the following files:

1. src/graphs/nodes/verify.py:
   - verify_grounding_node(state: GraphState) -> dict
   - Checks if answer is supported by retrieved chunks
   - Calculates confidence score
   - Identifies unsupported claims
   - Returns verification result

2. src/graphs/grounding.py:
   - GroundingChecker class
   - check_grounding(answer: str, chunks: list[Chunk]) -> GroundingResult
   - Uses simple text matching (can be enhanced with LLM)

3. tests/graphs/nodes/test_verify.py:
   - Test grounding check with supported answer
   - Test grounding check with unsupported answer
   - Test confidence calculation
   - Test with mocked grounding checker

Example:
```python
from src.graphs.nodes import verify_grounding_node

result = verify_grounding_node({
    "answer": "Federalism is a system of government...",
    "chunks": [Chunk(content="Federalism divides power...")]
})
# Returns {"confidence": 0.85, "is_grounded": True}
```

Run: pytest tests/graphs/nodes/test_verify.py -v
All tests should pass.
```

---

### Prompt 7.6: Retry/Refuse Logic

```
Implement retry and refusal logic.

Create the following files:

1. src/graphs/nodes/retry.py:
   - retry_node(state: GraphState) -> dict
   - Increments retry count
   - Decides whether to retry or refuse
   - Returns decision

2. src/graphs/nodes/refuse.py:
   - refuse_node(state: GraphState) -> dict
   - Generates refusal response
   - Provides helpful refusal reason
   - Returns final state

3. tests/graphs/nodes/test_retry.py:
   - Test retry count increment
   - Test retry decision logic
   - Test max retries reached

4. tests/graphs/nodes/test_refuse.py:
   - Test refusal response generation
   - Test refusal reason is helpful

Example:
```python
from src.graphs.nodes import retry_node, refuse_node

retry_result = retry_node({"retry_count": 0, "confidence": 0.3})
# Returns {"retry_count": 1, "action": "retry"}

refuse_result = refuse_node({"question": "What is the meaning of life?"})
# Returns {"answer": None, "refusal_reason": "No relevant information found..."}
```

Run: pytest tests/graphs/nodes/test_retry.py tests/graphs/nodes/test_refuse.py -v
All tests should pass.
```

---

### Prompt 7.7: Graph Assembly

```
Assemble the LangGraph workflow from all nodes.

Create the following files:

1. src/graphs/qna_graph.py:
   - create_qna_graph(retrieval: RetrievalService, llm: BaseLLM, config: GraphConfig) -> StateGraph
   - Wire all nodes together
   - Add conditional edges for retry/refuse
   - Compile graph

2. src/graphs/__init__.py (update):
   - Export create_qna_graph

3. tests/graphs/test_qna_graph.py:
   - Test graph creation
   - Test graph execution with sample query
   - Test retry path
   - Test refusal path
   - Test successful path

Example:
```python
from src.graphs import create_qna_graph

graph = create_qna_graph(retrieval_service, llm, config)
result = graph.invoke({"question": "What is federalism?"})
print(result["answer"])
```

Run: pytest tests/graphs/ -v
All tests should pass.
```

---

## CHUNK 8: LLM Providers

### Prompt 8.1: Base LLM Interface

```
Create the abstract base class for LLM providers.

Create the following files:

1. src/llm/base.py:
   - BaseLLM abstract class:
     - generate(prompt: str, **kwargs) -> str
     - generate_with_context(prompt: str, context: list[str], **kwargs) -> str
     - count_tokens(text: str) -> int
     - model_name property

2. src/llm/__init__.py:
   - Export BaseLLM

3. tests/llm/test_base.py:
   - Test that BaseLLM cannot be instantiated
   - Test that subclass must implement all methods

Example:
```python
from src.llm import BaseLLM

class MyLLM(BaseLLM):
    @property
    def model_name(self) -> str:
        return "my-model"
    
    def generate(self, prompt: str, **kwargs) -> str:
        return "Generated response"
    
    # ... other methods
```

Run: pytest tests/llm/test_base.py -v
All tests should pass.
```

---

### Prompt 8.2: Bedrock Claude

```
Implement LLM provider using AWS Bedrock Claude.

Create the following files:

1. src/llm/bedrock.py:
   - BedrockLLM class extending BaseLLM
   - Uses boto3 Bedrock client
   - Support Claude 3 models
   - Handle streaming (optional)
   - Implement retry logic

2. tests/llm/test_bedrock.py:
   - Mock boto3 client for unit tests
   - Test generate method
   - Test token counting
   - Test error handling

Example:
```python
from src.llm.bedrock import BedrockLLM

llm = BedrockLLM(model="anthropic.claude-3-sonnet")
response = llm.generate("What is federalism?")
```

Run: pytest tests/llm/test_bedrock.py -v
All tests should pass.
```

---

### Prompt 8.3: OpenAI GPT-4

```
Implement LLM provider using OpenAI API.

Create the following files:

1. src/llm/openai_llm.py:
   - OpenAILLM class extending BaseLLM
   - Uses openai library
   - Support GPT-4 and GPT-3.5-turbo
   - Handle streaming (optional)
   - Implement retry logic

2. tests/llm/test_openai.py:
   - Mock OpenAI client for unit tests
   - Test generate method
   - Test token counting
   - Test error handling

Example:
```python
from src.llm.openai_llm import OpenAILLM

llm = OpenAILLM(model="gpt-4-turbo-preview")
response = llm.generate("What is federalism?")
```

Run: pytest tests/llm/test_openai.py -v
All tests should pass.
```

---

### Prompt 8.4: Local Ollama

```
Implement LLM provider using local Ollama.

Create the following files:

1. src/llm/ollama.py:
   - OllamaLLM class extending BaseLLM
   - Uses ollama library or HTTP API
   - Support various local models
   - Handle connection errors

2. tests/llm/test_ollama.py:
   - Mock Ollama for unit tests
   - Test generate method
   - Test connection error handling

Example:
```python
from src.llm.ollama import OllamaLLM

llm = OllamaLLM(model="llama2")
response = llm.generate("What is federalism?")
```

Run: pytest tests/llm/test_ollama.py -v
All tests should pass.
```

---

### Prompt 8.5: LLM Factory

```
Create factory for LLM instantiation.

Create the following files:

1. src/llm/factory.py:
   - LLMFactory class
   - get_llm(config: LLMConfig) -> BaseLLM
   - Support providers: "bedrock_claude", "openai", "ollama"

2. src/llm/__init__.py (update):
   - Export all LLM classes and factory

3. tests/llm/test_factory.py:
   - Test factory returns correct type
   - Test factory with different configs

Example:
```python
from src.llm import LLMFactory
from src.config import LLMConfig

config = LLMConfig(provider="openai", model_name="gpt-4-turbo")
llm = LLMFactory.get_llm(config)
```

Run: pytest tests/llm/ -v
All tests should pass.
```

---

## CHUNK 9: FastAPI Application

### Prompt 9.1: FastAPI Setup

```
Create the FastAPI application with basic configuration.

Create the following files:

1. src/app/main.py:
   - FastAPI app instance
   - CORS middleware
   - Exception handlers
   - OpenAPI configuration
   - Lifespan context manager for initialization

2. src/app/__init__.py:
   - Export app

3. tests/app/test_main.py:
   - Test app creation
   - Test CORS configuration
   - Test OpenAPI docs are available

Example:
```python
from src.app import app

# Run with: uvicorn src.app.main:app --reload
```

Run: pytest tests/app/test_main.py -v
All tests should pass.
```

---

### Prompt 9.2: Query Endpoint

```
Implement the main query endpoint.

Create the following files:

1. src/app/schemas.py:
   - QueryRequest model (question, mode, k)
   - QueryResponse model (answer, citations, confidence, refusal_reason, latency_ms)

2. src/app/routes/query.py:
   - POST /query endpoint
   - Wire to LangGraph workflow
   - Handle errors gracefully
   - Return structured response

3. src/app/main.py (update):
   - Include query router

4. tests/app/test_query.py:
   - Test query endpoint with valid request
   - Test query endpoint with invalid request
   - Test error handling
   - Test response format

Example:
```python
# POST /query
{
    "question": "What is federalism?",
    "mode": "qna",
    "k": 5
}

# Response
{
    "answer": "Federalism is a system of government...",
    "citations": [...],
    "confidence": 0.92,
    "refusal_reason": null,
    "latency_ms": 1234
}
```

Run: pytest tests/app/test_query.py -v
All tests should pass.
```

---

### Prompt 9.3: Health Endpoint

```
Implement health check endpoint.

Create the following files:

1. src/app/routes/health.py:
   - GET /health endpoint
   - Check component health (embeddings, vector store, LLM)
   - Return detailed status

2. src/app/main.py (update):
   - Include health router

3. tests/app/test_health.py:
   - Test health endpoint returns 200
   - Test health check with unhealthy component

Example:
```python
# GET /health
# Response
{
    "status": "healthy",
    "components": {
        "embeddings": "ok",
        "vector_store": "ok",
        "llm": "ok"
    },
    "version": "1.0.0"
}
```

Run: pytest tests/app/test_health.py -v
All tests should pass.
```

---

### Prompt 9.4: Ingestion Endpoint

```
Implement ingestion endpoint for triggering document ingestion.

Create the following files:

1. src/app/routes/ingest.py:
   - POST /ingest endpoint
   - Trigger ingestion pipeline
   - Return ingestion status
   - Support async ingestion (optional)

2. src/app/main.py (update):
   - Include ingest router

3. tests/app/test_ingest.py:
   - Test ingest endpoint
   - Test ingestion status
   - Test error handling

Example:
```python
# POST /ingest
{
    "corpus": "public_domain_gov"
}

# Response
{
    "status": "completed",
    "documents_count": 42,
    "chunks_count": 1250,
    "errors": []
}
```

Run: pytest tests/app/test_ingest.py -v
All tests should pass.
```

---

### Prompt 9.5: Error Handling

```
Implement comprehensive error handling.

Create the following files:

1. src/app/exceptions.py:
   - HTTPException subclasses
   - Error response models

2. src/app/middleware.py:
   - Error handling middleware
   - Request logging middleware
   - Correlation ID middleware

3. src/app/main.py (update):
   - Add middleware

4. tests/app/test_errors.py:
   - Test various error scenarios
   - Test error response format
   - Test correlation ID propagation

Example error response:
```json
{
    "error": {
        "code": "RETRIEVAL_ERROR",
        "message": "Failed to retrieve documents",
        "details": {...}
    },
    "correlation_id": "abc-123"
}
```

Run: pytest tests/app/ -v
All tests should pass.
```

---

## CHUNK 10: Evaluation Harness

### Prompt 10.1: Eval Dataset Format

```
Define the evaluation dataset format.

Create the following files:

1. src/eval/dataset.py:
   - EvalDataset model
   - EvalQuestion model (question, expected_answer, expected_citations, expected_refusal)
   - Dataset loader from JSON/YAML

2. src/eval/__init__.py:
   - Export dataset classes

3. tests/eval/test_dataset.py:
   - Test dataset loading
   - Test dataset validation

4. src/eval/datasets/sample_qna.yaml:
   - Sample evaluation dataset with 10 questions

Example dataset:
```yaml
name: sample_qna
questions:
  - question: "What is federalism?"
    expected_answer_contains: ["system of government", "power"]
    expected_citations_min: 1
    expected_refusal: false
  - question: "What is the capital of France?"
    expected_refusal: true
    refusal_reason: "not in corpus"
```

Run: pytest tests/eval/test_dataset.py -v
All tests should pass.
```

---

### Prompt 10.2: Groundedness Scorer

```
Implement groundedness scoring.

Create the following files:

1. src/eval/scorers/groundedness.py:
   - GroundednessScorer class
   - score(answer: str, chunks: list[Chunk]) -> float
   - Check if answer claims are supported by chunks
   - Return score 0.0-1.0

2. tests/eval/scorers/test_groundedness.py:
   - Test scoring with grounded answer
   - Test scoring with ungrounded answer
   - Test scoring with partial grounding

Example:
```python
from src.eval.scorers import GroundednessScorer

scorer = GroundednessScorer()
score = scorer.score("Federalism divides power.", chunks)
# Returns 0.95 if chunks support this
```

Run: pytest tests/eval/scorers/test_groundedness.py -v
All tests should pass.
```

---

### Prompt 10.3: Relevance Scorer

```
Implement relevance scoring.

Create the following files:

1. src/eval/scorers/relevance.py:
   - RelevanceScorer class
   - score(answer: str, question: str) -> float
   - Check if answer addresses the question
   - Return score 0.0-1.0

2. tests/eval/scorers/test_relevance.py:
   - Test scoring with relevant answer
   - Test scoring with irrelevant answer
   - Test scoring with partial relevance

Example:
```python
from src.eval.scorers import RelevanceScorer

scorer = RelevanceScorer()
score = scorer.score("Federalism is...", "What is federalism?")
# Returns 0.90 if answer is relevant
```

Run: pytest tests/eval/scorers/test_relevance.py -v
All tests should pass.
```

---

### Prompt 10.4: Refusal Correctness

```
Implement refusal correctness checking.

Create the following files:

1. src/eval/scorers/refusal.py:
   - RefusalScorer class
   - score(result: QueryResult, expected: EvalQuestion) -> bool
   - Check if refusal was appropriate

2. tests/eval/scorers/test_refusal.py:
   - Test correct refusal
   - Test incorrect refusal (should have answered)
   - Test incorrect answer (should have refused)

Example:
```python
from src.eval.scorers import RefusalScorer

scorer = RefusalScorer()
is_correct = scorer.score(result, expected)
# Returns True if refusal was appropriate
```

Run: pytest tests/eval/scorers/test_refusal.py -v
All tests should pass.
```

---

### Prompt 10.5: Evaluation Runner and Reports

```
Create the evaluation runner and report generator.

Create the following files:

1. src/eval/runner.py:
   - EvalRunner class
   - run(dataset: EvalDataset, graph: StateGraph) -> EvalReport
   - Run all questions through graph
   - Collect scores

2. src/eval/report.py:
   - EvalReport model
   - Aggregate metrics
   - Per-question results
   - to_json(), to_html() methods

3. scripts/eval.py:
   - CLI script to run evaluation
   - Arguments: --suite, --output
   - Print summary and save report

4. tests/eval/test_runner.py:
   - Test evaluation runner
   - Test report generation

Example:
```bash
python scripts/eval.py --suite sample_qna --output reports/
```

Run: pytest tests/eval/ -v
All tests should pass.
```

---

## CHUNK 11: Integration & Polish

### Prompt 11.1: CLI Scripts

```
Finalize CLI scripts for ingestion and evaluation.

Update/Create the following files:

1. scripts/ingest.py (update):
   - Full implementation
   - Progress bar support
   - Error handling
   - Summary output

2. scripts/eval.py (update):
   - Full implementation
   - Multiple output formats
   - Comparison with baseline

3. scripts/query.py (new):
   - Interactive query CLI
   - Debug mode with intermediate steps

4. tests/test_scripts.py:
   - Test CLI scripts work correctly

Example:
```bash
python scripts/ingest.py --corpus public_domain_gov
python scripts/query.py --question "What is federalism?"
python scripts/eval.py --suite sample_qna
```

Run: pytest tests/test_scripts.py -v
All tests should pass.
```

---

### Prompt 11.2: Docker Setup

```
Create Docker configuration for deployment.

Create the following files:

1. Dockerfile:
   - Multi-stage build
   - Python 3.11 base
   - Install dependencies
   - Copy application code
   - Set up entry point

2. docker-compose.yml:
   - API service
   - Optional: local LLM service

3. .dockerignore:
   - Exclude unnecessary files

4. tests/test_docker.py:
   - Test Docker build (optional, may be slow)

Example:
```bash
docker build -t graphrag-tutor .
docker run -p 8000:8000 graphrag-tutor
```

Run: docker build -t graphrag-tutor . (manual test)
```

---

### Prompt 11.3: Documentation

```
Create comprehensive documentation.

Create the following files:

1. README.md (update):
   - Full installation instructions
   - Configuration guide
   - API documentation
   - Usage examples

2. docs/API.md:
   - Detailed API documentation
   - Request/response examples
   - Error codes

3. docs/CONFIGURATION.md:
   - All configuration options
   - Environment variables
   - Provider setup

4. docs/DEPLOYMENT.md:
   - Docker deployment
   - Cloud deployment options
   - Scaling considerations

5. docs/EVALUATION.md:
   - How to run evaluations
   - Interpreting results
   - Creating custom datasets

No tests for documentation, but verify links and examples work.
```

---

### Prompt 11.4: End-to-End Testing

```
Create comprehensive end-to-end tests.

Create the following files:

1. tests/e2e/test_full_pipeline.py:
   - Test full ingestion to query flow
   - Test with sample documents
   - Verify answer quality

2. tests/e2e/test_api.py:
   - Test API endpoints with real components
   - Test error scenarios
   - Test concurrent requests

3. tests/e2e/test_eval.py:
   - Run evaluation and verify scores meet thresholds

4. pytest.ini:
   - Configure test markers
   - Configure slow tests
   - Configure e2e tests

Example:
```bash
pytest tests/e2e/ -v --run-e2e
```

Run: pytest tests/e2e/ -v --run-e2e
All tests should pass.
```

---

### Prompt 11.5: Final Integration

```
Final integration and polish.

Update/Create the following files:

1. src/__init__.py (update):
   - Export main classes
   - Version info

2. src/app/main.py (update):
   - Wire all components together
   - Initialize on startup
   - Graceful shutdown

3. configs/default.yaml (update):
   - Production-ready defaults
   - Comments explaining options

4. Add sample corpus:
   - data/sample_docs/
   - A few sample documents for testing

5. Final test run:
   - Run all tests
   - Verify coverage
   - Fix any issues

Run: pytest tests/ -v --cov=src
All tests should pass with >80% coverage.
```

---

## Summary

This blueprint provides 45 incremental prompts that build the Grounded GraphRAG Tutor from scratch. Each prompt:

1. **Builds on previous work** - No orphaned code
2. **Is testable** - Clear success criteria
3. **Is appropriately sized** - Not too big, not too small
4. **Follows best practices** - Type hints, docstrings, error handling
5. **Ends with integration** - Each chunk wires components together

### Execution Order

Execute prompts in order: 1.1 → 1.2 → ... → 11.5

Each prompt should be given to an LLM with the context of previous work completed. The LLM should implement the code, write tests, and verify all tests pass before moving to the next prompt.
