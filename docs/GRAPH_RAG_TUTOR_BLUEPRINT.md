# Grounded GraphRAG Tutor - Detailed Build Blueprint

## Project Overview

This blueprint provides a comprehensive, step-by-step implementation plan for building a content-agnostic RAG service with LangGraph orchestration. The system answers questions only when supported by sources, returns citations, and refuses gracefully when evidence is insufficient.

---

## Architecture Deep Dive

### System Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API LAYER                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  FastAPI Endpoints: /query, /health, /ingest, /eval                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LANGGRAPH WORKFLOW                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────────┐         │
│  │  route   │──▶│ retrieve │──▶│  answer  │──▶│ verify_grounding │         │
│  └──────────┘   └──────────┘   └──────────┘   └──────────────────┘         │
│                                       │              │                      │
│                                       │              ▼                      │
│                                       │    ┌──────────────────┐            │
│                                       │    │ retry / refuse   │            │
│                                       │    └──────────────────┘            │
│                                       │              │                      │
│                                       ▼              ▼                      │
│                                ┌───────────────────────┐                   │
│                                │        END            │                   │
│                                └───────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SERVICE LAYER                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Connectors   │  │  Embeddings  │  │ Vector Store │  │  Retrieval   │    │
│  │ (doc load)   │  │  (vectors)   │  │  (index)     │  │  (search)    │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PROVIDER LAYER                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Bedrock      │  │   OpenAI     │  │   Local      │  │   FAISS/     │    │
│  │ (Claude)     │  │   (GPT-4)    │  │   (Ollama)   │  │   Chroma     │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Document Source → Connector → Ingestion Pipeline → Chunks → Embeddings → Vector Store
                                                                              │
Query → LangGraph → Retrieval → Vector Store → Top-K Chunks → LLM → Answer → Verification
                                                                              │
                                                                    Citations + Confidence
```

---

## Iterative Build Chunks

### Chunk 1: Foundation & Project Structure
**Goal**: Establish project skeleton with configuration management

**Deliverables**:
- Project directory structure
- Configuration system (YAML + env vars)
- Logging infrastructure
- Type definitions and interfaces

### Chunk 2: Document Connectors
**Goal**: Load documents from various sources

**Deliverables**:
- Base connector interface
- Local file connector
- S3 connector (stub)
- Web connector (stub)
- Document model

### Chunk 3: Ingestion Pipeline
**Goal**: Clean, chunk, and prepare documents

**Deliverables**:
- Text cleaning utilities
- Chunking strategies (fixed, semantic)
- Metadata extraction
- Ingestion orchestrator

### Chunk 4: Embedding Providers
**Goal**: Generate vector embeddings for text

**Deliverables**:
- Base embedding interface
- Bedrock Titan embeddings
- OpenAI embeddings
- Local sentence-transformers
- Embedding cache

### Chunk 5: Vector Store
**Goal**: Store and retrieve embeddings

**Deliverables**:
- Base vector store interface
- FAISS implementation
- Chroma implementation
- Index persistence
- Metadata filtering

### Chunk 6: Retrieval Service
**Goal**: Search and retrieve relevant chunks

**Deliverables**:
- Similarity search
- Top-k retrieval
- Citation extraction
- Optional reranking

### Chunk 7: LangGraph Q&A Workflow
**Goal**: Orchestrate RAG pipeline with state machine

**Deliverables**:
- Graph state definition
- Route node
- Retrieve node
- Answer node
- Verify grounding node
- Retry/refuse logic

### Chunk 8: LLM Providers
**Goal**: Generate answers with various LLMs

**Deliverables**:
- Base LLM interface
- Bedrock Claude
- OpenAI GPT-4
- Local Ollama
- Prompt templates

### Chunk 9: FastAPI Application
**Goal**: Expose functionality via HTTP API

**Deliverables**:
- FastAPI app setup
- Query endpoint
- Health endpoint
- Ingestion endpoint
- Error handling

### Chunk 10: Evaluation Harness
**Goal**: Measure system quality

**Deliverables**:
- Eval dataset format
- Groundedness scorer
- Relevance scorer
- Refusal correctness
- Report generation

### Chunk 11: Integration & Polish
**Goal**: Wire everything together

**Deliverables**:
- End-to-end integration
- CLI scripts
- Docker setup
- Documentation

---

## Detailed Implementation Steps

### CHUNK 1: Foundation & Project Structure

#### Step 1.1: Create Project Skeleton
- Create directory structure per README layout
- Initialize pyproject.toml with dependencies
- Create .env.example template
- Add .gitignore

#### Step 1.2: Configuration System
- Create config loader (YAML-based)
- Support environment variable overrides
- Create settings classes with validation
- Add config for: corpus, vectorstore, embeddings, llm, graph

#### Step 1.3: Logging Infrastructure
- Configure structured logging
- Add correlation IDs for request tracing
- Create log formatters
- Add log level configuration

#### Step 1.4: Type Definitions
- Create base types: Document, Chunk, Citation, Answer
- Create result types: QueryResult, IngestResult
- Create provider interfaces (protocols)
- Add Pydantic models for validation

---

### CHUNK 2: Document Connectors

#### Step 2.1: Base Connector Interface
- Create abstract base class for connectors
- Define load() method signature
- Define list_documents() method
- Add error handling base classes

#### Step 2.2: Document Model
- Create Document dataclass
- Add metadata fields
- Add content field
- Add source tracking

#### Step 2.3: Local File Connector
- Implement local directory scanning
- Support PDF, TXT, MD, HTML
- Handle encoding detection
- Add file filtering

#### Step 2.4: Connector Factory
- Create factory for connector selection
- Add connector registration
- Support config-based instantiation

---

### CHUNK 3: Ingestion Pipeline

#### Step 3.1: Text Cleaning
- Create text normalizer
- Remove boilerplate (HTML)
- Handle encoding issues
- Preserve structure markers

#### Step 3.2: Chunking Strategies
- Create base chunker interface
- Implement fixed-size chunking
- Implement sentence-based chunking
- Add overlap support

#### Step 3.3: Metadata Extraction
- Extract source metadata
- Add position tracking
- Create chunk IDs
- Preserve document hierarchy

#### Step 3.4: Ingestion Orchestrator
- Create ingestion pipeline class
- Wire connector → cleaner → chunker
- Add progress tracking
- Handle errors gracefully

---

### CHUNK 4: Embedding Providers

#### Step 4.1: Base Embedding Interface
- Create abstract embedding class
- Define embed_documents() method
- Define embed_query() method
- Add dimension tracking

#### Step 4.2: Bedrock Titan Embeddings
- Implement Bedrock client wrapper
- Add Titan embedding support
- Handle rate limiting
- Add retry logic

#### Step 4.3: OpenAI Embeddings
- Implement OpenAI client wrapper
- Add text-embedding-3 support
- Handle rate limiting
- Add retry logic

#### Step 4.4: Local Sentence Transformers
- Implement local embedding model
- Add model caching
- Support multiple models
- Add GPU detection

#### Step 4.5: Embedding Cache
- Create embedding cache interface
- Implement disk-based cache
- Add cache invalidation
- Support cache warming

---

### CHUNK 5: Vector Store

#### Step 5.1: Base Vector Store Interface
- Create abstract vector store class
- Define add_embeddings() method
- Define similarity_search() method
- Define delete() method

#### Step 5.2: FAISS Implementation
- Implement FAISS index wrapper
- Add metadata storage
- Support index persistence
- Handle index updates

#### Step 5.3: Chroma Implementation
- Implement Chroma client wrapper
- Add collection management
- Support filtering
- Handle persistence

#### Step 5.4: Vector Store Factory
- Create factory for store selection
- Support config-based instantiation
- Add health checks

---

### CHUNK 6: Retrieval Service

#### Step 6.1: Similarity Search
- Create retrieval service class
- Implement top-k search
- Add score normalization
- Handle empty results

#### Step 6.2: Citation Extraction
- Create citation builder
- Extract source references
- Format citations consistently
- Add confidence scores

#### Step 6.3: Optional Reranking
- Create reranker interface
- Implement cross-encoder reranking
- Add reranking toggle
- Measure reranking impact

---

### CHUNK 7: LangGraph Q&A Workflow

#### Step 7.1: Graph State Definition
- Create state TypedDict
- Define state fields: question, chunks, answer, citations, etc.
- Add state validators
- Create state initializers

#### Step 7.2: Route Node
- Implement query router
- Classify query type
- Route to appropriate path
- Handle edge cases

#### Step 7.3: Retrieve Node
- Implement retrieval node
- Call retrieval service
- Update state with chunks
- Handle retrieval failures

#### Step 7.4: Answer Node
- Implement answer generation
- Build prompt with context
- Call LLM provider
- Parse LLM response

#### Step 7.5: Verify Grounding Node
- Implement grounding checker
- Check citation support
- Calculate confidence
- Flag unsupported claims

#### Step 7.6: Retry/Refuse Logic
- Implement retry counter
- Add retry decision logic
- Implement refusal response
- Add refusal reasons

#### Step 7.7: Graph Assembly
- Create LangGraph graph
- Wire nodes together
- Add conditional edges
- Compile and test graph

---

### CHUNK 8: LLM Providers

#### Step 8.1: Base LLM Interface
- Create abstract LLM class
- Define generate() method
- Define stream() method
- Add token counting

#### Step 8.2: Bedrock Claude
- Implement Bedrock client
- Add Claude model support
- Handle streaming
- Add retry logic

#### Step 8.3: OpenAI GPT-4
- Implement OpenAI client
- Add GPT-4 support
- Handle streaming
- Add retry logic

#### Step 8.4: Local Ollama
- Implement Ollama client
- Add local model support
- Handle streaming
- Add health checks

#### Step 8.5: Prompt Templates
- Create prompt template system
- Add RAG prompt template
- Add refusal prompt template
- Support template variables

---

### CHUNK 9: FastAPI Application

#### Step 9.1: FastAPI Setup
- Create FastAPI app instance
- Add CORS middleware
- Add exception handlers
- Configure OpenAPI docs

#### Step 9.2: Query Endpoint
- Create /query POST endpoint
- Define request/response models
- Wire to LangGraph workflow
- Add request validation

#### Step 9.3: Health Endpoint
- Create /health GET endpoint
- Check component health
- Return status details
- Add dependency checks

#### Step 9.4: Ingestion Endpoint
- Create /ingest POST endpoint
- Trigger ingestion pipeline
- Return ingestion status
- Add async support

#### Step 9.5: Error Handling
- Create custom exceptions
- Add error response models
- Implement error logging
- Add error recovery

---

### CHUNK 10: Evaluation Harness

#### Step 10.1: Eval Dataset Format
- Create dataset schema
- Define question-answer pairs
- Add expected citations
- Support multiple corpora

#### Step 10.2: Groundedness Scorer
- Implement grounding checker
- Compare answer to sources
- Calculate groundedness score
- Add detailed breakdown

#### Step 10.3: Relevance Scorer
- Implement relevance checker
- Compare answer to question
- Calculate relevance score
- Add detailed breakdown

#### Step 10.4: Refusal Correctness
- Implement refusal checker
- Verify refusal appropriateness
- Calculate refusal accuracy
- Add edge case handling

#### Step 10.5: Report Generation
- Create report generator
- Aggregate metrics
- Generate JSON/HTML reports
- Add visualization

---

### CHUNK 11: Integration & Polish

#### Step 11.1: CLI Scripts
- Create ingest.py script
- Create eval.py script
- Add argument parsing
- Add progress output

#### Step 11.2: Docker Setup
- Create Dockerfile
- Create docker-compose.yml
- Add environment configuration
- Test container deployment

#### Step 11.3: Documentation
- Write API documentation
- Write configuration guide
- Write deployment guide
- Add usage examples

#### Step 11.4: End-to-End Testing
- Create integration tests
- Test full pipeline
- Test error scenarios
- Add performance tests

---

## Dependency Graph

```
Chunk 1 (Foundation)
    │
    ├──▶ Chunk 2 (Connectors)
    │        │
    │        └──▶ Chunk 3 (Ingestion)
    │                  │
    ├──▶ Chunk 4 (Embeddings) ──┐
    │                           │
    └──▶ Chunk 5 (Vector Store) │
              │                 │
              └──▶ Chunk 6 (Retrieval)
                        │
                        └──▶ Chunk 7 (LangGraph) ──▶ Chunk 8 (LLMs)
                                  │
                                  └──▶ Chunk 9 (FastAPI)
                                            │
                                            └──▶ Chunk 10 (Eval)
                                                      │
                                                      └──▶ Chunk 11 (Integration)
```

---

## Testing Strategy

### Unit Tests
- Each component has isolated unit tests
- Mock external dependencies
- Test edge cases and error handling

### Integration Tests
- Test component interactions
- Test with real embeddings (small model)
- Test with mock LLM responses

### End-to-End Tests
- Full pipeline tests
- Real document ingestion
- Query and verify responses

### Evaluation Tests
- Groundedness verification
- Answer quality metrics
- Refusal correctness

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| LLM API costs | Use local models for development, cache responses |
| Embedding latency | Batch embeddings, use cache |
| Vector store memory | Use disk-based indexes, implement pagination |
| Hallucination | Strict grounding verification, retry logic |
| Poor retrieval | Multiple retrieval strategies, reranking |

---

## Success Criteria

### Per-Chunk Success Criteria
- All tests pass
- Code coverage > 80%
- Documentation updated
- No linting errors

### Final Success Criteria
- Query endpoint returns grounded answers
- Citations are accurate
- Refusals are appropriate
- Eval scores meet thresholds:
  - Groundedness > 90%
  - Relevance > 85%
  - Refusal correctness > 95%
