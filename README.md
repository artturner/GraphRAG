# Grounded GraphRAG Tutor

A content-agnostic RAG service with LangGraph orchestration that answers questions only when supported by sources, returns citations, and refuses gracefully when evidence is insufficient.

## Key Features

- **Grounded Responses** -- answers are generated only when supported by retrieved source documents
- **Citation Support** -- every answer includes citations pointing to source chunks
- **Graceful Refusal** -- the system refuses to answer when evidence is insufficient
- **LangGraph Orchestration** -- a stateful retry/refuse loop with grounding verification
- **Multi-Provider Support** -- OpenAI, AWS Bedrock (Claude), Ollama, and local sentence-transformers
- **Evaluation Harness** -- built-in scorers for groundedness, relevance, and refusal correctness
- **Docker Ready** -- multi-stage Dockerfile and Compose file included

## Architecture

```
                           ┌──────────────────────────────────┐
                           │          FastAPI Layer            │
                           │  /query   /health   /ingest      │
                           └────────────────┬─────────────────┘
                                            │
                           ┌────────────────▼─────────────────┐
                           │      LangGraph Q&A Workflow       │
                           │                                   │
                           │  route ─► retrieve ─► answer      │
                           │              ▲          │         │
                           │              │          ▼         │
                           │           retry ◄── verify        │
                           │              │                    │
                           │              ▼                    │
                           │           refuse ─► END           │
                           └────────────────┬─────────────────┘
                                            │
          ┌─────────────┬───────────────────┼───────────────┬──────────┐
          │             │                   │               │          │
     Connectors    Embeddings         Vector Store     Retrieval      LLM
     (local/S3)    (local/OpenAI/     (FAISS/Chroma)   (search +    (OpenAI/
                    Bedrock)                            citations)   Bedrock/
                                                                    Ollama)
```

## Project Structure

```
grounded-graphrag-tutor/
├── src/
│   ├── app/            # FastAPI application, routes, middleware
│   ├── connectors/     # Document source adapters (local, S3, web)
│   ├── ingestion/      # Text cleaning and chunking pipeline
│   ├── embeddings/     # Embedding providers (local, OpenAI, Bedrock)
│   ├── store/          # Vector store adapters (FAISS, ChromaDB)
│   ├── retrieval/      # Retrieval service with reranking and citations
│   ├── llm/            # LLM providers (OpenAI, Bedrock, Ollama)
│   ├── graphs/         # LangGraph workflow (nodes, state, grounding)
│   ├── eval/           # Evaluation: datasets, scorers, runner, report
│   ├── config.py       # Pydantic-settings configuration
│   ├── types.py        # Shared Pydantic models
│   └── exceptions.py   # Exception hierarchy
├── tests/              # Test suite (mirrors src/ structure)
├── configs/            # YAML configuration files
├── scripts/            # CLI scripts (ingest, eval, query)
├── docs/               # Documentation
├── Dockerfile          # Multi-stage Docker build
├── docker-compose.yml  # Compose with optional Ollama service
├── pyproject.toml      # Project metadata and dependencies
└── .env.example        # Environment variable template
```

## Quick Start

### Prerequisites

- Python 3.10+
- An LLM API key (OpenAI, AWS Bedrock, or a local Ollama instance)

### Installation

```bash
# Clone
git clone https://github.com/artturner/GraphRAG.git
cd GraphRAG

# Virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# Install (with dev tools)
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env — at minimum set OPENAI_API_KEY or choose another provider
```

### Start the API Server

```bash
uvicorn src.app.main:app --reload
# API docs at http://localhost:8000/docs
```

### Ingest Documents

```bash
python scripts/ingest.py --corpus ./data
```

### Ask a Question

```bash
# Single question
python scripts/query.py --question "What is federalism?"

# Interactive REPL
python scripts/query.py --interactive

# With debug output
python scripts/query.py -q "What is federalism?" --debug
```

### Run via Docker

```bash
docker build -t graphrag-tutor .
docker run -p 8000:8000 --env-file .env graphrag-tutor
# Or with Compose:
docker compose up --build
```

### Run Evaluation

```bash
python scripts/eval.py --suite sample_qna --output reports/
```

## API Endpoints

| Endpoint                  | Method | Description                       |
|---------------------------|--------|-----------------------------------|
| `POST /query`             | POST   | Submit a question for processing  |
| `GET  /health`            | GET    | Check system health               |
| `POST /ingest`            | POST   | Trigger document ingestion        |
| `GET  /ingest/status/{c}` | GET    | Check async ingestion status      |

See [docs/API.md](docs/API.md) for full request/response schemas and examples.

## Configuration

Configuration uses three layers (later overrides earlier):

1. **Defaults** in code
2. **YAML file** (`configs/default.yaml`)
3. **Environment variables** (prefix pattern: `SECTION_KEY`)

Key sections: `corpus`, `embeddings`, `vectorstore`, `llm`, `graph`.

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for the complete reference.

## Evaluation

The built-in evaluation harness measures three metrics:

| Metric               | Target | Description                                      |
|----------------------|--------|--------------------------------------------------|
| **Groundedness**     | > 90%  | Is the answer supported by retrieved chunks?     |
| **Relevance**        | > 85%  | Does the answer address the original question?   |
| **Refusal Accuracy** | > 95%  | Does the system correctly refuse out-of-scope Qs?|

See [docs/EVALUATION.md](docs/EVALUATION.md) for details on running evals and creating custom datasets.

## Deployment

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for Docker, cloud, and scaling guidance.

## Development

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/eval/ -v
pytest tests/graphs/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### CLI Scripts

```bash
python scripts/ingest.py --help
python scripts/eval.py --help
python scripts/query.py --help
```

## License

MIT License -- see LICENSE file for details.
