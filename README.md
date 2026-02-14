# Grounded GraphRAG Tutor

A content-agnostic RAG service with LangGraph orchestration that answers questions only when supported by sources, returns citations, and refuses gracefully when evidence is insufficient.

## Overview

This project implements a Retrieval-Augmented Generation (RAG) service with the following key features:

- **Grounded Responses**: Answers are only generated when supported by retrieved source documents
- **Citation Support**: All answers include citations pointing to source documents
- **Graceful Refusal**: The system refuses to answer when evidence is insufficient
- **LangGraph Orchestration**: Uses LangGraph for workflow state management
- **Multi-Provider Support**: Works with OpenAI, AWS Bedrock, and local LLMs

## Architecture

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

## Project Structure

```
grounded-graphrag-tutor/
├── src/
│   ├── connectors/     # Document source adapters (local, S3, web)
│   ├── ingestion/      # Document cleaning and chunking pipeline
│   ├── embeddings/     # Embedding providers (Bedrock, OpenAI, local)
│   ├── store/          # Vector store adapters (FAISS, Chroma)
│   ├── retrieval/      # Retrieval service with citation extraction
│   ├── graphs/         # LangGraph workflow definitions
│   ├── app/            # FastAPI API layer
│   └── eval/           # Evaluation datasets and runner
├── tests/              # Test suite (mirrors src structure)
├── configs/            # Configuration files
├── scripts/            # CLI scripts for ingestion and evaluation
├── docs/               # Documentation
├── pyproject.toml      # Project metadata and dependencies
├── .env.example        # Environment variable template
└── .gitignore          # Git ignore patterns
```

## Quick Start

### Prerequisites

- Python 3.10 or higher
- pip or uv package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/example/grounded-graphrag-tutor.git
   cd grounded-graphrag-tutor
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Copy environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

### Running the Service

Start the API server:
```bash
uvicorn src.app.main:app --reload
```

### Ingesting Documents

```bash
python scripts/ingest.py --source ./data
```

### Running Evaluation

```bash
python scripts/eval.py --dataset ./eval/datasets/default.json
```

## Configuration

Configuration is managed via YAML files and environment variables. See [`configs/default.yaml`](configs/default.yaml) for all available options.

Key configuration areas:

- **Corpus**: Document source configuration
- **Ingestion**: Chunking and cleaning settings
- **Embeddings**: Embedding provider and model selection
- **VectorStore**: Vector database configuration
- **LLM**: Language model provider settings
- **Graph**: LangGraph workflow parameters

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Submit a question for RAG processing |
| `/health` | GET | Check service health status |
| `/ingest` | POST | Trigger document ingestion |
| `/eval` | POST | Run evaluation harness |

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Style

This project follows Python best practices with type hints and docstrings.

## Success Criteria

The system aims to meet these quality thresholds:

- **Groundedness**: > 90%
- **Relevance**: > 85%
- **Refusal Correctness**: > 95%

## License

MIT License - see LICENSE file for details.
