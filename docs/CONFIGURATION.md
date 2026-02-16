# Configuration Reference

The Grounded GraphRAG Tutor uses a three-layer configuration system. Each layer overrides the previous:

1. **Code defaults** -- sensible values baked into Pydantic models
2. **YAML file** -- `configs/default.yaml` (or a custom path via `--config`)
3. **Environment variables** -- override any YAML or default value

## YAML Configuration

The default configuration file is `configs/default.yaml`:

```yaml
corpus:
  name: "default"
  path: "./data"
  connector_type: "local"

vectorstore:
  type: "faiss"
  persist_directory: "./.vectorstore"
  collection_name: "default"

embeddings:
  provider: "local"
  model_name: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384

llm:
  provider: "openai"
  model_name: "gpt-4"
  temperature: 0.0
  max_tokens: 1024

graph:
  type: "rag"
  max_retries: 2
  refusal_threshold: 0.8

debug: false
```

---

## Configuration Sections

### Corpus

| Key              | Env Variable             | Default     | Description                    |
|------------------|--------------------------|-------------|--------------------------------|
| `name`           | `CORPUS_NAME`            | `default`   | Corpus identifier              |
| `path`           | `CORPUS_PATH`            | `./data`    | Path to document directory     |
| `connector_type` | `CORPUS_CONNECTOR_TYPE`  | `local`     | `local`, `s3`, or `web`        |

### Vector Store

| Key                | Env Variable                    | Default          | Description                |
|--------------------|---------------------------------|------------------|----------------------------|
| `type`             | `VECTORSTORE_TYPE`              | `faiss`          | `faiss` or `chroma`        |
| `persist_directory`| `VECTORSTORE_PERSIST_DIRECTORY` | `./.vectorstore` | Directory for persistence  |
| `collection_name`  | `VECTORSTORE_COLLECTION_NAME`   | `default`        | ChromaDB collection name   |

### Embeddings

| Key          | Env Variable           | Default                                     | Description              |
|--------------|------------------------|---------------------------------------------|--------------------------|
| `provider`   | `EMBEDDINGS_PROVIDER`  | `local`                                     | See provider table below |
| `model_name` | `EMBEDDINGS_MODEL_NAME`| `sentence-transformers/all-MiniLM-L6-v2`    | Model identifier         |
| `dimension`  | `EMBEDDINGS_DIMENSION` | `384`                                       | Vector dimensionality    |

**Supported embedding providers:**

| Provider        | Model Examples                              | Dimension |
|-----------------|---------------------------------------------|-----------|
| `local`         | `sentence-transformers/all-MiniLM-L6-v2`   | 384       |
| `openai`        | `text-embedding-3-small`                    | 1536      |
| `openai`        | `text-embedding-3-large`                    | 3072      |
| `bedrock`       | `amazon.titan-embed-text-v1`                | 1536      |
| `bedrock_titan` | `amazon.titan-embed-text-v1`                | 1536      |

### LLM

| Key           | Env Variable      | Default   | Description                       |
|---------------|-------------------|-----------|-----------------------------------|
| `provider`    | `LLM_PROVIDER`    | `openai`  | See provider table below          |
| `model_name`  | `LLM_MODEL_NAME`  | `gpt-4`   | Model identifier                  |
| `temperature` | `LLM_TEMPERATURE` | `0.0`     | Sampling temperature (0.0--2.0)   |
| `max_tokens`  | `LLM_MAX_TOKENS`  | `1024`    | Maximum response tokens           |

**Supported LLM providers:**

| Provider         | Model Examples                                    | API Key Required      |
|------------------|---------------------------------------------------|-----------------------|
| `openai`         | `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo`         | `OPENAI_API_KEY`      |
| `bedrock`        | `anthropic.claude-3-sonnet-20240229-v1:0`         | AWS credentials       |
| `bedrock_claude` | Same as `bedrock`                                 | AWS credentials       |
| `ollama`         | `llama2`, `mistral`, `codellama`                  | None (local)          |

### Graph

| Key                 | Env Variable              | Default | Description                                 |
|---------------------|---------------------------|---------|---------------------------------------------|
| `type`              | `GRAPH_TYPE`              | `rag`   | `rag` or `multi_turn`                       |
| `max_retries`       | `GRAPH_MAX_RETRIES`       | `2`     | Retry attempts before refusing (0--5)       |
| `refusal_threshold` | `GRAPH_REFUSAL_THRESHOLD` | `0.8`   | Minimum confidence to accept answer (0--1)  |

### Global

| Key     | Env Variable | Default | Description                    |
|---------|--------------|---------|--------------------------------|
| `debug` | `DEBUG`      | `false` | Enable debug-level logging     |

---

## Environment Variables

All settings can be overridden with environment variables. The naming convention is `SECTION_KEY` in upper case.

### API Keys

```bash
# OpenAI (required when using openai provider)
OPENAI_API_KEY=sk-...

# AWS Bedrock (required when using bedrock provider)
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1

# Ollama (required when using ollama provider)
OLLAMA_BASE_URL=http://localhost:11434
```

### Quick Setup by Provider

**OpenAI (simplest)**:
```bash
LLM_PROVIDER=openai
LLM_MODEL_NAME=gpt-4
OPENAI_API_KEY=sk-...
EMBEDDINGS_PROVIDER=local       # use local embeddings to avoid OpenAI cost
```

**AWS Bedrock**:
```bash
LLM_PROVIDER=bedrock_claude
LLM_MODEL_NAME=anthropic.claude-3-sonnet-20240229-v1:0
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
EMBEDDINGS_PROVIDER=bedrock
EMBEDDINGS_MODEL_NAME=amazon.titan-embed-text-v1
EMBEDDINGS_DIMENSION=1536
```

**Ollama (fully local, no API keys)**:
```bash
LLM_PROVIDER=ollama
LLM_MODEL_NAME=llama2
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDINGS_PROVIDER=local
```

---

## Using a Custom Config File

```bash
# CLI scripts accept --config
python scripts/ingest.py --config configs/production.yaml
python scripts/query.py --config configs/production.yaml

# API server reads CONFIG_PATH environment variable
CONFIG_PATH=configs/production.yaml uvicorn src.app.main:app
```

---

## Programmatic Access

```python
from src.config import Settings, settings

# Use the global singleton
print(settings.llm.provider)      # "openai"
print(settings.embeddings.dimension)  # 384

# Or create a fresh instance with overrides
custom = Settings(config_path="configs/prod.yaml")
```
