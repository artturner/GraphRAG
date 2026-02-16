# API Reference

The Grounded GraphRAG Tutor exposes a REST API via FastAPI. Interactive documentation is available at `/docs` (Swagger UI) and `/redoc` (ReDoc) when the server is running.

## Base URL

```
http://localhost:8000
```

## Authentication

No authentication is required by default. Add an auth middleware or reverse proxy for production deployments.

## Common Headers

| Header              | Description                                      |
|---------------------|--------------------------------------------------|
| `Content-Type`      | `application/json` for all POST requests         |
| `X-Correlation-ID`  | Optional. If provided, echoed back in responses. If omitted, a UUID is generated automatically. |

---

## POST /query

Submit a question to the GraphRAG pipeline.

### Request Body

| Field      | Type   | Required | Default   | Description                                        |
|------------|--------|----------|-----------|----------------------------------------------------|
| `question` | string | yes      | --        | The question to ask (1--2000 characters)           |
| `mode`     | string | no       | `"qna"`   | Retrieval mode: `qna`, `vector`, or `hybrid`       |
| `k`        | int    | no       | `5`       | Number of chunks to retrieve (1--20)               |

### Example Request

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is federalism?", "mode": "qna", "k": 5}'
```

### Success Response (200)

```json
{
  "answer": "Federalism is a system of government in which power is divided between a national government and regional governments.",
  "citations": [
    {
      "source": "data/constitution.txt",
      "chunk_id": "chunk-042",
      "text": "Federalism divides power between national and state governments...",
      "score": 0.92
    }
  ],
  "confidence": 0.95,
  "refusal_reason": null,
  "latency_ms": 1250.5
}
```

### Refusal Response (200)

When the system cannot find sufficient evidence, it returns a refusal:

```json
{
  "answer": null,
  "citations": [],
  "confidence": 0.0,
  "refusal_reason": "Insufficient evidence in the corpus to answer this question.",
  "latency_ms": 320.1
}
```

### Error Responses

**400 Bad Request** -- invalid input:

```json
{
  "error": {
    "code": "BAD_REQUEST",
    "message": "Question cannot be empty",
    "details": null
  },
  "correlation_id": "abc-123-def"
}
```

**500 Internal Server Error** -- processing failure:

```json
{
  "error": {
    "code": "RETRIEVAL_ERROR",
    "message": "Failed to process query",
    "details": "Vector store connection refused"
  },
  "correlation_id": "abc-123-def"
}
```

---

## GET /health

Check the health of the system and its components.

### Example Request

```bash
curl http://localhost:8000/health
```

### Healthy Response (200)

```json
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

### Unhealthy Response (503)

```json
{
  "status": "unhealthy",
  "components": {
    "embeddings": "ok",
    "vector_store": "error",
    "llm": "ok"
  },
  "version": "1.0.0"
}
```

---

## POST /ingest

Trigger document ingestion from a named corpus.

### Request Body

| Field          | Type   | Required | Default | Description                                      |
|----------------|--------|----------|---------|--------------------------------------------------|
| `corpus`       | string | yes      | --      | Corpus name (1--100 characters)                  |
| `async_ingest` | bool   | no       | `false` | If `true`, return immediately and run in background |

### Example Request

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"corpus": "public_domain_gov", "async_ingest": false}'
```

### Success Response (200)

```json
{
  "status": "completed",
  "documents_count": 42,
  "chunks_count": 318,
  "errors": []
}
```

### Async Response (200)

When `async_ingest` is `true`:

```json
{
  "status": "in_progress",
  "documents_count": 0,
  "chunks_count": 0,
  "errors": []
}
```

---

## GET /ingest/status/{corpus}

Check the status of an async ingestion job.

### Path Parameters

| Parameter | Type   | Description        |
|-----------|--------|--------------------|
| `corpus`  | string | The corpus name    |

### Example Request

```bash
curl http://localhost:8000/ingest/status/public_domain_gov
```

### Success Response (200)

```json
{
  "status": "completed",
  "documents_count": 42,
  "chunks_count": 318,
  "errors": []
}
```

### Not Found (404)

```json
{
  "detail": "No ingestion found for corpus: unknown_corpus"
}
```

---

## Error Code Reference

All error responses follow this structure:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable description",
    "details": "Optional additional information"
  },
  "correlation_id": "request-uuid"
}
```

### Client Errors (4xx)

| Code               | HTTP | Description                        |
|--------------------|------|------------------------------------|
| `BAD_REQUEST`      | 400  | Invalid request parameters         |
| `UNAUTHORIZED`     | 401  | Missing or invalid credentials     |
| `FORBIDDEN`        | 403  | Insufficient permissions           |
| `NOT_FOUND`        | 404  | Resource not found                 |
| `VALIDATION_ERROR` | 422  | Request body validation failed     |

### Server Errors (5xx)

| Code                 | HTTP | Description                        |
|----------------------|------|------------------------------------|
| `INTERNAL_ERROR`     | 500  | Unspecified server error           |
| `RETRIEVAL_ERROR`    | 500  | Vector search or retrieval failed  |
| `INGESTION_ERROR`    | 500  | Document ingestion failed          |
| `EMBEDDING_ERROR`    | 500  | Embedding generation failed        |
| `VECTOR_STORE_ERROR` | 500  | Vector store operation failed      |
| `LLM_ERROR`          | 500  | Language model call failed         |
| `CONNECTOR_ERROR`    | 500  | Document source connector failed   |
| `CONFIGURATION_ERROR`| 500  | Invalid system configuration       |

---

## Middleware

The API applies the following middleware (outermost first):

1. **Error Handling** -- catches all exceptions, returns structured JSON errors
2. **Request Logging** -- logs method, path, status, and latency (excludes `/health`)
3. **Correlation ID** -- reads or generates `X-Correlation-ID`, adds to response headers
4. **CORS** -- allows all origins by default (restrict in production)

---

## OpenAPI / Swagger

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`
