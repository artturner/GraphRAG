# Deployment Guide

## Local Development

```bash
# Install
pip install -e ".[dev]"
cp .env.example .env           # edit with your API keys

# Start the server
uvicorn src.app.main:app --reload --port 8000
```

---

## Docker

### Build and Run

```bash
# Build the image
docker build -t graphrag-tutor .

# Run with environment file
docker run -p 8000:8000 --env-file .env graphrag-tutor

# Override a single setting
docker run -p 8000:8000 --env-file .env \
  -e LLM_PROVIDER=ollama \
  graphrag-tutor
```

### Docker Compose

```bash
# Start the API service
docker compose up --build

# Start with the optional local Ollama LLM
docker compose --profile ollama up --build

# Run in background
docker compose up -d
```

**Compose services:**

| Service  | Profile  | Port  | Description                          |
|----------|----------|-------|--------------------------------------|
| `api`    | default  | 8000  | FastAPI application                  |
| `ollama` | `ollama` | 11434 | Local Ollama LLM (optional)          |

### Volumes

The Compose file defines two named volumes:

- `vectorstore_data` -- persists the FAISS / Chroma index across restarts
- `ollama_data` -- persists downloaded Ollama models

Your `./data` directory is mounted read-only at `/app/data` and `./reports` is mounted at `/app/reports` for evaluation output.

### Health Check

Both the Dockerfile and Compose include a health check:

```bash
curl http://localhost:8000/health
```

The container is marked unhealthy if `/health` returns a non-200 status or times out after 5 seconds.

---

## Production Considerations

### Environment Variables

Never commit `.env` to version control. In production, inject secrets via your platform's secret manager:

- **AWS ECS/Fargate**: Task definition secrets
- **Kubernetes**: Secrets + ConfigMaps
- **Docker Swarm**: Docker secrets

### CORS

The default configuration allows all origins (`allow_origins=["*"]`). For production, restrict this in the middleware or via a reverse proxy.

### Reverse Proxy

Place NGINX, Caddy, or a cloud load balancer in front of the API:

```nginx
# NGINX example
upstream graphrag {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name graphrag.example.com;

    location / {
        proxy_pass http://graphrag;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Correlation-ID $request_id;
    }
}
```

### Workers

For production throughput, run multiple Uvicorn workers:

```bash
uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Or use Gunicorn as the process manager:

```bash
gunicorn src.app.main:app -w 4 -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

In Docker, override the CMD:

```bash
docker run -p 8000:8000 --env-file .env graphrag-tutor \
  gunicorn src.app.main:app -w 4 -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

---

## Cloud Deployment

### AWS ECS / Fargate

1. Push the image to ECR:
   ```bash
   aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
   docker tag graphrag-tutor:latest <account>.dkr.ecr.<region>.amazonaws.com/graphrag-tutor:latest
   docker push <account>.dkr.ecr.<region>.amazonaws.com/graphrag-tutor:latest
   ```

2. Create a task definition that references the image, sets environment variables from Secrets Manager, and exposes port 8000.

3. Create a Fargate service behind an ALB with a target group pointing to port 8000.

4. Use the `/health` endpoint for ALB health checks.

### Google Cloud Run

```bash
gcloud run deploy graphrag-tutor \
  --image <region>-docker.pkg.dev/<project>/graphrag/graphrag-tutor:latest \
  --port 8000 \
  --set-env-vars "LLM_PROVIDER=openai,EMBEDDINGS_PROVIDER=local" \
  --set-secrets "OPENAI_API_KEY=openai-key:latest" \
  --memory 2Gi \
  --cpu 2
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: graphrag-tutor
spec:
  replicas: 2
  selector:
    matchLabels:
      app: graphrag-tutor
  template:
    metadata:
      labels:
        app: graphrag-tutor
    spec:
      containers:
        - name: api
          image: graphrag-tutor:latest
          ports:
            - containerPort: 8000
          envFrom:
            - secretRef:
                name: graphrag-secrets
            - configMapRef:
                name: graphrag-config
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
          resources:
            requests:
              memory: "1Gi"
              cpu: "500m"
            limits:
              memory: "2Gi"
              cpu: "2"
```

---

## Scaling Considerations

### Vector Store

- **FAISS** is in-process and fast but not shared across replicas. Each replica loads its own index. Best for single-instance or read-only deployments.
- **ChromaDB** can run as a separate server, allowing multiple API replicas to share one index. Preferred for multi-replica setups.

### Embeddings

- **Local sentence-transformers** run on CPU and add latency (~50--200 ms per embed). If you need lower latency, switch to an API provider (OpenAI, Bedrock) or ensure GPU availability.
- **API-based embeddings** are faster per call but incur network latency and cost per token.

### LLM

- The LLM call is the bottleneck in most deployments (300--3000 ms per query).
- Use `max_tokens` to limit response length and reduce latency.
- The retry loop (`max_retries`) multiplies LLM calls -- keep it at 1--2 in latency-sensitive environments.

### Persistent Storage

- Mount a shared volume or object store for `.vectorstore/` in multi-replica setups.
- The `data/` corpus directory can be a read-only mount or S3 bucket.

### Resource Requirements

| Component             | CPU     | Memory  | Notes                           |
|-----------------------|---------|---------|---------------------------------|
| API (minimal)         | 0.5 CPU | 512 MB  | No local embeddings             |
| API (local embeddings)| 2 CPU   | 2 GB    | sentence-transformers model     |
| Ollama (local LLM)    | 4 CPU   | 8 GB    | Depends on model size           |
| ChromaDB server       | 1 CPU   | 1 GB    | Scales with collection size     |
