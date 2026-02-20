# ADR-001: Core Architecture Decisions

**Status:** Accepted
**Date:** 2026-02-16

## Context

The Grounded GraphRAG Tutor is a retrieval-augmented generation system designed to answer questions grounded in a curated document corpus. The system must produce verifiably sourced answers, refuse when it cannot, and support future extension toward multi-turn tutoring workflows. This ADR records the rationale behind four foundational design choices.

## Decisions

### 1. LangGraph over simple chains

A naive RAG pipeline (retrieve → generate) can be implemented as a linear chain. We chose LangGraph's `StateGraph` instead because:

- **Conditional branching.** The workflow requires routing decisions (supported vs. unsupported queries) and a retry loop (low-confidence answers loop back to retrieval). These are cyclic, stateful control flows that linear chains cannot express without awkward workarounds.
- **Explicit state.** Each node reads from and writes to a typed `GraphState` dict. This makes the data contract between nodes visible and testable, rather than buried in prompt-chaining glue code.
- **Extensibility.** Adding a tutoring workflow (Socratic follow-ups, hint scaffolding, learner state tracking) means adding nodes and edges to the graph, not rewriting a monolithic chain. The config already reserves a `multi_turn` graph type for this.
- **Observability.** LangGraph provides built-in Mermaid visualization and step-level tracing, which simplifies debugging a multi-step workflow with conditional paths.

### 2. FAISS and Chroma as vector store backends

We support two vector stores with different trade-off profiles:

- **FAISS** is the default. It runs in-process with no external dependencies, making it ideal for local development, single-replica deployments, and CI/testing. It is fast, well-understood, and adds zero operational overhead.
- **Chroma** is the alternative for scenarios requiring persistence across processes or shared access from multiple replicas. It can run as a standalone server with a simple HTTP API.
- **Why not a managed cloud service?** Services like Pinecone, Weaviate, or OpenSearch Serverless add network latency, cost, vendor lock-in, and infrastructure setup. For an educational tool operating on a curated (not web-scale) corpus, the document counts are modest enough that in-process or single-server stores are sufficient. The `VectorStoreFactory` pattern makes it straightforward to add a cloud backend later without changing any calling code.

### 3. Verify/refuse loop

The workflow does not trust the LLM's output at face value. After answer generation, a grounding verification step checks whether the answer is supported by the retrieved source chunks:

- **Why verify?** LLMs hallucinate. In an educational context, presenting fabricated information as fact is worse than admitting uncertainty. The verify node scores each sentence in the answer against the retrieved chunks and produces an aggregate confidence score.
- **Why a retry loop?** A single retrieval pass may miss relevant chunks due to embedding similarity thresholds or chunking boundaries. If confidence is below the threshold but retries remain, the system loops back to retrieval, giving it a second chance before giving up. This improves answer rates without sacrificing groundedness.
- **Why refuse?** If the answer remains ungrounded after exhausting retries, the system explicitly refuses rather than returning a low-confidence answer. This is a deliberate design choice: for a tutoring system, "I don't have reliable information on that" is preferable to a confident-sounding wrong answer. The refusal includes a reason so the user understands why.

### 4. Content-agnostic connectors

Document ingestion is decoupled from the rest of the pipeline through a `BaseConnector` abstraction:

- **Separation of concerns.** The ingestion pipeline (cleaning, chunking, embedding) does not know or care whether documents came from a local directory, an S3 bucket, or a web scrape. This keeps the chunker and embedder testable in isolation.
- **Practical flexibility.** Educational content lives in many places: local files during development, cloud storage in production, web sources for supplementary material. The connector interface (`validate_source`, `list_documents`, `load`) is minimal enough to implement for any source without forcing a lowest-common-denominator API.
- **Format handling stays local.** Each connector owns the file-format-specific extraction logic (PDF via pypdf, HTML tag stripping, encoding detection for plain text). Downstream components receive clean `Document` objects with uniform string content regardless of the original format.

## Consequences

- The LangGraph dependency is heavier than a simple function pipeline, but it pays for itself as soon as the workflow has more than one branching point.
- Supporting two vector stores means two code paths to maintain, but the factory pattern and shared `BaseVectorStore` interface keep the duplication minimal.
- The verify/refuse loop adds latency (potentially 2-3x for retried queries), which is acceptable for an educational tool where correctness matters more than speed.
- The connector abstraction requires implementing a small interface for each new source, but this is a one-time cost per source type.
