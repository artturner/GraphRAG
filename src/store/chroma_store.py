"""ChromaDB-based vector store implementation.

This module provides a vector store implementation using ChromaDB
for efficient similarity search with persistent storage and metadata filtering.
"""

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models import Collection
from chromadb.config import Settings

from src.exceptions import VectorStoreError
from src.store.base import BaseVectorStore, SearchResult
from src.types import Chunk, EmbeddingVector

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """Vector store implementation using ChromaDB for similarity search.
    
    This implementation uses ChromaDB for storing and retrieving embeddings
    with support for persistent storage, metadata filtering, and collection
    management.
    
    ChromaDB uses cosine similarity by default for similarity search.
    
    Attributes:
        collection_name: Name of the ChromaDB collection.
        persist_directory: Optional directory for persistent storage.
        distance_metric: Distance function to use (default: "cosine").
    
    Example:
        ```python
        store = ChromaVectorStore(
            collection_name="documents",
            persist_directory="./data/chroma"
        )
        store.add_embeddings(chunks, embeddings)
        results = store.similarity_search(query_embedding, k=5)
        for result in results:
            print(f"Score: {result.score}, Text: {result.chunk.content[:50]}")
        ```
    """
    
    def __init__(
        self,
        collection_name: str,
        persist_directory: str | Path | None = None,
        distance_metric: str = "cosine",
        client: ClientAPI | None = None,
    ) -> None:
        """Initialize the ChromaDB vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection to use.
            persist_directory: Optional directory path for persistent storage.
                             If provided, data will be persisted to disk.
            distance_metric: Distance function to use. Options: "cosine", "l2", "ip".
                           Default is "cosine" for cosine similarity.
            client: Optional pre-configured ChromaDB client. If not provided,
                   a new client will be created.
        
        Raises:
            VectorStoreError: If initialization fails.
        """
        self._collection_name = collection_name
        self._persist_dir = Path(persist_directory) if persist_directory else None
        self._distance_metric = distance_metric
        
        try:
            if client is not None:
                self._client = client
            elif self._persist_dir:
                # Create persistent client
                self._persist_dir.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=str(self._persist_dir),
                    settings=Settings(anonymized_telemetry=False)
                )
            else:
                # Use in-memory client
                self._client = chromadb.EphemeralClient(
                    settings=Settings(anonymized_telemetry=False)
                )
            
            # Get or create the collection
            self._collection: Collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": distance_metric}
            )
            
            logger.debug(
                f"Initialized ChromaDB store '{collection_name}' with "
                f"{self._collection.count()} existing vectors"
            )
            
        except Exception as e:
            raise VectorStoreError(
                f"Failed to initialize ChromaDB store '{collection_name}'",
                details=str(e)
            )
    
    @property
    def collection_name(self) -> str:
        """Return the collection name."""
        return self._collection_name
    
    @property
    def persist_directory(self) -> Path | None:
        """Return the persist directory path."""
        return self._persist_dir
    
    @property
    def distance_metric(self) -> str:
        """Return the distance metric used."""
        return self._distance_metric
    
    def add_embeddings(
        self, chunks: list[Chunk], embeddings: list[EmbeddingVector]
    ) -> int:
        """Add chunks and their embeddings to the vector store.
        
        Args:
            chunks: A list of Chunk objects to store.
            embeddings: A list of embedding vectors corresponding to each chunk.
                       Must be the same length as chunks.
        
        Returns:
            The number of chunks successfully added to the store.
        
        Raises:
            ValueError: If chunks and embeddings have different lengths.
            VectorStoreError: If the operation fails.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) must match "
                f"number of embeddings ({len(embeddings)})"
            )
        
        if not chunks:
            return 0
        
        try:
            # Prepare data for ChromaDB
            ids: list[str] = []
            documents: list[str] = []
            metadatas: list[dict[str, Any]] = []
            embeddings_to_add: list[EmbeddingVector] = []
            
            for i, chunk in enumerate(chunks):
                # Prepare metadata - ChromaDB requires values to be str, int, float, or bool
                metadata = self._prepare_metadata(chunk)
                
                ids.append(chunk.id)
                documents.append(chunk.content)
                metadatas.append(metadata)
                embeddings_to_add.append(embeddings[i])
            
            # Add to collection (upsert behavior - updates if exists)
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings_to_add,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.debug(f"Added {len(chunks)} chunks to ChromaDB store")
            return len(chunks)
            
        except Exception as e:
            raise VectorStoreError(
                "Failed to add embeddings to ChromaDB store",
                details=str(e)
            )
    
    def _prepare_metadata(self, chunk: Chunk) -> dict[str, Any]:
        """Prepare chunk metadata for ChromaDB storage.
        
        ChromaDB metadata values must be str, int, float, or bool.
        This method converts complex types to compatible formats.
        
        Args:
            chunk: The chunk to prepare metadata for.
        
        Returns:
            A dictionary with ChromaDB-compatible metadata values.
        """
        metadata: dict[str, Any] = {
            "document_id": chunk.document_id,
            "start_idx": chunk.start_idx,
            "end_idx": chunk.end_idx,
        }
        
        # Add chunk metadata, converting values as needed
        for key, value in chunk.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                metadata[key] = value
            elif value is None:
                metadata[key] = ""
            else:
                # Convert other types to string
                metadata[key] = str(value)
        
        return metadata
    
    def similarity_search(
        self, query_embedding: EmbeddingVector, k: int,
        filter_metadata: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        """Find the most similar chunks to a query embedding.
        
        Args:
            query_embedding: The embedding vector of the query.
            k: The maximum number of results to return.
            filter_metadata: Optional metadata filters to apply.
                           Example: {"source": "document.txt", "category": "tech"}
        
        Returns:
            A list of SearchResult objects sorted by similarity score
            (highest first), with at most k results.
        
        Raises:
            VectorStoreError: If the search operation fails.
        """
        if self.count() == 0:
            return []
        
        try:
            # Ensure k doesn't exceed available vectors
            k = min(k, self.count())
            
            if k == 0:
                return []
            
            # Build the where clause for metadata filtering
            where_filter = self._build_where_filter(filter_metadata)
            
            # Query the collection
            query_params: dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": k,
                "include": ["documents", "metadatas", "distances"]
            }
            
            if where_filter:
                query_params["where"] = where_filter
            
            results = self._collection.query(**query_params)
            
            # Build SearchResult objects
            search_results: list[SearchResult] = []
            
            if not results["ids"] or not results["ids"][0]:
                return []
            
            ids = results["ids"][0]
            documents = results["documents"][0] if results["documents"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            distances = results["distances"][0] if results["distances"] else []
            
            for i, chunk_id in enumerate(ids):
                # Get document content
                content = documents[i] if i < len(documents) else ""
                
                # Get metadata
                metadata = metadatas[i] if i < len(metadatas) else {}
                
                # Get distance and convert to similarity score
                distance = distances[i] if i < len(distances) else 1.0
                score = self._distance_to_score(distance)
                
                # Reconstruct Chunk object
                chunk = Chunk(
                    id=chunk_id,
                    document_id=metadata.get("document_id", ""),
                    content=content,
                    start_idx=metadata.get("start_idx", 0),
                    end_idx=metadata.get("end_idx", 0),
                    metadata={k: v for k, v in metadata.items() 
                             if k not in ("document_id", "start_idx", "end_idx")}
                )
                
                result = SearchResult(
                    chunk_id=chunk_id,
                    score=score,
                    chunk=chunk,
                    metadata=metadata
                )
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            raise VectorStoreError(
                "Failed to perform similarity search in ChromaDB",
                details=str(e)
            )
    
    def _build_where_filter(
        self, filter_metadata: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Build ChromaDB where filter from metadata dictionary.
        
        Args:
            filter_metadata: Dictionary of metadata key-value pairs to filter by.
        
        Returns:
            ChromaDB-compatible where filter or None if no filter.
        """
        if not filter_metadata:
            return None
        
        # For simple equality filters, ChromaDB accepts a direct dictionary
        # For more complex filters, we would use $and, $or operators
        if len(filter_metadata) == 1:
            key, value = next(iter(filter_metadata.items()))
            return {key: value}
        else:
            # Multiple filters - use $and operator
            conditions = [{k: v} for k, v in filter_metadata.items()]
            return {"$and": conditions}
    
    def _distance_to_score(self, distance: float) -> float:
        """Convert ChromaDB distance to similarity score.
        
        ChromaDB returns distances, not similarities. This method converts
        the distance to a similarity score in the range [0, 1].
        
        For cosine distance: similarity = 1 - distance (distance is 1 - cosine_similarity)
        For L2 distance: similarity = 1 / (1 + distance)
        For inner product: ChromaDB returns negative inner product as distance
                          similarity = -distance (for normalized vectors, this is cosine)
        
        Args:
            distance: The distance returned by ChromaDB.
        
        Returns:
            A similarity score between 0.0 and 1.0.
        """
        if self._distance_metric == "cosine":
            # Cosine distance = 1 - cosine_similarity
            # So similarity = 1 - distance
            score = 1.0 - distance
        elif self._distance_metric == "l2":
            # L2 distance - convert to similarity using inverse
            score = 1.0 / (1.0 + distance)
        elif self._distance_metric == "ip":
            # ChromaDB returns negative inner product as distance for "ip" space
            # For normalized vectors, inner product = cosine similarity
            # So similarity = -distance
            score = -distance
        else:
            # Default fallback
            score = max(0.0, 1.0 - distance)
        
        # Ensure score is in valid range
        return max(0.0, min(1.0, score))
    
    def delete(self, chunk_ids: list[str]) -> int:
        """Delete chunks from the vector store by their IDs.
        
        Args:
            chunk_ids: A list of chunk IDs to delete.
        
        Returns:
            The number of chunks successfully deleted.
        
        Raises:
            VectorStoreError: If the delete operation fails.
        """
        if not chunk_ids:
            return 0
        
        try:
            # Check which IDs exist before deletion
            existing = self._collection.get(ids=chunk_ids)
            existing_ids = set(existing["ids"]) if existing["ids"] else set()
            
            if not existing_ids:
                return 0
            
            # Delete from collection
            self._collection.delete(ids=list(existing_ids))
            
            logger.debug(f"Deleted {len(existing_ids)} chunks from ChromaDB store")
            return len(existing_ids)
            
        except Exception as e:
            raise VectorStoreError(
                "Failed to delete chunks from ChromaDB store",
                details=str(e)
            )
    
    def count(self) -> int:
        """Return the total number of chunks in the vector store.
        
        Returns:
            The total number of chunks currently stored.
        """
        return self._collection.count()
    
    def clear(self) -> None:
        """Remove all chunks and embeddings from the vector store.
        
        This method completely clears the store, removing all stored
        chunks and their associated embeddings.
        """
        try:
            # Get all IDs in the collection
            all_items = self._collection.get()
            
            if all_items["ids"]:
                self._collection.delete(ids=all_items["ids"])
            
            logger.debug("Cleared ChromaDB store")
            
        except Exception as e:
            raise VectorStoreError(
                "Failed to clear ChromaDB store",
                details=str(e)
            )
    
    def get_by_ids(self, chunk_ids: list[str]) -> list[Chunk]:
        """Retrieve chunks by their IDs.
        
        Args:
            chunk_ids: A list of chunk IDs to retrieve.
        
        Returns:
            A list of Chunk objects for the found IDs.
        
        Raises:
            VectorStoreError: If the retrieval fails.
        """
        if not chunk_ids:
            return []
        
        try:
            results = self._collection.get(
                ids=chunk_ids,
                include=["documents", "metadatas"]
            )
            
            chunks: list[Chunk] = []
            
            if not results["ids"]:
                return []
            
            for i, chunk_id in enumerate(results["ids"]):
                content = results["documents"][i] if results["documents"] else ""
                metadata = results["metadatas"][i] if results["metadatas"] else {}
                
                chunk = Chunk(
                    id=chunk_id,
                    document_id=metadata.get("document_id", ""),
                    content=content,
                    start_idx=metadata.get("start_idx", 0),
                    end_idx=metadata.get("end_idx", 0),
                    metadata={k: v for k, v in metadata.items() 
                             if k not in ("document_id", "start_idx", "end_idx")}
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            raise VectorStoreError(
                "Failed to retrieve chunks by IDs from ChromaDB",
                details=str(e)
            )
    
    def delete_collection(self) -> None:
        """Delete the entire collection from ChromaDB.
        
        This is a destructive operation that removes the collection
        and all its data.
        
        Raises:
            VectorStoreError: If the deletion fails.
        """
        try:
            self._client.delete_collection(self._collection_name)
            logger.debug(f"Deleted ChromaDB collection '{self._collection_name}'")
        except Exception as e:
            raise VectorStoreError(
                f"Failed to delete collection '{self._collection_name}'",
                details=str(e)
            )
    
    def get_collection_stats(self) -> dict[str, Any]:
        """Get statistics about the collection.
        
        Returns:
            A dictionary with collection statistics.
        """
        return {
            "name": self._collection_name,
            "count": self.count(),
            "metadata": self._collection.metadata,
            "persist_directory": str(self._persist_dir) if self._persist_dir else None,
            "distance_metric": self._distance_metric,
        }
