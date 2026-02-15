"""FAISS-based vector store implementation.

This module provides a vector store implementation using Facebook's FAISS library
for efficient similarity search with inner product (cosine similarity with normalized vectors).
"""

import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.exceptions import VectorStoreError
from src.store.base import BaseVectorStore, SearchResult
from src.types import Chunk, EmbeddingVector

logger = logging.getLogger(__name__)


class FAISSVectorStore(BaseVectorStore):
    """Vector store implementation using FAISS for similarity search.
    
    This implementation uses FAISS IndexFlatIP (inner product) for similarity
    search. When vectors are normalized, inner product is equivalent to cosine
    similarity.
    
    The store uses IndexIDMap to associate custom IDs with vectors, enabling
    proper deletion and updates. Embeddings are stored in memory to support
    index rebuilding after deletions.
    
    Attributes:
        dimension: The dimensionality of the embedding vectors.
        persist_dir: Optional directory for persisting the index to disk.
        normalize: Whether to normalize vectors before adding/searching.
    
    Example:
        ```python
        store = FAISSVectorStore(dimension=384, persist_dir="./data/index")
        store.add_embeddings(chunks, embeddings)
        results = store.similarity_search(query_embedding, k=5)
        for result in results:
            print(f"Score: {result.score}, Text: {result.chunk.content[:50]}")
        ```
    """
    
    def __init__(
        self,
        dimension: int,
        persist_dir: str | Path | None = None,
        normalize: bool = True,
    ) -> None:
        """Initialize the FAISS vector store.
        
        Args:
            dimension: The dimensionality of embedding vectors.
            persist_dir: Optional directory path for persisting the index.
                        If provided, the index will be loaded from disk if it exists.
            normalize: Whether to normalize vectors before adding/searching.
                      Default is True for cosine similarity via inner product.
        
        Raises:
            VectorStoreError: If loading from persist_dir fails.
        """
        self._dimension = dimension
        self._persist_dir = Path(persist_dir) if persist_dir else None
        self._normalize = normalize
        
        # Initialize FAISS index with ID mapping
        # IndexIDMap wraps the base index and allows custom IDs
        base_index = faiss.IndexFlatIP(dimension)
        self._index: faiss.IndexIDMap = faiss.IndexIDMap(base_index)
        
        # Store metadata separately (FAISS only stores vectors and IDs)
        # Maps chunk_id to chunk data
        self._chunks: dict[str, Chunk] = {}
        # Maps chunk_id to embedding (needed for rebuilding index after deletions)
        self._embeddings: dict[str, np.ndarray] = {}
        # Counter for generating unique integer IDs for FAISS
        self._id_counter: int = 0
        # Maps chunk_id to FAISS integer ID
        self._chunk_to_faiss_id: dict[str, int] = {}
        # Maps FAISS integer ID to chunk_id
        self._faiss_id_to_chunk: dict[int, str] = {}
        # Track freed FAISS IDs for reuse
        self._free_faiss_ids: list[int] = []
        
        # Load from disk if persist_dir exists
        if self._persist_dir:
            self._load_from_disk()
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors to unit length.
        
        Args:
            vectors: Input vector or matrix of vectors (2D array).
        
        Returns:
            Normalized vector(s).
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms > 0, norms, 1.0)
        return vectors / norms
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize a single vector to unit length.
        
        Args:
            vector: Input vector (1D array).
        
        Returns:
            Normalized vector.
        """
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def _get_index_path(self) -> Path:
        """Get the path to the FAISS index file."""
        return self._persist_dir / "index.faiss"  # type: ignore[operator]
    
    def _get_metadata_path(self) -> Path:
        """Get the path to the metadata JSON file."""
        return self._persist_dir / "metadata.json"  # type: ignore[operator]
    
    def _get_embeddings_path(self) -> Path:
        """Get the path to the embeddings numpy file."""
        return self._persist_dir / "embeddings.npz"  # type: ignore[operator]
    
    def _load_from_disk(self) -> None:
        """Load the index and metadata from disk.
        
        Raises:
            VectorStoreError: If loading fails.
        """
        if not self._persist_dir:
            return
        
        index_path = self._get_index_path()
        metadata_path = self._get_metadata_path()
        embeddings_path = self._get_embeddings_path()
        
        if not index_path.exists() or not metadata_path.exists():
            logger.info(
                f"No existing index found at {self._persist_dir}, "
                "starting with empty store"
            )
            return
        
        try:
            # Load FAISS index
            loaded_index = faiss.read_index(str(index_path))
            if not isinstance(loaded_index, faiss.IndexIDMap):
                raise VectorStoreError(
                    "Loaded index is not an IndexIDMap",
                    details="Index format mismatch"
                )
            self._index = loaded_index
            
            # Load metadata
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            # Restore chunk data
            self._chunks = {}
            for chunk_id, chunk_data in metadata.get("chunks", {}).items():
                self._chunks[chunk_id] = Chunk(**chunk_data)
            
            # Restore ID mappings
            self._id_counter = metadata.get("id_counter", 0)
            self._chunk_to_faiss_id = metadata.get("chunk_to_faiss_id", {})
            self._faiss_id_to_chunk = {
                int(k): v for k, v in metadata.get("faiss_id_to_chunk", {}).items()
            }
            self._free_faiss_ids = metadata.get("free_faiss_ids", [])
            
            # Load embeddings
            if embeddings_path.exists():
                embeddings_data = np.load(str(embeddings_path), allow_pickle=True)
                self._embeddings = {}
                for chunk_id in self._chunks:
                    if chunk_id in embeddings_data.files:
                        self._embeddings[chunk_id] = embeddings_data[chunk_id]
            
            logger.info(
                f"Loaded FAISS index with {len(self._chunks)} vectors from "
                f"{self._persist_dir}"
            )
        except Exception as e:
            raise VectorStoreError(
                f"Failed to load index from {self._persist_dir}",
                details=str(e)
            )
    
    def save(self) -> None:
        """Save the index and metadata to disk.
        
        Raises:
            VectorStoreError: If persist_dir is not set or saving fails.
        """
        if not self._persist_dir:
            raise VectorStoreError(
                "Cannot save without persist_dir",
                details="Initialize with persist_dir to enable persistence"
            )
        
        try:
            # Create directory if it doesn't exist
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self._index, str(self._get_index_path()))
            
            # Save metadata
            chunks_data = {
                chunk_id: chunk.model_dump()
                for chunk_id, chunk in self._chunks.items()
            }
            metadata = {
                "chunks": chunks_data,
                "id_counter": self._id_counter,
                "chunk_to_faiss_id": self._chunk_to_faiss_id,
                "faiss_id_to_chunk": {str(k): v for k, v in self._faiss_id_to_chunk.items()},
                "free_faiss_ids": self._free_faiss_ids,
                "dimension": self._dimension,
            }
            
            with open(self._get_metadata_path(), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            
            # Save embeddings
            if self._embeddings:
                embeddings_data = {chunk_id: emb for chunk_id, emb in self._embeddings.items()}
                np.savez(str(self._get_embeddings_path()), **embeddings_data)
            
            logger.info(
                f"Saved FAISS index with {len(self._chunks)} vectors to "
                f"{self._persist_dir}"
            )
        except Exception as e:
            raise VectorStoreError(
                f"Failed to save index to {self._persist_dir}",
                details=str(e)
            )
    
    def _get_next_faiss_id(self) -> int:
        """Get the next available FAISS ID.
        
        Returns:
            An available integer ID for FAISS.
        """
        if self._free_faiss_ids:
            return self._free_faiss_ids.pop(0)
        faiss_id = self._id_counter
        self._id_counter += 1
        return faiss_id
    
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
            # Convert embeddings to numpy array
            vectors = np.array(embeddings, dtype=np.float32)
            
            # Validate dimensions
            if vectors.shape[1] != self._dimension:
                raise VectorStoreError(
                    f"Embedding dimension mismatch: expected {self._dimension}, "
                    f"got {vectors.shape[1]}"
                )
            
            # Normalize if configured
            if self._normalize:
                vectors = self._normalize_vectors(vectors)
            
            # Process each chunk
            faiss_ids_to_add: list[int] = []
            vectors_to_add: list[np.ndarray] = []
            
            for i, chunk in enumerate(chunks):
                vector = vectors[i]
                
                # Check if chunk already exists (update case)
                if chunk.id in self._chunk_to_faiss_id:
                    # Remove old vector from index
                    old_faiss_id = self._chunk_to_faiss_id[chunk.id]
                    self._index.remove_ids(np.array([old_faiss_id]))
                    # Reuse the same FAISS ID
                    faiss_id = old_faiss_id
                    # Remove old mappings
                    if old_faiss_id in self._faiss_id_to_chunk:
                        del self._faiss_id_to_chunk[old_faiss_id]
                else:
                    # Get new FAISS ID
                    faiss_id = self._get_next_faiss_id()
                
                # Store chunk and embedding
                self._chunks[chunk.id] = chunk
                self._embeddings[chunk.id] = vector
                self._chunk_to_faiss_id[chunk.id] = faiss_id
                self._faiss_id_to_chunk[faiss_id] = chunk.id
                
                faiss_ids_to_add.append(faiss_id)
                vectors_to_add.append(vector)
            
            # Add all vectors to FAISS index at once
            if vectors_to_add:
                vectors_array = np.vstack(vectors_to_add)
                ids_array = np.array(faiss_ids_to_add, dtype=np.int64)
                self._index.add_with_ids(vectors_array, ids_array)
            
            logger.debug(f"Added {len(chunks)} chunks to FAISS store")
            return len(chunks)
            
        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(
                "Failed to add embeddings to store",
                details=str(e)
            )
    
    def similarity_search(
        self, query_embedding: EmbeddingVector, k: int
    ) -> list[SearchResult]:
        """Find the most similar chunks to a query embedding.
        
        Args:
            query_embedding: The embedding vector of the query.
            k: The maximum number of results to return.
        
        Returns:
            A list of SearchResult objects sorted by similarity score
            (highest first), with at most k results.
        
        Raises:
            VectorStoreError: If the search operation fails.
        """
        if self.count() == 0:
            return []
        
        try:
            # Convert query to numpy array (2D for FAISS)
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Validate dimension
            if query_vector.shape[1] != self._dimension:
                raise VectorStoreError(
                    f"Query dimension mismatch: expected {self._dimension}, "
                    f"got {query_vector.shape[1]}"
                )
            
            # Normalize if configured
            if self._normalize:
                query_vector = self._normalize_vectors(query_vector)
            
            # Ensure k doesn't exceed available vectors
            k = min(k, self.count())
            
            # Search
            scores, faiss_ids = self._index.search(query_vector, k)
            
            # Build results
            results: list[SearchResult] = []
            for i in range(len(faiss_ids[0])):
                faiss_id = int(faiss_ids[0][i])
                score = float(scores[0][i])
                
                # Skip invalid indices (can happen with FAISS)
                if faiss_id < 0 or faiss_id not in self._faiss_id_to_chunk:
                    continue
                
                chunk_id = self._faiss_id_to_chunk[faiss_id]
                
                if chunk_id not in self._chunks:
                    continue
                
                chunk = self._chunks[chunk_id]
                
                # Clamp score to [0, 1] range for SearchResult validation
                # Inner product with normalized vectors should be in [-1, 1]
                # but we clamp to [0, 1] for the score field
                clamped_score = max(0.0, min(1.0, score))
                
                result = SearchResult(
                    chunk_id=chunk.id,
                    score=clamped_score,
                    chunk=chunk,
                    metadata=chunk.metadata,
                )
                results.append(result)
            
            return results
            
        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(
                "Failed to perform similarity search",
                details=str(e)
            )
    
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
            deleted_count = 0
            faiss_ids_to_remove: list[int] = []
            
            for chunk_id in chunk_ids:
                if chunk_id not in self._chunk_to_faiss_id:
                    continue
                
                faiss_id = self._chunk_to_faiss_id[chunk_id]
                
                # Remove from metadata
                del self._chunks[chunk_id]
                del self._embeddings[chunk_id]
                del self._chunk_to_faiss_id[chunk_id]
                del self._faiss_id_to_chunk[faiss_id]
                
                # Track FAISS ID for removal and reuse
                faiss_ids_to_remove.append(faiss_id)
                self._free_faiss_ids.append(faiss_id)
                deleted_count += 1
            
            # Remove from FAISS index
            if faiss_ids_to_remove:
                ids_array = np.array(faiss_ids_to_remove, dtype=np.int64)
                self._index.remove_ids(ids_array)
            
            logger.debug(f"Deleted {deleted_count} chunks from FAISS store")
            return deleted_count
            
        except Exception as e:
            raise VectorStoreError(
                "Failed to delete chunks from store",
                details=str(e)
            )
    
    def count(self) -> int:
        """Return the total number of chunks in the vector store.
        
        Returns:
            The total number of chunks currently stored.
        """
        return len(self._chunks)
    
    def clear(self) -> None:
        """Remove all chunks and embeddings from the vector store.
        
        This method completely clears the store, removing all stored
        chunks and their associated embeddings.
        """
        # Reset index
        base_index = faiss.IndexFlatIP(self._dimension)
        self._index = faiss.IndexIDMap(base_index)
        
        # Clear all data structures
        self._chunks.clear()
        self._embeddings.clear()
        self._chunk_to_faiss_id.clear()
        self._faiss_id_to_chunk.clear()
        self._free_faiss_ids.clear()
        self._id_counter = 0
        
        logger.debug("Cleared FAISS store")
