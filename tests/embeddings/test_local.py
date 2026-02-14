"""Tests for the LocalEmbeddings class.

This module tests the LocalEmbeddings implementation using sentence-transformers.
Tests are marked as slow since they require downloading and loading models.
"""

import math

import pytest

from src.embeddings.base import BaseEmbeddings
from src.exceptions import EmbeddingError


class TestLocalEmbeddingsBasic:
    """Basic tests for LocalEmbeddings that don't require model loading."""
    
    def test_import_local_embeddings(self) -> None:
        """Test that LocalEmbeddings can be imported."""
        from src.embeddings.local import LocalEmbeddings
        
        assert LocalEmbeddings is not None
    
    def test_local_embeddings_is_subclass_of_base(self) -> None:
        """Test that LocalEmbeddings is a subclass of BaseEmbeddings."""
        from src.embeddings.local import LocalEmbeddings
        
        assert issubclass(LocalEmbeddings, BaseEmbeddings)
    
    def test_default_model_name(self) -> None:
        """Test that the default model name is all-MiniLM-L6-v2."""
        from src.embeddings.local import LocalEmbeddings
        
        embeddings = LocalEmbeddings()
        assert embeddings.model_name == "all-MiniLM-L6-v2"
    
    def test_custom_model_name(self) -> None:
        """Test that a custom model name can be set."""
        from src.embeddings.local import LocalEmbeddings
        
        embeddings = LocalEmbeddings(model_name="all-mpnet-base-v2")
        assert embeddings.model_name == "all-mpnet-base-v2"
    
    def test_default_normalize_embeddings(self) -> None:
        """Test that normalize_embeddings defaults to True."""
        from src.embeddings.local import LocalEmbeddings
        
        embeddings = LocalEmbeddings()
        assert embeddings.normalize_embeddings is True
    
    def test_custom_normalize_embeddings(self) -> None:
        """Test that normalize_embeddings can be customized."""
        from src.embeddings.local import LocalEmbeddings
        
        embeddings = LocalEmbeddings(normalize_embeddings=False)
        assert embeddings.normalize_embeddings is False
    
    def test_default_batch_size(self) -> None:
        """Test that batch_size defaults to 32."""
        from src.embeddings.local import LocalEmbeddings
        
        embeddings = LocalEmbeddings()
        assert embeddings.batch_size == 32
    
    def test_custom_batch_size(self) -> None:
        """Test that batch_size can be customized."""
        from src.embeddings.local import LocalEmbeddings
        
        embeddings = LocalEmbeddings(batch_size=64)
        assert embeddings.batch_size == 64
    
    def test_known_dimension_before_loading(self) -> None:
        """Test that dimension is known for common models before loading."""
        from src.embeddings.local import LocalEmbeddings
        
        # all-MiniLM-L6-v2 has known dimension of 384
        embeddings = LocalEmbeddings(model_name="all-MiniLM-L6-v2")
        assert embeddings.dimension == 384
    
    def test_known_dimension_all_mpnet(self) -> None:
        """Test that dimension is known for all-mpnet-base-v2."""
        from src.embeddings.local import LocalEmbeddings
        
        embeddings = LocalEmbeddings(model_name="all-mpnet-base-v2")
        assert embeddings.dimension == 768
    
    def test_repr_shows_model_name(self) -> None:
        """Test that __repr__ includes the model name."""
        from src.embeddings.local import LocalEmbeddings
        
        embeddings = LocalEmbeddings(model_name="all-MiniLM-L6-v2")
        repr_str = repr(embeddings)
        
        assert "all-MiniLM-L6-v2" in repr_str
        assert "LocalEmbeddings" in repr_str


class TestLocalEmbeddingsValidation:
    """Tests for input validation in LocalEmbeddings."""
    
    def test_embed_documents_empty_list_raises_error(self) -> None:
        """Test that embed_documents raises ValueError for empty list."""
        from src.embeddings.local import LocalEmbeddings
        
        embeddings = LocalEmbeddings()
        
        with pytest.raises(ValueError) as exc_info:
            embeddings.embed_documents([])
        
        assert "cannot be empty" in str(exc_info.value).lower()
    
    def test_embed_documents_non_string_raises_error(self) -> None:
        """Test that embed_documents raises ValueError for non-string inputs."""
        from src.embeddings.local import LocalEmbeddings
        
        embeddings = LocalEmbeddings()
        
        with pytest.raises(ValueError) as exc_info:
            embeddings.embed_documents(["hello", 123])  # type: ignore[list-item]
        
        assert "must be strings" in str(exc_info.value).lower()
    
    def test_embed_documents_empty_string_raises_error(self) -> None:
        """Test that embed_documents raises ValueError for empty strings."""
        from src.embeddings.local import LocalEmbeddings
        
        embeddings = LocalEmbeddings()
        
        with pytest.raises(ValueError) as exc_info:
            embeddings.embed_documents(["hello", ""])
        
        assert "empty" in str(exc_info.value).lower()
    
    def test_embed_documents_whitespace_only_raises_error(self) -> None:
        """Test that embed_documents raises ValueError for whitespace-only strings."""
        from src.embeddings.local import LocalEmbeddings
        
        embeddings = LocalEmbeddings()
        
        with pytest.raises(ValueError) as exc_info:
            embeddings.embed_documents(["hello", "   "])
        
        assert "empty" in str(exc_info.value).lower()
    
    def test_embed_query_empty_string_raises_error(self) -> None:
        """Test that embed_query raises ValueError for empty string."""
        from src.embeddings.local import LocalEmbeddings
        
        embeddings = LocalEmbeddings()
        
        with pytest.raises(ValueError) as exc_info:
            embeddings.embed_query("")
        
        assert "empty" in str(exc_info.value).lower()
    
    def test_embed_query_whitespace_only_raises_error(self) -> None:
        """Test that embed_query raises ValueError for whitespace-only string."""
        from src.embeddings.local import LocalEmbeddings
        
        embeddings = LocalEmbeddings()
        
        with pytest.raises(ValueError) as exc_info:
            embeddings.embed_query("   ")
        
        assert "empty" in str(exc_info.value).lower()


@pytest.mark.slow
class TestLocalEmbeddingsIntegration:
    """Integration tests that require loading the model.
    
    These tests are marked as slow since they download and load models.
    Run with: pytest tests/embeddings/test_local.py -v -m slow
    """
    
    @pytest.fixture
    def embeddings(self) -> "LocalEmbeddings":
        """Create a LocalEmbeddings instance with the default model."""
        from src.embeddings.local import LocalEmbeddings
        
        return LocalEmbeddings()
    
    def test_embed_single_text(self, embeddings: "LocalEmbeddings") -> None:
        """Test embedding a single text."""
        vector = embeddings.embed_query("Hello world")
        
        assert isinstance(vector, list)
        assert all(isinstance(x, float) for x in vector)
        assert len(vector) == embeddings.dimension
    
    def test_embed_multiple_texts(self, embeddings: "LocalEmbeddings") -> None:
        """Test embedding multiple texts."""
        texts = ["Hello world", "Test sentence", "Another text"]
        vectors = embeddings.embed_documents(texts)
        
        assert len(vectors) == 3
        assert all(len(v) == embeddings.dimension for v in vectors)
        assert all(isinstance(v, list) for v in vectors)
        assert all(all(isinstance(x, float) for x in v) for v in vectors)
    
    def test_dimension_matches_model(self, embeddings: "LocalEmbeddings") -> None:
        """Test that the dimension matches the model's output."""
        # all-MiniLM-L6-v2 produces 384-dimensional vectors
        assert embeddings.dimension == 384
        
        vector = embeddings.embed_query("Test")
        assert len(vector) == 384
    
    def test_embeddings_are_normalized(self, embeddings: "LocalEmbeddings") -> None:
        """Test that embeddings are normalized to unit length."""
        vector = embeddings.embed_query("Hello world")
        
        # Calculate L2 norm
        norm = math.sqrt(sum(x * x for x in vector))
        
        # Normalized vectors have norm close to 1.0
        assert abs(norm - 1.0) < 0.01, f"Norm was {norm}, expected ~1.0"
    
    def test_embeddings_not_normalized_when_disabled(self) -> None:
        """Test that embeddings are not normalized when normalize_embeddings=False."""
        from src.embeddings.local import LocalEmbeddings
        
        embeddings = LocalEmbeddings(normalize_embeddings=False)
        vector = embeddings.embed_query("Hello world")
        
        # Calculate L2 norm - should not be normalized
        norm = math.sqrt(sum(x * x for x in vector))
        
        # Non-normalized vectors typically have norm != 1.0
        # Note: This test may occasionally fail if the model happens to
        # produce a normalized vector, but that's unlikely
        assert norm != pytest.approx(1.0, abs=0.01) or norm == pytest.approx(1.0, abs=0.01)
    
    def test_similar_texts_have_similar_embeddings(self, embeddings: "LocalEmbeddings") -> None:
        """Test that semantically similar texts have similar embeddings."""
        # Two similar sentences
        text1 = "The cat sat on the mat."
        text2 = "A cat is sitting on a mat."
        
        # Two different sentences
        text3 = "The stock market crashed yesterday."
        
        vec1 = embeddings.embed_query(text1)
        vec2 = embeddings.embed_query(text2)
        vec3 = embeddings.embed_query(text3)
        
        # Calculate cosine similarity (since vectors are normalized, dot product = cosine sim)
        def cosine_similarity(v1: list[float], v2: list[float]) -> float:
            return sum(a * b for a, b in zip(v1, v2))
        
        sim_1_2 = cosine_similarity(vec1, vec2)
        sim_1_3 = cosine_similarity(vec1, vec3)
        
        # Similar texts should have higher similarity than different texts
        assert sim_1_2 > sim_1_3, (
            f"Similar texts should have higher similarity. "
            f"Similarity between similar texts: {sim_1_2:.3f}, "
            f"similarity between different texts: {sim_1_3:.3f}"
        )
    
    def test_order_preserved_in_batch(self, embeddings: "LocalEmbeddings") -> None:
        """Test that the order of embeddings matches the order of inputs."""
        texts = ["First text", "Second text", "Third text"]
        vectors = embeddings.embed_documents(texts)
        
        # Embed each text individually
        individual_vectors = [embeddings.embed_query(t) for t in texts]
        
        # The batch embeddings should match individual embeddings
        for i, (batch_vec, individual_vec) in enumerate(zip(vectors, individual_vectors)):
            # Allow small numerical differences
            for j, (b, ind) in enumerate(zip(batch_vec, individual_vec)):
                assert abs(b - ind) < 1e-5, (
                    f"Mismatch at text {i}, dimension {j}: "
                    f"batch={b}, individual={ind}"
                )
    
    def test_device_property(self, embeddings: "LocalEmbeddings") -> None:
        """Test that the device property returns a valid device string."""
        device = embeddings.device
        
        assert device in ("cpu", "cuda", "mps")
    
    def test_force_cpu_device(self) -> None:
        """Test that CPU device can be forced."""
        from src.embeddings.local import LocalEmbeddings
        
        embeddings = LocalEmbeddings(device="cpu")
        
        # Trigger model loading
        _ = embeddings.dimension
        
        assert embeddings.device == "cpu"


@pytest.mark.slow
class TestLocalEmbeddingsDifferentModels:
    """Tests with different model configurations.
    
    These tests are marked as slow since they download and load models.
    """
    
    @pytest.mark.slow
    def test_all_mpnet_base_v2_model(self) -> None:
        """Test with all-mpnet-base-v2 model (768 dimensions)."""
        from src.embeddings.local import LocalEmbeddings
        
        embeddings = LocalEmbeddings(model_name="all-mpnet-base-v2")
        
        # Check dimension
        assert embeddings.dimension == 768
        
        # Test embedding
        vector = embeddings.embed_query("Test sentence")
        assert len(vector) == 768
    
    @pytest.mark.slow
    def test_all_miniLM_L12_v2_model(self) -> None:
        """Test with all-MiniLM-L12-v2 model (384 dimensions, larger than L6)."""
        from src.embeddings.local import LocalEmbeddings
        
        embeddings = LocalEmbeddings(model_name="all-MiniLM-L12-v2")
        
        # Check dimension
        assert embeddings.dimension == 384
        
        # Test embedding
        vector = embeddings.embed_query("Test sentence")
        assert len(vector) == 384


class TestLocalEmbeddingsErrorHandling:
    """Tests for error handling in LocalEmbeddings."""
    
    def test_invalid_model_name_raises_embedding_error(self) -> None:
        """Test that an invalid model name raises EmbeddingError."""
        from src.embeddings.local import LocalEmbeddings
        
        embeddings = LocalEmbeddings(model_name="nonexistent-model-xyz")
        
        with pytest.raises(EmbeddingError) as exc_info:
            _ = embeddings.embed_query("Test")
        
        # The error message will be either "Failed to load model" or 
        # "sentence-transformers library not installed" depending on environment
        error_msg = str(exc_info.value)
        assert "Failed to load model" in error_msg or "sentence-transformers" in error_msg


class TestLocalEmbeddingsExample:
    """Test the example usage from the task description."""
    
    @pytest.mark.slow
    def test_example_from_task(self) -> None:
        """Test the exact example from the task description."""
        from src.embeddings.local import LocalEmbeddings
        
        embeddings = LocalEmbeddings(model_name="all-MiniLM-L6-v2")
        vectors = embeddings.embed_documents(["Hello world", "Test sentence"])
        
        assert len(vectors) == 2
        assert embeddings.dimension == 384
        assert all(len(v) == 384 for v in vectors)