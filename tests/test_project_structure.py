"""Tests for verifying the project structure of Grounded GraphRAG Tutor."""

from pathlib import Path

import pytest
import yaml


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


class TestProjectStructure:
    """Test class for verifying project structure exists."""

    def test_project_root_exists(self):
        """Verify the project root directory exists."""
        assert PROJECT_ROOT.exists(), f"Project root {PROJECT_ROOT} does not exist"
        assert PROJECT_ROOT.is_dir(), f"Project root {PROJECT_ROOT} is not a directory"

    def test_src_directory_exists(self):
        """Verify the src directory exists."""
        src_dir = PROJECT_ROOT / "src"
        assert src_dir.exists(), f"src directory {src_dir} does not exist"
        assert src_dir.is_dir(), f"src path {src_dir} is not a directory"

    def test_tests_directory_exists(self):
        """Verify the tests directory exists."""
        tests_dir = PROJECT_ROOT / "tests"
        assert tests_dir.exists(), f"tests directory {tests_dir} does not exist"
        assert tests_dir.is_dir(), f"tests path {tests_dir} is not a directory"

    def test_configs_directory_exists(self):
        """Verify the configs directory exists."""
        configs_dir = PROJECT_ROOT / "configs"
        assert configs_dir.exists(), f"configs directory {configs_dir} does not exist"
        assert configs_dir.is_dir(), f"configs path {configs_dir} is not a directory"

    def test_scripts_directory_exists(self):
        """Verify the scripts directory exists."""
        scripts_dir = PROJECT_ROOT / "scripts"
        assert scripts_dir.exists(), f"scripts directory {scripts_dir} does not exist"
        assert scripts_dir.is_dir(), f"scripts path {scripts_dir} is not a directory"


class TestSrcSubdirectories:
    """Test class for verifying src subdirectories exist."""

    @pytest.fixture
    def src_dir(self):
        """Return the src directory path."""
        return PROJECT_ROOT / "src"

    def test_connectors_directory_exists(self, src_dir):
        """Verify the connectors directory exists."""
        connectors_dir = src_dir / "connectors"
        assert connectors_dir.exists(), f"connectors directory {connectors_dir} does not exist"
        assert connectors_dir.is_dir(), f"connectors path {connectors_dir} is not a directory"

    def test_ingestion_directory_exists(self, src_dir):
        """Verify the ingestion directory exists."""
        ingestion_dir = src_dir / "ingestion"
        assert ingestion_dir.exists(), f"ingestion directory {ingestion_dir} does not exist"
        assert ingestion_dir.is_dir(), f"ingestion path {ingestion_dir} is not a directory"

    def test_embeddings_directory_exists(self, src_dir):
        """Verify the embeddings directory exists."""
        embeddings_dir = src_dir / "embeddings"
        assert embeddings_dir.exists(), f"embeddings directory {embeddings_dir} does not exist"
        assert embeddings_dir.is_dir(), f"embeddings path {embeddings_dir} is not a directory"

    def test_store_directory_exists(self, src_dir):
        """Verify the store directory exists."""
        store_dir = src_dir / "store"
        assert store_dir.exists(), f"store directory {store_dir} does not exist"
        assert store_dir.is_dir(), f"store path {store_dir} is not a directory"

    def test_retrieval_directory_exists(self, src_dir):
        """Verify the retrieval directory exists."""
        retrieval_dir = src_dir / "retrieval"
        assert retrieval_dir.exists(), f"retrieval directory {retrieval_dir} does not exist"
        assert retrieval_dir.is_dir(), f"retrieval path {retrieval_dir} is not a directory"

    def test_graphs_directory_exists(self, src_dir):
        """Verify the graphs directory exists."""
        graphs_dir = src_dir / "graphs"
        assert graphs_dir.exists(), f"graphs directory {graphs_dir} does not exist"
        assert graphs_dir.is_dir(), f"graphs path {graphs_dir} is not a directory"

    def test_app_directory_exists(self, src_dir):
        """Verify the app directory exists."""
        app_dir = src_dir / "app"
        assert app_dir.exists(), f"app directory {app_dir} does not exist"
        assert app_dir.is_dir(), f"app path {app_dir} is not a directory"

    def test_eval_directory_exists(self, src_dir):
        """Verify the eval directory exists."""
        eval_dir = src_dir / "eval"
        assert eval_dir.exists(), f"eval directory {eval_dir} does not exist"
        assert eval_dir.is_dir(), f"eval path {eval_dir} is not a directory"


class TestInitFiles:
    """Test class for verifying __init__.py files exist."""

    def test_src_init_exists(self):
        """Verify src/__init__.py exists."""
        init_file = PROJECT_ROOT / "src" / "__init__.py"
        assert init_file.exists(), f"src/__init__.py {init_file} does not exist"
        assert init_file.is_file(), f"src/__init__.py {init_file} is not a file"

    def test_connectors_init_exists(self):
        """Verify src/connectors/__init__.py exists."""
        init_file = PROJECT_ROOT / "src" / "connectors" / "__init__.py"
        assert init_file.exists(), f"connectors/__init__.py {init_file} does not exist"
        assert init_file.is_file(), f"connectors/__init__.py {init_file} is not a file"

    def test_ingestion_init_exists(self):
        """Verify src/ingestion/__init__.py exists."""
        init_file = PROJECT_ROOT / "src" / "ingestion" / "__init__.py"
        assert init_file.exists(), f"ingestion/__init__.py {init_file} does not exist"
        assert init_file.is_file(), f"ingestion/__init__.py {init_file} is not a file"

    def test_embeddings_init_exists(self):
        """Verify src/embeddings/__init__.py exists."""
        init_file = PROJECT_ROOT / "src" / "embeddings" / "__init__.py"
        assert init_file.exists(), f"embeddings/__init__.py {init_file} does not exist"
        assert init_file.is_file(), f"embeddings/__init__.py {init_file} is not a file"

    def test_store_init_exists(self):
        """Verify src/store/__init__.py exists."""
        init_file = PROJECT_ROOT / "src" / "store" / "__init__.py"
        assert init_file.exists(), f"store/__init__.py {init_file} does not exist"
        assert init_file.is_file(), f"store/__init__.py {init_file} is not a file"

    def test_retrieval_init_exists(self):
        """Verify src/retrieval/__init__.py exists."""
        init_file = PROJECT_ROOT / "src" / "retrieval" / "__init__.py"
        assert init_file.exists(), f"retrieval/__init__.py {init_file} does not exist"
        assert init_file.is_file(), f"retrieval/__init__.py {init_file} is not a file"

    def test_graphs_init_exists(self):
        """Verify src/graphs/__init__.py exists."""
        init_file = PROJECT_ROOT / "src" / "graphs" / "__init__.py"
        assert init_file.exists(), f"graphs/__init__.py {init_file} does not exist"
        assert init_file.is_file(), f"graphs/__init__.py {init_file} is not a file"

    def test_app_init_exists(self):
        """Verify src/app/__init__.py exists."""
        init_file = PROJECT_ROOT / "src" / "app" / "__init__.py"
        assert init_file.exists(), f"app/__init__.py {init_file} does not exist"
        assert init_file.is_file(), f"app/__init__.py {init_file} is not a file"

    def test_eval_init_exists(self):
        """Verify src/eval/__init__.py exists."""
        init_file = PROJECT_ROOT / "src" / "eval" / "__init__.py"
        assert init_file.exists(), f"eval/__init__.py {init_file} does not exist"
        assert init_file.is_file(), f"eval/__init__.py {init_file} is not a file"

    def test_tests_init_exists(self):
        """Verify tests/__init__.py exists."""
        init_file = PROJECT_ROOT / "tests" / "__init__.py"
        assert init_file.exists(), f"tests/__init__.py {init_file} does not exist"
        assert init_file.is_file(), f"tests/__init__.py {init_file} is not a file"


class TestTestsSubdirectories:
    """Test class for verifying tests subdirectories mirror src structure."""

    @pytest.fixture
    def tests_dir(self):
        """Return the tests directory path."""
        return PROJECT_ROOT / "tests"

    def test_connectors_test_directory_exists(self, tests_dir):
        """Verify tests/connectors directory exists."""
        connectors_dir = tests_dir / "connectors"
        assert connectors_dir.exists(), f"tests/connectors directory {connectors_dir} does not exist"

    def test_ingestion_test_directory_exists(self, tests_dir):
        """Verify tests/ingestion directory exists."""
        ingestion_dir = tests_dir / "ingestion"
        assert ingestion_dir.exists(), f"tests/ingestion directory {ingestion_dir} does not exist"

    def test_embeddings_test_directory_exists(self, tests_dir):
        """Verify tests/embeddings directory exists."""
        embeddings_dir = tests_dir / "embeddings"
        assert embeddings_dir.exists(), f"tests/embeddings directory {embeddings_dir} does not exist"

    def test_store_test_directory_exists(self, tests_dir):
        """Verify tests/store directory exists."""
        store_dir = tests_dir / "store"
        assert store_dir.exists(), f"tests/store directory {store_dir} does not exist"

    def test_retrieval_test_directory_exists(self, tests_dir):
        """Verify tests/retrieval directory exists."""
        retrieval_dir = tests_dir / "retrieval"
        assert retrieval_dir.exists(), f"tests/retrieval directory {retrieval_dir} does not exist"

    def test_graphs_test_directory_exists(self, tests_dir):
        """Verify tests/graphs directory exists."""
        graphs_dir = tests_dir / "graphs"
        assert graphs_dir.exists(), f"tests/graphs directory {graphs_dir} does not exist"

    def test_app_test_directory_exists(self, tests_dir):
        """Verify tests/app directory exists."""
        app_dir = tests_dir / "app"
        assert app_dir.exists(), f"tests/app directory {app_dir} does not exist"

    def test_eval_test_directory_exists(self, tests_dir):
        """Verify tests/eval directory exists."""
        eval_dir = tests_dir / "eval"
        assert eval_dir.exists(), f"tests/eval directory {eval_dir} does not exist"


class TestConfigFiles:
    """Test class for verifying configuration files."""

    def test_default_config_exists(self):
        """Verify configs/default.yaml exists."""
        config_file = PROJECT_ROOT / "configs" / "default.yaml"
        assert config_file.exists(), f"configs/default.yaml {config_file} does not exist"
        assert config_file.is_file(), f"configs/default.yaml {config_file} is not a file"

    def test_default_config_is_valid_yaml(self):
        """Verify configs/default.yaml is valid YAML."""
        config_file = PROJECT_ROOT / "configs" / "default.yaml"
        with open(config_file, "r", encoding="utf-8") as f:
            try:
                config = yaml.safe_load(f)
                assert config is not None, "configs/default.yaml is empty"
                assert isinstance(config, dict), "configs/default.yaml should contain a YAML mapping"
            except yaml.YAMLError as e:
                pytest.fail(f"configs/default.yaml is not valid YAML: {e}")

    def test_default_config_has_required_sections(self):
        """Verify configs/default.yaml has required configuration sections."""
        config_file = PROJECT_ROOT / "configs" / "default.yaml"
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        required_sections = [
            "corpus",
            "ingestion",
            "embeddings",
            "vectorstore",
            "retrieval",
            "llm",
            "graph",
            "api",
            "logging",
            "eval",
        ]

        for section in required_sections:
            assert section in config, f"configs/default.yaml missing required section: {section}"


class TestProjectFiles:
    """Test class for verifying root project files."""

    def test_pyproject_toml_exists(self):
        """Verify pyproject.toml exists."""
        pyproject_file = PROJECT_ROOT / "pyproject.toml"
        assert pyproject_file.exists(), f"pyproject.toml {pyproject_file} does not exist"
        assert pyproject_file.is_file(), f"pyproject.toml {pyproject_file} is not a file"

    def test_env_example_exists(self):
        """Verify .env.example exists."""
        env_file = PROJECT_ROOT / ".env.example"
        assert env_file.exists(), f".env.example {env_file} does not exist"
        assert env_file.is_file(), f".env.example {env_file} is not a file"

    def test_gitignore_exists(self):
        """Verify .gitignore exists."""
        gitignore_file = PROJECT_ROOT / ".gitignore"
        assert gitignore_file.exists(), f".gitignore {gitignore_file} does not exist"
        assert gitignore_file.is_file(), f".gitignore {gitignore_file} is not a file"

    def test_readme_exists(self):
        """Verify README.md exists."""
        readme_file = PROJECT_ROOT / "README.md"
        assert readme_file.exists(), f"README.md {readme_file} does not exist"
        assert readme_file.is_file(), f"README.md {readme_file} is not a file"

    def test_ingest_script_exists(self):
        """Verify scripts/ingest.py exists."""
        ingest_file = PROJECT_ROOT / "scripts" / "ingest.py"
        assert ingest_file.exists(), f"scripts/ingest.py {ingest_file} does not exist"
        assert ingest_file.is_file(), f"scripts/ingest.py {ingest_file} is not a file"

    def test_eval_script_exists(self):
        """Verify scripts/eval.py exists."""
        eval_file = PROJECT_ROOT / "scripts" / "eval.py"
        assert eval_file.exists(), f"scripts/eval.py {eval_file} does not exist"
        assert eval_file.is_file(), f"scripts/eval.py {eval_file} is not a file"
