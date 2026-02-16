"""Tests for Docker configuration files.

These tests validate that the Docker configuration files are
well-formed and consistent with the project structure.  They do
NOT require Docker to be installed — they only parse the files.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Dockerfile
# ---------------------------------------------------------------------------


class TestDockerfile:
    """Validate the Dockerfile structure."""

    @pytest.fixture()
    def dockerfile(self) -> str:
        path = ROOT / "Dockerfile"
        assert path.exists(), "Dockerfile not found"
        return path.read_text(encoding="utf-8")

    def test_has_from_instruction(self, dockerfile):
        assert "FROM" in dockerfile

    def test_multi_stage_build(self, dockerfile):
        froms = [l for l in dockerfile.splitlines() if l.strip().startswith("FROM")]
        assert len(froms) >= 2, "Expected multi-stage build with >= 2 FROM instructions"

    def test_python_311_base(self, dockerfile):
        assert "python:3.11" in dockerfile

    def test_uses_slim_image(self, dockerfile):
        assert "slim" in dockerfile

    def test_has_workdir(self, dockerfile):
        assert "WORKDIR" in dockerfile

    def test_copies_source(self, dockerfile):
        assert "COPY src/" in dockerfile or "COPY ./src" in dockerfile

    def test_copies_configs(self, dockerfile):
        assert "COPY configs/" in dockerfile or "COPY ./configs" in dockerfile

    def test_exposes_port(self, dockerfile):
        assert "EXPOSE 8000" in dockerfile

    def test_has_cmd_or_entrypoint(self, dockerfile):
        assert "CMD" in dockerfile or "ENTRYPOINT" in dockerfile

    def test_uvicorn_entrypoint(self, dockerfile):
        assert "uvicorn" in dockerfile
        assert "src.app.main:app" in dockerfile

    def test_healthcheck(self, dockerfile):
        assert "HEALTHCHECK" in dockerfile

    def test_host_0000(self, dockerfile):
        """Server must bind to 0.0.0.0 inside the container."""
        assert "0.0.0.0" in dockerfile

    def test_pip_no_cache(self, dockerfile):
        """Verify --no-cache-dir is used to keep image small."""
        assert "--no-cache-dir" in dockerfile

    def test_unbuffered_python(self, dockerfile):
        assert "PYTHONUNBUFFERED" in dockerfile


# ---------------------------------------------------------------------------
# docker-compose.yml
# ---------------------------------------------------------------------------


class TestDockerCompose:
    """Validate docker-compose.yml structure."""

    @pytest.fixture()
    def compose(self) -> dict:
        path = ROOT / "docker-compose.yml"
        assert path.exists(), "docker-compose.yml not found"
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_has_services(self, compose):
        assert "services" in compose
        assert len(compose["services"]) >= 1

    def test_api_service_exists(self, compose):
        assert "api" in compose["services"]

    def test_api_port_mapping(self, compose):
        api = compose["services"]["api"]
        ports = api.get("ports", [])
        assert any("8000" in str(p) for p in ports)

    def test_api_build_context(self, compose):
        api = compose["services"]["api"]
        build = api.get("build", {})
        assert build.get("context") == "."
        assert build.get("dockerfile") == "Dockerfile"

    def test_api_healthcheck(self, compose):
        api = compose["services"]["api"]
        assert "healthcheck" in api

    def test_api_env_file(self, compose):
        api = compose["services"]["api"]
        env_file = api.get("env_file", [])
        assert ".env" in env_file

    def test_api_volumes(self, compose):
        api = compose["services"]["api"]
        volumes = api.get("volumes", [])
        assert len(volumes) >= 1

    def test_ollama_service_exists(self, compose):
        assert "ollama" in compose["services"]

    def test_ollama_has_profile(self, compose):
        ollama = compose["services"]["ollama"]
        profiles = ollama.get("profiles", [])
        assert "ollama" in profiles

    def test_ollama_port(self, compose):
        ollama = compose["services"]["ollama"]
        ports = ollama.get("ports", [])
        assert any("11434" in str(p) for p in ports)

    def test_volumes_defined(self, compose):
        assert "volumes" in compose
        assert "vectorstore_data" in compose["volumes"]


# ---------------------------------------------------------------------------
# .dockerignore
# ---------------------------------------------------------------------------


class TestDockerignore:
    """Validate .dockerignore content."""

    @pytest.fixture()
    def dockerignore(self) -> str:
        path = ROOT / ".dockerignore"
        assert path.exists(), ".dockerignore not found"
        return path.read_text(encoding="utf-8")

    def test_excludes_git(self, dockerignore):
        assert ".git" in dockerignore

    def test_excludes_pycache(self, dockerignore):
        assert "__pycache__" in dockerignore

    def test_excludes_tests(self, dockerignore):
        assert "tests/" in dockerignore

    def test_excludes_env(self, dockerignore):
        assert ".env" in dockerignore

    def test_excludes_venv(self, dockerignore):
        assert "venv" in dockerignore or ".venv" in dockerignore

    def test_excludes_vectorstore(self, dockerignore):
        assert ".vectorstore" in dockerignore

    def test_excludes_egg_info(self, dockerignore):
        assert "egg-info" in dockerignore


# ---------------------------------------------------------------------------
# Cross-file consistency
# ---------------------------------------------------------------------------


class TestCrossFileConsistency:
    """Verify Docker files are consistent with each other and the project."""

    def test_dockerfile_port_matches_compose(self):
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        compose = yaml.safe_load(
            (ROOT / "docker-compose.yml").read_text(encoding="utf-8"),
        )
        # Both should reference port 8000
        assert "8000" in dockerfile
        api_ports = compose["services"]["api"]["ports"]
        assert any("8000" in str(p) for p in api_ports)

    def test_configs_not_in_dockerignore(self):
        """configs/ must NOT be ignored — the app needs it."""
        content = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        lines = [
            l.strip()
            for l in content.splitlines()
            if l.strip() and not l.strip().startswith("#")
        ]
        assert "configs/" not in lines

    def test_src_not_in_dockerignore(self):
        """src/ must NOT be ignored — it's the application code."""
        content = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        lines = [
            l.strip()
            for l in content.splitlines()
            if l.strip() and not l.strip().startswith("#")
        ]
        assert "src/" not in lines

    def test_scripts_not_in_dockerignore(self):
        """scripts/ must NOT be ignored — CLI tools are included."""
        content = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        lines = [
            l.strip()
            for l in content.splitlines()
            if l.strip() and not l.strip().startswith("#")
        ]
        assert "scripts/" not in lines

    def test_pyproject_exists(self):
        """pyproject.toml must exist — Dockerfile COPYs it."""
        assert (ROOT / "pyproject.toml").exists()

    def test_default_config_exists(self):
        """configs/default.yaml must exist — referenced in Dockerfile ENV."""
        assert (ROOT / "configs" / "default.yaml").exists()
