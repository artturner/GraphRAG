"""Root-level conftest — registers shared CLI options and markers."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--run-e2e`` command-line flag."""
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Run end-to-end tests (skipped by default)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip e2e-marked tests unless ``--run-e2e`` is passed."""
    if config.getoption("--run-e2e"):
        return

    skip_e2e = pytest.mark.skip(reason="need --run-e2e option to run")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip_e2e)
