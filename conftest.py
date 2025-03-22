import pytest
import os

# Configuration globale pour pytest
def pytest_addoption(parser):
    parser.addoption(
        "--run-interactive", action="store_true", default=False, help="run interactive tests"
    )
    parser.addoption(
        "--run-visual", action="store_true", default=False, help="run visual tests"
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "interactive: mark test as interactive")
    config.addinivalue_line("markers", "visual: mark test as visual")

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-interactive"):
        skip_interactive = pytest.mark.skip(reason="need --run-interactive option to run")
        for item in items:
            if "interactive" in item.keywords:
                item.add_marker(skip_interactive)

    if not config.getoption("--run-visual"):
        skip_visual = pytest.mark.skip(reason="need --run-visual option to run")
        for item in items:
            if "visual" in item.keywords:
                item.add_marker(skip_visual)