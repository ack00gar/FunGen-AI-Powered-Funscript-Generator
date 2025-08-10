import pytest

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "hardware: marks tests that require specific hardware")
    config.addinivalue_line("markers", "gui: marks tests that require GUI interaction")
    config.addinivalue_line("markers", "cli: marks tests for CLI functionality")
    config.addinivalue_line("markers", "ui_state: marks tests for UI state management")


def pytest_addoption(parser):
    """Add command line options for test execution."""
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="run end-to-end tests"
    )