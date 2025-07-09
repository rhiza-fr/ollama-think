import pytest

from ollama_think import Client


def pytest_addoption(parser):
    parser.addoption(
        "--host",
        action="store",
        default="http://localhost:11434",
        help="Ollama host to run tests against",
    )
    parser.addoption(
        "--model", action="store", default=None, help="run tests only on the specified model"
    )


@pytest.fixture(scope="session")
def host(request):
    """A fixture to provide the host URL to tests."""
    return request.config.getoption("--host")


@pytest.fixture(scope="session")
def client(host):
    """A session-scoped client fixture that uses the host fixture."""
    return Client(host=host)
