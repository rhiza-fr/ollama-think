# test_client.py

import pytest

# Corrected import: httpx uses ConnectError for connection issues
from ollama import ChatResponse, Message

from ollama_think import Client


@pytest.fixture
def mocked_client_deps(mocker):
    """
    A pytest fixture to mock the dependencies of the Client class.
    It mocks the Cache and the upstream OllamaClient.chat method.
    """
    # Mock the Cache class
    mock_cache_class = mocker.patch("ollama_think.client.Cache")
    mock_cache_instance = mock_cache_class.return_value
    # Default to a cache miss
    mock_cache_instance.get.return_value = None

    # Mock the upstream chat method
    mock_chat = mocker.patch("ollama_think.client.OllamaClient.chat")

    # Yield the necessary mock objects to the tests
    yield mock_cache_instance, mock_chat


def test_clear_cache_on_init(mocked_client_deps):
    """Test if the cache is cleared when clear_cache=True."""
    mock_cache_instance, _ = mocked_client_deps

    Client(clear_cache=True)

    mock_cache_instance.clear.assert_called_once()


def test_close_method(mocked_client_deps):
    """Test that the explicit close method calls the cache's close method."""
    mock_cache_instance, _ = mocked_client_deps

    client = Client()
    client.close()

    mock_cache_instance.close.assert_called_once()


def test_call_with_prompt_and_cache_miss(mocked_client_deps):
    """Test a standard call that results in a cache miss and stores the result."""
    mock_cache_instance, mock_chat = mocked_client_deps

    mock_chat.return_value = ChatResponse(
        model="llama2",
        created_at="",
        message=Message(role="assistant", content="Hello, world!"),
        done=True,
    )
    client = Client()
    response = client.call(model="llama2", prompt="Hello, world!")

    assert response.content == "Hello, world!"
    mock_cache_instance.get.assert_called_once()
    mock_chat.assert_called_once()
    mock_cache_instance.set.assert_called_once()


def test_call_with_cache_hit(mocked_client_deps):
    """Test that a call retrieves a response from the cache."""
    mock_cache_instance, mock_chat = mocked_client_deps

    cached_response = ChatResponse(
        model="llama2",
        created_at="",
        message=Message(role="assistant", content="Cached response"),
        done=True,
    )
    mock_cache_instance.get.return_value = cached_response
    client = Client()
    response = client.call(model="llama2", prompt="Cache me")

    assert response.content == "Cached response"
    mock_cache_instance.get.assert_called_once()
    mock_chat.assert_not_called()
    mock_cache_instance.set.assert_not_called()


def test_call_with_use_cache_false(mocked_client_deps):
    """Test that use_cache=False makes an API call and does not save to cache."""
    mock_cache_instance, mock_chat = mocked_client_deps

    mock_chat.return_value = ChatResponse(
        model="llama2",
        created_at="",
        message=Message(role="assistant", content="Fresh response"),
        done=True,
    )
    client = Client()
    client.call(model="llama2", prompt="No cache", use_cache=False)

    mock_cache_instance.get.assert_not_called()
    mock_chat.assert_called_once()
    mock_cache_instance.set.assert_not_called()


def test_stream_with_cache_miss(mocked_client_deps):
    """Test a standard stream that results in a cache miss and stores the result."""
    mock_cache_instance, mock_chat = mocked_client_deps

    mock_chat.return_value = iter(
        [
            ChatResponse(
                message=Message(role="assistant", content="Hello, "),
                done=False,
                model="l2",
                created_at="",
            ),
            ChatResponse(
                message=Message(role="assistant", content="world!"),
                done=True,
                model="l2",
                created_at="",
            ),
        ]
    )
    client = Client()
    responses = list(client.stream(model="llama2", prompt="Hello, world!"))

    assert len(responses) == 2
    assert responses[0].content == "Hello, "
    mock_cache_instance.get.assert_called_once()
    mock_chat.assert_called_once()
    mock_cache_instance.set.assert_called_once()


def test_load_config():
    path = "src/ollama_think/config.yaml"
    client = Client()
    client.load_config(path)
    assert client.config.enable_hacks is True

    path = "DOESNOTEXIST.yaml"
    client.load_config(path)
    assert client.config.enable_hacks is False
