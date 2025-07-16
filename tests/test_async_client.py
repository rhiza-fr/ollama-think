# test_async_client.py

import pytest
from ollama import ChatResponse, Message

from ollama_think.client import AsyncClient


@pytest.fixture
def mocked_async_client_deps(mocker):
    """
    A pytest fixture to mock the dependencies of the AsyncClient class.
    It mocks the Cache and the upstream OllamaAsyncClient.chat method.
    """
    mock_cache_class = mocker.patch("ollama_think.client.Cache")
    mock_cache_instance = mock_cache_class.return_value
    mock_cache_instance.get.return_value = None

    mock_chat = mocker.patch("ollama_think.client.OllamaAsyncClient.chat")

    async def async_chat_generator(*args, **kwargs):
        for chunk in kwargs.get("chunks", []):
            yield chunk

    mock_chat.return_value = async_chat_generator()

    yield mock_cache_instance, mock_chat


@pytest.mark.asyncio
async def test_clear_cache_on_init(mocked_async_client_deps):
    """Test if the cache is cleared when clear_cache=True."""
    mock_cache_instance, _ = mocked_async_client_deps

    AsyncClient(clear_cache=True)

    mock_cache_instance.clear.assert_called_once()


@pytest.mark.asyncio
async def test_close_method(mocked_async_client_deps):
    """Test that the explicit close method calls the cache's close method."""
    mock_cache_instance, _ = mocked_async_client_deps

    client = AsyncClient()
    await client.close()

    mock_cache_instance.close.assert_called_once()


@pytest.mark.asyncio
async def test_call_with_prompt_and_cache_miss(mocked_async_client_deps):
    """Test a standard call that results in a cache miss and stores the result."""
    mock_cache_instance, mock_chat = mocked_async_client_deps

    mock_chat.return_value = ChatResponse(
        model="llama2",
        created_at="",
        message=Message(role="assistant", content="Hello, world!"),
        done=True,
    )
    client = AsyncClient()
    response = await client.call(model="llama2", prompt="Hello, world!")

    assert response.content == "Hello, world!"
    mock_cache_instance.get.assert_called_once()
    mock_chat.assert_called_once()
    mock_cache_instance.set.assert_called_once()


@pytest.mark.asyncio
async def test_call_with_cache_hit(mocked_async_client_deps):
    """Test that a call retrieves a response from the cache."""
    mock_cache_instance, mock_chat = mocked_async_client_deps

    cached_response = ChatResponse(
        model="llama2",
        created_at="",
        message=Message(role="assistant", content="Cached response"),
        done=True,
    )
    mock_cache_instance.get.return_value = cached_response
    client = AsyncClient()
    response = await client.call(model="llama2", prompt="Cache me")

    assert response.content == "Cached response"
    mock_cache_instance.get.assert_called_once()
    mock_chat.assert_not_called()
    mock_cache_instance.set.assert_not_called()


@pytest.mark.asyncio
async def test_call_with_use_cache_false(mocked_async_client_deps):
    """Test that use_cache=False makes an API call and does not save to cache."""
    mock_cache_instance, mock_chat = mocked_async_client_deps

    mock_chat.return_value = ChatResponse(
        model="llama2",
        created_at="",
        message=Message(role="assistant", content="Fresh response"),
        done=True,
    )
    client = AsyncClient()
    await client.call(model="llama2", prompt="No cache", use_cache=False)

    mock_cache_instance.get.assert_not_called()
    mock_chat.assert_called_once()
    mock_cache_instance.set.assert_not_called()


@pytest.mark.asyncio
async def test_stream_with_cache_miss(mocked_async_client_deps):
    """Test a standard stream that results in a cache miss and stores the result."""
    mock_cache_instance, mock_chat = mocked_async_client_deps

    async def async_iterator():
        yield ChatResponse(
            message=Message(role="assistant", content="Hello, "),
            done=False,
            model="l2",
            created_at="",
        )
        yield ChatResponse(
            message=Message(role="assistant", content="world!"),
            done=True,
            model="l2",
            created_at="",
        )

    mock_chat.return_value = async_iterator()
    client = AsyncClient()
    responses = [resp async for resp in client.stream(model="llama2", prompt="Hello, world!")]

    assert len(responses) == 2
    assert responses[0].content == "Hello, "
    mock_cache_instance.get.assert_called_once()
    mock_chat.assert_called_once()
    mock_cache_instance.set.assert_called_once()


def test_load_config():
    path = "src/ollama_think/config.yaml"
    client = AsyncClient()
    client.load_config(path)
    assert client.config.enable_hacks is True

    path = "DOESNOTEXIST.yaml"
    client.load_config(path)
    assert client.config.enable_hacks is False
