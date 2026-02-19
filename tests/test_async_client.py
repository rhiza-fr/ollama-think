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
    mock_cache_instance.get.return_value = cached_response.model_dump(mode='json')
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


@pytest.mark.asyncio
async def test_async_context_manager(mocked_async_client_deps):
    """__aenter__ returns self, __aexit__ closes the cache."""
    mock_cache_instance, _ = mocked_async_client_deps
    async with AsyncClient() as client:
        assert isinstance(client, AsyncClient)
    mock_cache_instance.close.assert_called()


def test_async_client_del_handles_exception(mocked_async_client_deps):
    """__del__ swallows exceptions from cache.close()."""
    mock_cache_instance, _ = mocked_async_client_deps
    mock_cache_instance.close.side_effect = Exception("already closed")
    client = AsyncClient()
    client.__del__()  # must not raise


@pytest.mark.asyncio
async def test_stop(mocked_async_client_deps, mocker):
    """stop() calls generate with keep_alive=0."""
    _, _ = mocked_async_client_deps
    mock_generate = mocker.patch("ollama_think.client.OllamaAsyncClient.generate")
    mock_generate.return_value = None
    await AsyncClient().stop("llama2")
    mock_generate.assert_called_once_with(model="llama2", keep_alive=0.0)


@pytest.mark.asyncio
async def test_call_cache_read_error_falls_back_to_api(mocked_async_client_deps):
    """A corrupt cache entry is silently ignored and a fresh API call is made."""
    mock_cache_instance, mock_chat = mocked_async_client_deps
    mock_cache_instance.get.side_effect = Exception("corrupt json")
    mock_chat.return_value = ChatResponse(
        model="l2", created_at="", message=Message(role="assistant", content="fresh"), done=True
    )
    response = await AsyncClient().call(model="l2", prompt="test")
    assert response.content == "fresh"
    mock_chat.assert_called_once()


@pytest.mark.asyncio
async def test_call_applies_model_hacks(mocked_async_client_deps, mocker):
    """call() runs hack_request and hack_response when hacks are configured."""
    mock_cache_instance, mock_chat = mocked_async_client_deps
    mock_chat.return_value = ChatResponse(
        model="cogito",
        created_at="",
        message=Message(role="assistant", content="<think>a thought</think>the answer"),
        done=True,
    )
    hacks = {"enable_thinking": False, "add_message": None, "content_parsers": ["<think>(?P<thinking>.*?)</think>(?P<content>.*)"]}
    mocker.patch("ollama_think.client.Config.get_hacks_if_enabled", return_value=hacks)
    response = await AsyncClient().call(model="cogito:latest", prompt="test")
    assert response.thinking == "a thought"
    assert response.content == "the answer"


@pytest.mark.asyncio
async def test_stream_cache_hit(mocked_async_client_deps):
    """stream() yields cached chunks without calling the API."""
    mock_cache_instance, mock_chat = mocked_async_client_deps
    cached = ChatResponse(model="l2", created_at="", message=Message(role="assistant", content="cached!"), done=True)
    mock_cache_instance.get.return_value = [cached.model_dump(mode="json")]
    results = [r async for r in AsyncClient().stream(model="l2", prompt="test")]
    assert len(results) == 1
    assert results[0].content == "cached!"
    mock_chat.assert_not_called()


@pytest.mark.asyncio
async def test_stream_with_hack_parser_filters_none_chunks(mocked_async_client_deps, mocker):
    """stream() skips chunks where hack_stream_chunk returns None."""
    mock_cache_instance, mock_chat = mocked_async_client_deps

    async def async_chunks():
        yield ChatResponse(model="l2", created_at="", message=Message(role="assistant", content="<think>"), done=False)
        yield ChatResponse(model="l2", created_at="", message=Message(role="assistant", content="answer"), done=True)

    mock_chat.return_value = async_chunks()
    hacks = {"enable_thinking": False, "add_message": None, "content_parsers": ["<think>(?P<thinking>.*?)</think>(?P<content>.*)"]}
    mocker.patch("ollama_think.client.Config.get_hacks_if_enabled", return_value=hacks)
    mocker.patch("ollama_think.client.setup_stream_parser", return_value=mocker.MagicMock())
    answer_chunk = ChatResponse(model="l2", created_at="", message=Message(role="assistant", content="answer"), done=True)
    from ollama_think.thinkresponse import ThinkResponse
    mocker.patch("ollama_think.client.hack_stream_chunk", side_effect=[None, ThinkResponse(answer_chunk)])
    results = [r async for r in AsyncClient().stream(model="cogito:latest", prompt="test")]
    assert len(results) == 1
    assert results[0].content == "answer"
