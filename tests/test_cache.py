# test_cache.py
"""
Slow integration tests that verify the cache provides significant speedup.
These tests require a running Ollama instance.
"""

import tempfile
import time
import uuid

import pytest

from ollama_think import Client
from ollama_think.client import AsyncClient


def get_unique_prompt():
    """Generate a unique prompt to avoid cache collisions across test runs."""
    return f"What is 2 + 2? (test id: {uuid.uuid4()})"


def get_default_model(client: Client) -> str | None:
    """Get a model to test with, preferring smaller/faster models."""
    try:
        models = [m["model"] for m in client.list()["models"]]
    except Exception:
        return None

    if not models:
        return None

    # Prefer smaller models for faster tests
    preferred = ["qwen3:0.6b", "qwen3:1.7b", "qwen3:4b", "llama3.2:1b", "llama3.2:3b"]
    for pref in preferred:
        for model in models:
            if model.startswith(pref):
                return model

    # Fall back to any available model
    return models[0]


@pytest.fixture
def cache_test_client(host):
    """A client with a fresh temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        client = Client(host=host, cache_dir=tmpdir, clear_cache=True)
        yield client
        client.close()


@pytest.fixture
def async_cache_test_client(host):
    """An async client with a fresh temporary cache directory."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        client = AsyncClient(host=host, cache_dir=tmpdir, clear_cache=True)
        yield client
        # Close cache synchronously to release file handles on Windows
        client.cache.close()


@pytest.mark.slow
def test_call_cache_speedup(cache_test_client: Client, request):
    """Test that cached call responses are significantly faster than uncached ones."""
    model = request.config.getoption("--model") or get_default_model(cache_test_client)
    if model is None:
        pytest.skip("No models available on Ollama server")

    prompt = get_unique_prompt()

    # First call - uncached, should hit the model
    start = time.perf_counter()
    response1 = cache_test_client.call(model=model, prompt=prompt, think=False)
    first_call_time = time.perf_counter() - start

    # Second call - should be cached
    start = time.perf_counter()
    response2 = cache_test_client.call(model=model, prompt=prompt, think=False)
    cached_call_time = time.perf_counter() - start

    # Verify responses are the same
    assert response1.content == response2.content

    print("--------- CALL -----------")
    print("not cached:", first_call_time)
    print("    cached:", cached_call_time)
    # Cached call should be at least 10x faster, or under 0.1 seconds
    # (whichever is easier to achieve - model calls typically take 1+ seconds)
    assert cached_call_time < first_call_time / 10 or cached_call_time < 0.1, (
        f"Cached call ({cached_call_time:.3f}s) should be much faster than "
        f"uncached call ({first_call_time:.3f}s)"
    )


@pytest.mark.slow
def test_stream_cache_speedup(cache_test_client: Client, request):
    """Test that cached stream responses are significantly faster than uncached ones."""
    model = request.config.getoption("--model") or get_default_model(cache_test_client)
    if model is None:
        pytest.skip("No models available on Ollama server")

    prompt = get_unique_prompt()

    # First stream - uncached, should hit the model
    start = time.perf_counter()
    chunks1 = list(cache_test_client.stream(model=model, prompt=prompt, think=False))
    first_stream_time = time.perf_counter() - start

    # Second stream - should be cached
    start = time.perf_counter()
    chunks2 = list(cache_test_client.stream(model=model, prompt=prompt, think=False))
    cached_stream_time = time.perf_counter() - start

    print("--------- STREAM ----------")
    print("not cached:", first_stream_time)
    print("    cached:", cached_stream_time)

    # Verify responses contain the same content
    content1 = "".join(chunk.content for chunk in chunks1)
    content2 = "".join(chunk.content for chunk in chunks2)
    assert content1 == content2
    assert len(chunks1) == len(chunks2)

    # Cached stream should be at least 10x faster, or under 0.1 seconds
    assert cached_stream_time < first_stream_time / 10 or cached_stream_time < 0.1, (
        f"Cached stream ({cached_stream_time:.3f}s) should be much faster than "
        f"uncached stream ({first_stream_time:.3f}s)"
    )


@pytest.mark.slow
@pytest.mark.asyncio
async def test_async_call_cache_speedup(async_cache_test_client: AsyncClient, request):
    """Test that cached async call responses are significantly faster than uncached ones."""
    # Need to get model - create a sync client temporarily
    host = async_cache_test_client.host or "http://localhost:11434"
    sync_client = Client(host=host)
    model = request.config.getoption("--model") or get_default_model(sync_client)
    sync_client.close()

    if model is None:
        pytest.skip("No models available on Ollama server")

    prompt = get_unique_prompt()

    # First call - uncached, should hit the model
    start = time.perf_counter()
    response1 = await async_cache_test_client.call(model=model, prompt=prompt, think=False)
    first_call_time = time.perf_counter() - start

    # Second call - should be cached
    start = time.perf_counter()
    response2 = await async_cache_test_client.call(model=model, prompt=prompt, think=False)
    cached_call_time = time.perf_counter() - start

    print("--------- ASYNC CALL ----------")
    print("not cached:", first_call_time)
    print("    cached:", cached_call_time)

    # Verify responses are the same
    assert response1.content == response2.content

    # Cached call should be at least 10x faster, or under 0.1 seconds
    assert cached_call_time < first_call_time / 10 or cached_call_time < 0.1, (
        f"Cached call ({cached_call_time:.3f}s) should be much faster than "
        f"uncached call ({first_call_time:.3f}s)"
    )


@pytest.mark.slow
@pytest.mark.asyncio
async def test_async_stream_cache_speedup(async_cache_test_client: AsyncClient, request):
    """Test that cached async stream responses are significantly faster than uncached ones."""
    # Need to get model - create a sync client temporarily
    host = async_cache_test_client.host or "http://localhost:11434"
    sync_client = Client(host=host)
    model = request.config.getoption("--model") or get_default_model(sync_client)
    sync_client.close()

    if model is None:
        pytest.skip("No models available on Ollama server")

    prompt = get_unique_prompt()

    # First stream - uncached, should hit the model
    start = time.perf_counter()
    chunks1 = [
        chunk
        async for chunk in async_cache_test_client.stream(model=model, prompt=prompt, think=False)
    ]
    first_stream_time = time.perf_counter() - start

    # Second stream - should be cached
    start = time.perf_counter()
    chunks2 = [
        chunk
        async for chunk in async_cache_test_client.stream(model=model, prompt=prompt, think=False)
    ]
    cached_stream_time = time.perf_counter() - start

    print("--------- ASYNC STREAM ----------")
    print("not cached:", first_stream_time)
    print("    cached:", cached_stream_time)

    # Verify responses contain the same content
    content1 = "".join(chunk.content for chunk in chunks1)
    content2 = "".join(chunk.content for chunk in chunks2)
    assert content1 == content2
    assert len(chunks1) == len(chunks2)

    # Cached stream should be at least 10x faster, or under 0.1 seconds
    assert cached_stream_time < first_stream_time / 10 or cached_stream_time < 0.1, (
        f"Cached stream ({cached_stream_time:.3f}s) should be much faster than "
        f"uncached stream ({first_stream_time:.3f}s)"
    )
