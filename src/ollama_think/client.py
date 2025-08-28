"""
A client for interacting with the Ollama API, extending the base `ollama.Client`.

This module provides a `Client` class that adds several key features:
- Caching of API responses to improve performance and reduce redundant calls.
- A `ThinkResponse` wrapper for easier handling of model outputs, including 'thinking' steps.
- `call` and `stream` methods with a consistent interface for both blocking and streaming interactions.
- hacks older models to respect thinking separation where possible
"""

import hashlib
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from typing import Any, Literal, cast

from diskcache import Cache
from ollama import AsyncClient as OllamaAsyncClient
from ollama import ChatResponse
from ollama import Client as OllamaClient
from ollama._client import _copy_tools
from ollama._types import ChatRequest, GenerateResponse, Message, Options, Tool
from pydantic.json_schema import JsonSchemaValue

from ollama_think.config import Config
from ollama_think.thinking_hacks import (
    hack_request,
    hack_response,
    hack_stream_chunk,
    setup_stream_parser,
)
from ollama_think.thinkresponse import ThinkResponse


class Client(OllamaClient):
    """
    An enhanced Ollama client with built-in caching and response processing.

    This client inherits from `ollama.Client` and extends its functionality. It automatically
    caches responses to disk using `diskcache`, which can be configured or disabled on a
    per-call basis.

    Methods like `call` and `stream` return a `ThinkResponse` object (or an iterator of them),
    which provides convenient access to the model's content and 'thinking' process.

    The client should be closed when no longer in use to clean up the cache connection.
    This can be done by calling `close()` or by using the client as a context manager.

    Example:
        ```python
        with Client() as client:
            response = client.call(model="llama3", prompt="Why is the sky blue?")
            print(response.content)
        ```
    """

    def __init__(
        self,
        host: str | None = None,
        cache_dir: str = ".ollama_cache",
        clear_cache: bool = False,
    ) -> None:
        """
        Initializes the Ollama Think Client.

        This sets up the connection to the Ollama server and initializes the caching system.

        Args:
            host: The URL of the Ollama API server. If not provided, it defaults to the value of
                  the `OLLAMA_HOST` environment variable or `http://localhost:11434` if the
                  variable is not set.
            cache_dir: The directory where API responses will be cached. Defaults to `.ollama_cache`
                       in the current working directory.
            clear_cache: If True, the entire cache in `cache_dir` will be cleared upon
                         initialization. Defaults to False.

        Examples:
            Default initialization, using environment variables or the default host:

            .. code-block:: python

                client = Client()

            Connecting to a remote Ollama server:

            .. code-block:: python

                client = Client(host="http://192.168.1.50:11434")

            Using a custom cache directory and clearing it on startup:

            .. code-block:: python

                client = Client(cache_dir="~/.my_app_cache/ollama", clear_cache=True)
        """
        self.cache = Cache(directory=cache_dir)
        if clear_cache:
            self.cache.clear()
        self.config = Config()
        self.host = host
        super().__init__(host=host)

    def close(self):
        """
        Explicitly clean up the cache.

        Most often this will be done for you.
        """
        self.cache.close()

    def __enter__(self):
        """Enter the runtime context related to this object."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the runtime context and close the cache."""
        self.close()

    def __del__(self) -> None:
        """
        Clean up the cache when the Client instance is deleted.

        Try hard to clean up even if Diskcache is forgiving.
        """
        self.close()

    def _make_cache_key(self, request: ChatRequest) -> str:
        """
        Create a cache key by hashing the request payload.
        """
        str_key = request.model_dump_json() + f"{self.host or 'default'}"
        return hashlib.md5(str_key.encode()).hexdigest()

    def call(
        self,
        model: str = "",
        prompt: str | None = None,
        messages: Sequence[Mapping[str, Any] | Message] | None = None,
        tools: Sequence[Mapping[str, Any] | Tool | Callable] | None = None,
        think: bool | Literal['low', 'medium', 'high'] = False,
        format: JsonSchemaValue | Literal["", "json"] | None = None,
        options: Mapping[str, Any] | Options | None = None,
        keep_alive: float | str | None = None,
        use_cache: bool = True,
    ) -> ThinkResponse:
        """
        A non-streaming chat with the model.

        This method sends a request to the Ollama API and waits for the entire response.
        It supports caching to avoid repeated API calls for the same request.

        Args:
            model: The model name.
            prompt: A single user prompt. If provided, it will be converted to a `messages` list.
            messages: A list of messages in the chat.
            tools: A list of tools the model may call.
            think: If True, enables thinking mode in the response.
            format: The format to return a response in. None | 'json' | your_obj.model_json_schema()
            options: Additional model parameter dict: {'temperature': 0.1, 'num_ctx': 8192}
            keep_alive: Controls how long the model will stay loaded in memory following the request.
            use_cache: If True, attempts to retrieve the response from cache before making an API call.
                       The result of a new API call will be cached. If False, bypasses the cache.

        Returns:
            A `ThinkResponse` object containing the full response from the model.

        Example:

        .. code-block:: python

            client = Client(host="http://localhost:11434")

            tr = client.call(model="qwen3", prompt="Hello, world!" think=False)  # 'think' is default False, which avoids pages of text for this example

            # We can now look at the response object and see that it looks exactly the same as a ChatResponse object
            print(tr)
            ThinkResponse(
                model='qwen3',
                created_at='2025-07-03T14:06:49.3206657Z',
                done=True,
                done_reason='stop',
                total_duration=2142225800,
                load_duration=1819354700,
                prompt_eval_count=20,
                prompt_eval_duration=86665700,
                eval_count=14,
                eval_duration=235023100,
                message=Message(role='assistant', content='Hello, world! How can I assist you today? 😊', thinking=None, images=None, tool_calls=None)
            )

            # As normal, we can access the thinking and the content via the unaltered ChatResponse.message
            tr.message.thinking: 'None'
            tr.message.content: 'Hello, world! How can I assist you today? 😊'

            # but we also have some convenience methods
            tr.thinking: '' # if None this is an empty string
            tr.content: 'Hello, world! How can I assist you today? 😊'
            str(tr): 'Hello, world! How can I assist you today? 😊'
            f"{tr}": 'Hello, world! How can I assist you today? 😊'

            # For further convenience, we can unpack the ThinkResponse
            thinking, content = client.call(model="qwen3", prompt="Hello, world!")
            thinking: ''
            content: 'Hello, world! How can I assist you today? 😊'

        """
        if messages is None:
            if prompt is not None:
                messages = [Message(role="user", content=prompt)]
        request = ChatRequest(
            model=model,
            stream=False,
            options=options,
            format=format,
            keep_alive=keep_alive,
            messages=messages,
            tools=list(_copy_tools(tools)),
            think=think,
        )
        model_hacks = self.config.get_hacks_if_enabled(model)
        if model_hacks:
            request = hack_request(request, hacks=model_hacks)  # cludge ollama to respect thought
        hash_key = self._make_cache_key(request)
        response = None
        if use_cache:
            response = self.cache.get(hash_key, None)
        if response:
            response = cast(ChatResponse, response)
        else:
            response = super().chat(**request.__dict__)
            if use_cache:
                self.cache.set(hash_key, response, tag=model)
        tr = ThinkResponse(response)
        if model_hacks:
            tr = hack_response(tr, hacks=model_hacks)  # cludge ollama to respect thought
        return tr

    def stream(
        self,
        model: str = "",
        prompt: str | None = None,
        messages: Sequence[Mapping[str, Any] | Message] | None = None,
        tools: Sequence[Mapping[str, Any] | Tool | Callable] | None = None,
        think: bool | Literal['low', 'medium', 'high']  = True,
        format: JsonSchemaValue | Literal["", "json"] | None = None,
        options: Mapping[str, Any] | Options | None = None,
        keep_alive: float | str | None = None,
        use_cache: bool = True,
    ) -> Iterator[ThinkResponse]:
        """
        A streaming chat with the model.

        This method sends a request to the Ollama API and yields response chunks as they are received.
        It supports caching to avoid repeated API calls for the same request. When a response is retrieved
        from the cache, it will be yielded as if it were being streamed live.

        Args:
            model: The model name.
            prompt: A single user prompt. If provided, it will be converted to a `messages` list.
            messages: A list of messages in the chat.
            tools: A list of tools the model may call.
            think: If True, enables thinking mode in the response. (or 'high', 'medium', 'low' for gpt-oss)
            format: The format to return a response in. None | 'json' | your_obj.model_json_schema()
            options: Additional model parameter dict, such as {'temperature': 0.1, 'num_ctx': 8192}
            keep_alive: Controls how long the model will stay loaded in memory following the request.
            use_cache: If True, attempts to retrieve the response from cache. If not found, a new API
                       call is made, and the full list of streamed chunks is cached upon completion.
                       If False, bypasses the cache.

        Returns:
            An iterator of `ThinkResponse` objects, each containing a chunk of the response from the model.

        Example:

        .. code-block:: python

            client = Client(host="http://localhost:11434")
            for thinking, content in client.stream(model="qwen3", prompt="Why is the sky blue?", thinking=True):
                print(thinking, end="") # the thinking always comes first
                print(content, end="")


        """
        if messages is None:
            if prompt is not None:
                messages = [Message(role="user", content=prompt)]
        request = ChatRequest(
            model=model,
            stream=True,
            options=options,
            format=format,
            keep_alive=keep_alive,
            messages=messages,
            tools=list(_copy_tools(tools)),
            think=think,
        )
        model_hacks = self.config.get_hacks_if_enabled(model)
        if model_hacks:
            request = hack_request(request, hacks=model_hacks)  # cludge ollama to respect thought
        hash_key = self._make_cache_key(request)

        response = None
        if use_cache:
            response = self.cache.get(hash_key, None)
        if response:
            response = cast(list[ThinkResponse], response)
            yield from response
        else:
            hack_parser = setup_stream_parser(
                model, hacks=model_hacks
            )  # will be None if no hacks are required
            chunks: list[ThinkResponse] = []  # we will cache this list, when finished
            for chunk in super().chat(**request.__dict__):
                tr = ThinkResponse(chunk)
                if hack_parser:
                    tr = hack_stream_chunk(tr, hack_parser)
                    if not tr:  # we consumed a non-output chunk like a <think> tag
                        continue
                chunks.append(tr)
                yield tr
            if use_cache:
                self.cache.set(hash_key, chunks, tag=model)

    def stop(self, model: str = "") -> GenerateResponse:
        """
        Unloads a model from memory.

        This is a convenience method that calls the `generate` endpoint with a `keep_alive`
        of 0. This instructs Ollama to unload the specified model from memory. If no model
        is specified, all models will be unloaded.

        Args:
            model: The name of the model to unload. If empty, all models will be unloaded.

        Returns:
            A `GenerateResponse` object from the underlying API call.
        """
        return super().generate(model=model, keep_alive=0.0)

    def load_config(self, path: str) -> None:
        self.config.load_config(path)


class AsyncClient(OllamaAsyncClient):
    def __init__(
        self,
        host: str | None = None,
        cache_dir: str = ".ollama_cache",
        clear_cache: bool = False,
    ) -> None:
        self.cache = Cache(directory=cache_dir)
        if clear_cache:
            self.cache.clear()
        self.config = Config()
        self.host = host
        super().__init__(host=host)

    async def close(self):
        self.cache.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    def __del__(self) -> None:
        try:
            self.cache.close()
        except Exception:
            pass

    def _make_cache_key(self, request: ChatRequest) -> str:
        str_key = request.model_dump_json() + f"{self.host or 'default'}"
        return hashlib.md5(str_key.encode()).hexdigest()

    async def call(
        self,
        model: str = "",
        prompt: str | None = None,
        messages: Sequence[Mapping[str, Any] | Message] | None = None,
        tools: Sequence[Mapping[str, Any] | Tool | Callable] | None = None,
        think: bool | Literal['low', 'medium', 'high']  = False,
        format: JsonSchemaValue | Literal["", "json"] | None = None,
        options: Mapping[str, Any] | Options | None = None,
        keep_alive: float | str | None = None,
        use_cache: bool = True,
    ) -> ThinkResponse:
        if messages is None:
            if prompt is not None:
                messages = [Message(role="user", content=prompt)]
        request = ChatRequest(
            model=model,
            stream=False,
            options=options,
            format=format,
            keep_alive=keep_alive,
            messages=messages,
            tools=list(_copy_tools(tools)),
            think=think,
        )
        model_hacks = self.config.get_hacks_if_enabled(model)
        if model_hacks:
            request = hack_request(request, hacks=model_hacks)
        hash_key = self._make_cache_key(request)
        response = None
        if use_cache:
            response = self.cache.get(hash_key, None)
        if response:
            response = cast(ChatResponse, response)
        else:
            response = await super().chat(**request.__dict__)
            if use_cache:
                self.cache.set(hash_key, response, tag=model)
        tr = ThinkResponse(response)
        if model_hacks:
            tr = hack_response(tr, hacks=model_hacks)
        return tr

    async def stream(
        self,
        model: str = "",
        prompt: str | None = None,
        messages: Sequence[Mapping[str, Any] | Message] | None = None,
        tools: Sequence[Mapping[str, Any] | Tool | Callable] | None = None,
        think: bool | Literal['low', 'medium', 'high']  = True,
        format: JsonSchemaValue | Literal["", "json"] | None = None,
        options: Mapping[str, Any] | Options | None = None,
        keep_alive: float | str | None = None,
        use_cache: bool = True,
    ) -> AsyncIterator[ThinkResponse]:
        if messages is None:
            if prompt is not None:
                messages = [Message(role="user", content=prompt)]
        request = ChatRequest(
            model=model,
            stream=True,
            options=options,
            format=format,
            keep_alive=keep_alive,
            messages=messages,
            tools=list(_copy_tools(tools)),
            think=think,
        )
        model_hacks = self.config.get_hacks_if_enabled(model)
        if model_hacks:
            request = hack_request(request, hacks=model_hacks)
        hash_key = self._make_cache_key(request)

        response = None
        if use_cache:
            response = self.cache.get(hash_key, None)
        if response:
            response = cast(list[ThinkResponse], response)
            for r in response:
                yield r
        else:
            hack_parser = setup_stream_parser(model, hacks=model_hacks)
            chunks: list[ThinkResponse] = []
            response_iterator = await super().chat(**request.__dict__)
            async for chunk in response_iterator:
                tr = ThinkResponse(chunk)
                if hack_parser:
                    tr = hack_stream_chunk(tr, hack_parser)
                    if not tr:
                        continue
                chunks.append(tr)
                yield tr
            if use_cache:
                self.cache.set(hash_key, chunks, tag=model)

    async def stop(self, model: str = "") -> GenerateResponse:
        return await super().generate(model=model, keep_alive=0.0)

    def load_config(self, path: str) -> None:
        self.config.load_config(path)
