import hashlib
from typing import Any, Iterator, Literal, Mapping, Sequence, cast

from diskcache import Cache
from ollama import ChatResponse
from ollama import Client as UpStreaamOllamaClient
from ollama._types import ChatRequest, Message, Options, Tool
from pydantic.json_schema import JsonSchemaValue
from rich import print

from ollamapp.thinkresponse import ThinkResponse


class OllamaClient(UpStreaamOllamaClient):
    """
    Ollama client for interacting with the Ollama API.

    This client supports both non-streaming and streaming generation and chat functionalities.
    """

    def __init__(self, host: str | None = None, cache_dir=".ollama_cache", clear_cache: bool = False) -> None:
        self.cache = Cache(cache_dir)
        if clear_cache:
            self.cache.clear()
        super().__init__(host=host)

    def close(self):
        """
        Clean up the cache.
        """
        self.cache.close()

    def __del__(self):
        """
        Clean up the cache when the OllamaClient instance is deleted.
        """
        self.close()

    def _make_cache_key(self, request: ChatRequest) -> str:
        """
        Create a cache key by hashing the request payload.
        """
        str_key = request.model_dump_json()
        return hashlib.md5(str_key.encode()).hexdigest()

    def call(
        self,
        model: str = "",
        prompt: str | None = None,
        messages: Sequence[Mapping[str, Any] | Message] | None = None,
        tools: Sequence[Tool] | None = None,
        think: bool = False,
        format: JsonSchemaValue | Literal["", "json"] | None = None,
        options: Mapping[str, Any] | Options | None = None,
        keep_alive: float | str | None = None,
        use_cache: bool = True,
    ) -> ThinkResponse:
        """
        Chat with the model using a list of messages.

        Args:
            model: The model to use for chatting.
            messages: A list of messages in the format [{"role": "user", "content": "Hello"}].
            stream: Whether to stream the response or not.
            **kwargs: Additional parameters for the Ollama API (e.g., temperature, seed).

        Returns:
            A response object or a generator if streaming is enabled.
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
            tools=tools,
            think=think,
        )
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

        return ThinkResponse(response)

    def stream(
        self,
        model: str = "",
        prompt: str | None = None,
        messages: Sequence[Mapping[str, Any] | Message] | None = None,
        tools: Sequence[Tool] | None = None,
        think: bool = True,
        format: JsonSchemaValue | Literal["", "json"] | None = None,
        options: Mapping[str, Any] | Options | None = None,
        keep_alive: float | str | None = None,
        use_cache: bool = True,
    ) -> Iterator[ThinkResponse]:
        """
        Chat with the model using a list of messages.

        Args:
            model: The model to use for chatting.
            messages: A list of messages in the format [{"role": "user", "content": "Hello"}].
            stream: Whether to stream the response or not.
            **kwargs: Additional parameters for the Ollama API (e.g., temperature, seed).

        Returns:
            A response object or a generator if streaming is enabled.
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
            tools=tools,
            think=think,
        )
        hash_key = self._make_cache_key(request)

        response = None
        if use_cache:
            response = self.cache.get(hash_key, None)
        if response:
            response = cast(list[ChatResponse], response)
            yield from (ThinkResponse(chunk) for chunk in response)
        else:
            chunks: list[ChatResponse] = []
            for chunk in super().chat(**request.__dict__):
                chunks.append(chunk)
                yield ThinkResponse(chunk)
            if use_cache:
                self.cache.set(hash_key, chunks, tag=model)


def examples():
    """Display the return objects of the Ollama client methods."""
    client = OllamaClient()

    print("----- client.call with think=False -----")
    resp = client.call(model="qwen3", prompt="Hello, world!", think=False)
    print(f"Thinking: {resp.thinking}")
    print(f" Content: {resp}")  # resp.content is the same
    print("-----------------------------------------")

    print("----- client.call with think=True -----")
    resp = client.call(model="qwen3", prompt="Hello, world!", think=True)
    print(f"Thinking: {resp.thinking}")
    print(f" Content: {resp}")
    print("-----------------------------------------")

    print("----- unpacking client.call -----")
    thinking, content = client.call(model="qwen3", prompt="Hello, world!", think=True)
    print(f"Thinking: {thinking}")
    print(f" Content: {content}")
    print("-----------------------------------------")

    print("----- client.stream with think=False -----")
    stream = client.stream(
        model="qwen3",
        prompt="Hello, how are you?",
        think=False,
    )
    for chunk in stream:
        print(f"{chunk.thinking}", end="")  # Chunk.thinking it "" when chunk.message.thinking is None
        print(f"{chunk.content}", end="")
    print("\n-----------------------------------------")

    print("----- client.stream with think=True -----")
    stream = client.stream(
        model="qwen3",
        prompt="Hello, how are you?",
        think=True,
    )
    chunk: ThinkResponse | None = None
    for chunk in stream:
        print(f"{chunk.thinking}", end="")
        print(f"{chunk.content}", end="")
    print("\n-----------------------------------------")
    print("Last chunk:")
    print(chunk)

    print("----- unpacking stream -----")
    stream = client.stream(
        model="qwen3",
        prompt="Hello, how are you?",
        think=True,
    )
    for thinking, content in stream:
        print(f"{thinking}", end="")
        print(f"{content}", end="")
    print("\n-----------------------------------------")


def main():
    examples()


if __name__ == "__main__":
    main()
