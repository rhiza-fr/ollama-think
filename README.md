# Ollama Python Client with Caching and Thinking

This project provides a Python client for the Ollama API, extending the official `ollama-python` library with additional features like caching and a "thinking" response mode.

## Features

- **Caching**: Automatically caches responses to speed up repeated requests.
- **Thinking Mode**: A `think` parameter to observe the model's reasoning process.
- **Streaming and Non-streaming**: Supports both streaming and non-streaming responses.
- **Easy to Use**: Simple and intuitive API that builds on the official Ollama client.

## Installation

```bash
pip install ollamapp
```

## Usage

Here's a simple example of how to use the `OllamaClient`:

```python
from ollama_think.client import OllamaClient

# Initialize the client
client = OllamaClient()

# Make a non-streaming call
response = client.call(model="qwen3", prompt="Hello, world!")
print(response.content)

# Make a streaming call
stream = client.stream(model="qwen3", prompt="Hello, how are you?")
for chunk in stream:
    print(chunk.content, end="")
```

### Thinking Mode

The `think` parameter allows you to see the model's "thinking" process. When `think=True`, the response object will contain both `thinking` and `content` attributes.

```python
# Non-streaming call with think=True
thinking, content = client.call(model="qwen3", prompt="Why is the sky blue?", think=True)
print("Thinking:", thinking)
print("Content:", content)

# Streaming call with think=True
stream = client.stream(model="qwen3", prompt="Why is the sky blue?", think=True)
for thinking, content in stream:
    print("Thinking:", thinking, end="")
    print("Content:", content, end="")
```

### Caching

The client automatically caches responses to avoid re-generating them for the same request. You can disable this behavior by setting `use_cache=False`.

```python
# This call will be cached
response1 = client.call(model="qwen3", prompt="Hello, world!")

# This call will use the cached response
response2 = client.call(model="qwen3", prompt="Hello, world!")

# This call will not use the cache
response3 = client.call(model="qwen3", prompt="Hello, world!", use_cache=False)
```

You can also clear the cache by passing `clear_cache=True` when initializing the client:

```python
client = OllamaClient(clear_cache=True)
```

## API Reference

### `OllamaClient`

- `__init__(self, host: str | None = None, cache_dir=".ollama_cache", clear_cache: bool = False)`
- `call(self, model: str = "", prompt: str | None = None, messages: Sequence[Mapping[str, Any] | Message] | None = None, tools: Sequence[Tool] | None = None, think: bool = False, format: JsonSchemaValue | Literal["", "json"] | None = None, options: Mapping[str, Any] | Options | None = None, keep_alive: float | str | None = None, use_cache: bool = True) -> ThinkResponse`
- `stream(self, model: str = "", prompt: str | None = None, messages: Sequence[Mapping[str, Any] | Message] | None = None, tools: Sequence[Tool] | None = None, think: bool = True, format: JsonSchemaValue | Literal["", "json"] | None = None, options: Mapping[str, Any] | Options | None = None, keep_alive: float | str | None = None, use_cache: bool = True) -> Iterator[ThinkResponse]`

### `ThinkResponse`

- `thinking` (property): The thinking content from the message.
- `content` (property): The content from the message.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
