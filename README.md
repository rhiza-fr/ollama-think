# Ollama-Think Library

This project provides a Python client for the Ollama API, extending the official `ollama-python` library with the addition of caching and a generous dose of syntax sugar.

## Features

- **Caching**: Automatically caches responses to speed up repeated requests.
- **Thinking Mode**: A `think` parameter to observe the model's reasoning process.
- **Streaming and Non-streaming**: Separates the underlying streaming and non-streaming interface to provide clean type hints.
- **Syntax Sugar**: Less boiler-plate, so that you can maintain your flow.

## Install

```bash
pip install ollama-think
```
or
```bash
poetry add ollama-think
```
or
```bash
uv add ollama-think
```

## Usage

### Initialization

You can initialize the client with default settings, which will look for the `OLLAMA_HOST` environment variable or default to `http://localhost:11434`.


```python
from ollama_think.client import Client

# Initialize with default settings
client = Client()
```

You can provide explicit settings for the host, cache directory, and whether to clear the cache on startup.

```python
# Initialize with custom settings
client = Client(
    host="http://localhost:11434",
    cache_dir=".ollama_cache",
    clear_cache=False
)
```

### Making Calls

The `call` method provides a strongly typed access to the underlying `Chat` method in non-streaming mode. It returns a `ThinkResponse` object which is a subclass of `ollama.ChatResponse` and adds some convenience properties.

```python
# Make a non-streaming call
response = client.call(model="qwen3", prompt="Hello, world!")

# The response object contains all the original data from the Ollama ChatResponse
print(response)
ThinkResponse(
    model='qwen3',
    created_at='2025-07-03T14:16:05.8452406Z',
    done=True,
    done_reason='stop',
    total_duration=2461619200,
    load_duration=2111438400,
    prompt_eval_count=20,
    prompt_eval_duration=78409600,
    eval_count=16,
    eval_duration=271104600,
    message=Message(role='assistant', content='Hello, world! 🌍✨ How can I assist you today?', thinking=None, images=None, tool_calls=None)
)

# As normal, we can access the thinking and the content via the unaltered ChatResponse.message
print(response.message.thinking) # this is empty because we used the default think=False
# None
print(response.message.content)
# 'Hello, world! 🌍✨ How can I assist you toda

# For convenience, you can access the content directly
print(response.thinking)
# '' - an empty string
print(response.content)
# 'Hello, world! ...'

# The response object can be used as a string which will show just the 'content'
print(f"The model said: {response}")
# The model said: Hello, world! ...

# For further convenience, you can unpack the response into thinking and content
thinking, content = response
print(f"Thinking: {thinking}, Content: {content}")
```

### Streaming

The `stream` method provides a strongly typed access to the underlying `Chat` method in streaming mode. It returns a an iterator of `ThinkResponse` chunks

```python
# Make a streaming call
stream = client.stream(model="qwen3", prompt="Tell me a short story.")
for chunk in stream:
    print(chunk.thinking, end="")
    print(chunk.content, end="")
```

### Thinking Mode

The `think` parameter allows you to see the model's "thinking" process. When `think=True`, the `thinking` attribute of the response will be populated.

Note: Not all models officially support thinking. They will return an error even if they have '&lt;think&gt;' tags in their content. It would be possible to intercept this just as ollama probably does under the hood, but this is not implemented here.

```python
# Non-streaming call with think=True
thinking, content = client.call(model="qwen3", prompt="Why is the sky blue?", think=True)
print("--- Thinking ---")
print(thinking)
print("\n--- Content ---")
print(content)

# Streaming call with think=True
stream = client.stream(model="qwen3", prompt="Why is the sky blue?", think=True)
for thinking_chunk, content_chunk in stream:
    print(thinking_chunk, end="") # The 'thinking' part usually comes first
    print(content_chunk, end="")
```

### Caching

The client automatically caches responses using the light-weight DiskCache library to avoid re-generating them for the same request. You can disable this behavior by setting `use_cache=False`.


```python
# This call will be cached
response1 = client.call(model="qwen3", prompt="Hello, world!") # 0.31 seconds

# This call will use the cached response
response2 = client.call(model="qwen3", prompt="Hello, world!") # 0.0001 seconds

# This call will not attempt to get from the cache and will not store the result
response3 = client.call(model="qwen3", prompt="Hello, world!", use_cache=False)
```

You can clear the cache by passing `clear_cache=True` when initializing the client:

```python
client = Client(clear_cache=True)
```

### Options

### Tool Calling

### Response Formats


### Access to the underlying ollama client

Since the `ollama-think.client` is a thin wrapper around the `ollama.client`, you can still access the all the underlying ollama client methods.

```python
from ollama-think.client import Client
from ollama import ChatResponse

client = Client()
response: ChatResponse = client.chat(model='llama3.2', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response['message']['content'])

# but this seems so much easier
print(client.call(model='llama3.2', prompt='Why is the sky blue?'))

```

### Prompts and Messages

```python
# the prompt parameter in `call` and `stream` is just a shortcut for
prompt = 'Why is the sky blue?'
message =  {'role': 'user', 'content': prompt}
client.call(model='llama3.2', messages=[message])
```

## API Reference

### `Client`

- `__init__(self, host: str | None = None, cache_dir=".ollama_cache", clear_cache: bool = False)`

- `call(self, model: str = "", prompt: str | None = None, messages: Sequence[Mapping[str, Any] | Message] | None = None, tools: Sequence[Tool] | None = None, think: bool = False, format: JsonSchemaValue | Literal["", "json"] | None = None, options: Mapping[str, Any] | Options | None = None, keep_alive: float | str | None = None, use_cache: bool = True) -> ThinkResponse`

- `stream(self, model: str = "", prompt: str | None = None, messages: Sequence[Mapping[str, Any] | Message] | None = None, tools: Sequence[Tool] | None = None, think: bool = True, format: JsonSchemaValue | Literal["", "json"] | None = None, options: Mapping[str, Any] | Options | None = None, keep_alive: float | str | None = None, use_cache: bool = True) -> Iterator[ThinkResponse]`

### `ThinkResponse`

- `thinking` : str - The thinking content from the message.
- `content` : str - The content from the message.

#### Inherited fields:

- `model`: str Model used to generate response.
- `created_at`: str - Time when the request was created.
-  `done`: bool - True if response is complete, otherwise False. Useful for streaming to detect the final response.
- `done_reason`: str - Reason for completion. Only present when done is True.
- `total_duration`: int - Total duration in nanoseconds.
- `load_duration`: int - Load duration in nanoseconds.
- `prompt_eval_count`: int - Number of tokens evaluated in the prompt.
- `prompt_eval_duration`: int - Duration of evaluating the prompt in nanoseconds.
- `eval_count`: int - Number of tokens evaluated in inference.
- `eval_duration`: int - Duration of evaluating inference in nanoseconds.
- `message`
    - `role`: str - Assumed role of the message. Response messages has role 'assistant' or 'tool'.
    - `content`: str - Content of the message. Response messages contains message fragments when streaming.
    - `thinking`: str - Thinking content. Only present when thinking is enabled.'
    - `images`: Sequence[Image] - List of image data for multimodal models.
    - `tool_calls`: Sequence[ToolCall]  -Tools calls to be made by the model.

## Credit to

- ollama https://ollama.com/
- ollama-python https://github.com/ollama/ollama-python
- diskcache https://github.com/grantjenks/python-diskcache/

## Reference docs

- Ollama Thinking - https://ollama.com/blog/thinking
- Ollama Tool support - https://ollama.com/blog/tool-support
- Ollama Structured Outputs - https://ollama.com/blog/structured-outputs
- Ollama Options - https://github.com/ollama/ollama-python/blob/main/ollama/_types.py



## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
