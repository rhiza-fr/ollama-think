# Ollama-Think Library

This project provides a Python client for the Ollama API, extending the official `ollama-python` library with the addition of caching, a generous dose of syntax sugar and increased `think` model compatibility.

## Features

- **Caching**: Automatically caches responses to speed up repeated requests.
- **Thinking**: Enables some officially unsupported models to use thinking mode.
- **Streaming and Non-streaming**: Separates the underlying streaming and non-streaming interface to provide clean type hints.
- **Syntax Sugar**: Less boiler-plate, so that you can maintain your flow.

## Install

```bash
pip install ollama-think
# or
poetry add ollama-think
# or
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

The `call` method provides strongly typed access to the underlying `Chat` method in non-streaming mode. It returns a `ThinkResponse` object which is a subclass of `ollama.ChatResponse` and adds some convenience properties. You can use `prompt` or `messages` as you prefer.

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
stream = client.stream(model="qwen3", prompt="Tell me a short story about italian chimpanzees and bananas")
for chunk in stream:
    print(chunk.thinking, end="") # empty, since think=False. Your choice.
    print(chunk.content, end="")
```

### Thinking Mode

The `think` parameter tells ollama to enable thinking for models that support this. For other models that use non-standard ways of enabling thinking we do the neccesary. See the default condiguration: [config.md](config.md)

Some models will think, even without 'enabling' thinking. This output is separated out of the `content` into `thinking`

See [Model Capabilities](model_capabilities.md)

Note: Not all models officially or unofficially support thinking. They will throw a `400` error if you try to enable thinking.

```python
# Non-streaming call with think=True
thinking, content = client.call(model="qwen3", prompt="Why is the sky red at night??", think=True)
print("--- Thinking ---")
print(thinking)
print("\n--- Content ---")
print(content)

# Streaming call with think=True
stream = client.stream(model="qwen3", prompt="What is bigger an egg or a mouse?", think=True)
for thinking_chunk, content_chunk in stream:
    print(thinking_chunk, end="")
    print(content_chunk, end="") # empty until thinking is finished for most models
```

### Caching

The client automatically caches responses using the light-weight `DiskCache` library to avoid re-generating them for the same request. You can disable this behavior by setting `use_cache=False`.

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

The `options` parameter of the underlying `chat` method can be used to change how the model
responds. The most commonly used parameters are

- `temperature` Low values keep the model deterministic, Higher values for more creativity Typically 0.1 -> 1.0
- `num_ctx` Ollama has a default context length of 2048, which can be increased if you have enough VRAM. If you send in more
than `num_ctx` tokens, ollama will silently truncate your message, which can lead to lost instructions.

```python
client = Client(host="http://localhost:11434")
prompt="Describe the earth to an alien who has just arrived."
options={'num_ctx': 8192, 'temperature': 0.9}

print("Using prompt:", prompt)
print("Using options:", options)

thinking, content = client.call(model="qwen3", prompt=prompt, think=True, options=options)
print("Thinking:", thinking)
print("Content:", content)
```

See [examples/options_example.py](examples/options_example.py) for a full list of options

### Tool Calling

Before, and underneath the concept of MCP servers are the humble tool_calls. By telling the model that you have a tool available,
the model can choose to reply with a special format that indicates that it wants to call a tool. Typically, this call is
intercepted, the tool is excecuted and the result sent back to the model. The model's second response can then be shown to a user.

See [examples/tool_calling_example.py](examples/tool_calling_example.py)

### Response Formats

Forcing JSON format can encourage some models to behave. It is usualy a good idea to mention JSON in the prompt.

```python
import json

text_json = client.call(
    model="qwen3",
    prompt="Design a json representation of a spiral galaxy",
    format="json",
).content

my_object = json.loads(text_json) # might explode if invalid json was returned
```

You can use pydantic models to describe more exactly the format you want.

```python
from pydantic import BaseModel, Field

class Heat(BaseModel):
    """A specially crafted response object to capture an iterpretation of heat"""
    reaoning: str = Field(..., description="your reasoning for the response")
    average_temperature: float = Field(..., description="average temperature")

text_obj = client.call(model="qwen3", prompt="How hot is the world?",
        format=Heat.model_json_schema()).content

my_obj = Heat.model_validate_json(text_obj) # might explode it the format is invalid
```

See [examples/response_format_example.py](examples/response_format_example.py)

### Access to the underlying ollama client

Since the `ollama_think.client` is a thin wrapper around the `ollama.client`, you can still access the all the underlying ollama client methods.

```python
from ollama_think.client import Client
from ollama import ChatResponse

client = Client()
response: ChatResponse = client.chat(model='llama3.2', messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response['message']['content'])
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

#### Inherited fields

- `model`: str Model used to generate response.
- `created_at`: str - Time when the request was created.
- `done`: bool - True if response is complete, otherwise False. Useful for streaming to detect the final response.
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

- ollama [https://ollama.com/]([https://ollama.com/)
- ollama-python [https://github.com/ollama/ollama-python]([https://github.com/ollama/ollama-python)
- diskcache [https://github.com/grantjenks/python-diskcache/]([https://github.com/grantjenks/python-diskcache/)
- pydantic [https://pydantic-docs.helpmanual.io/]([https://pydantic-docs.helpmanual.io/)

## Reference docs

- Ollama Thinking - [https://ollama.com/blog/thinking](https://ollama.com/blog/thinking)
- Ollama Tool support - [https://ollama.com/blog/tool-support]([https://ollama.com/blog/tool-support)
- Ollama Structured Outputs - [https://ollama.com/blog/structured-outputs]([https://ollama.com/blog/structured-outputs)
- Ollama Options - [https://github.com/ollama/ollama-python/blob/main/ollama/_types.py]([https://github.com/ollama/ollama-python/blob/main/ollama/_types.py)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

### Development Setup

This project uses `uv` for package management and `hatch` for task running.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ollama-think.git
    cd ollama-think
    ```

2.  **Create a virtual environment and install dependencies:**
    This command creates a virtual environment in `.venv` and installs all dependencies, including development tools.
    ```bash
    uv venv
    uv pip install -e .[dev]
    ```

3.  **Activate the virtual environment:**
    - On Windows:
      ```bash
      .venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```bash
      source .venv/bin/activate
      ```

### Running Checks

- **Linting and Formatting:**
  To automatically format and lint the code, run:
  ```bash
  uv run ruff format .
  uv run ruff check . --fix
  ```

- **Running Tests:**
  - To run the default (fast) unit tests:
    ```bash
    uv run hatch test
    ```
  - To run the full test suite, including `slow` integration tests that require a running Ollama instance:
    ```bash
    uv run hatch test -m "slow or not slow"
    ```
  - To pass a custom host to the integration tests:
    ```bash
    uv run hatch test -m "slow" --host http://192.168.0.101:11434
    ```

## License

This project is licensed under the MIT License.
