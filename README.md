# Ollama-Think Library

A thin wrapper around the [ollama-python](https://github.com/ollama/ollama-python) library with the addition of caching, increased `think` model compatibility and a little syntax sugar.


## Features

- **Caching**: Automatically caches responses to significantly speed up repeated requests.
- **Thinking**: Enables some officially unsupported models to use thinking mode. [Why hack?](why_hack.md)
- **Streaming and Non-streaming**: Separates the underlying streaming and non-streaming interface to provide clean type hints.
- **Syntax Sugar**: Less boiler-plate, so that you can maintain your flow.

## Quickstart

Get up and running in less than a minute.

**1. Install the library:**

```bash
pip install ollama-think
```

**2. Use:**

```python
from ollama_think import Client

# Initialize the client
client = Client(host="http://localhost:11434", cache_dir=".ollama_cache", clear_cache=False)

# unpack the response into thinking and content
thinking, content = client.call(
    model="qwen3",                 # or any other model
    prompt="Why is the sky blue?", # shortcut for messages=[{'role': 'user', 'content': 'Why is the sky blue?'}]
    think=True                     # Set to True to see the model's thinking process or ('low', 'medium', 'high' for gpt-oss)
)

print(f"Thinking: {thinking}, Content: {content}")
```

## Detailed Usage


### Non-streaming

The `call` method provides strongly typed access to the underlying `Chat` method in non-streaming mode. It returns a `ThinkResponse` object which is a subclass of `ollama.ChatResponse` and adds some convenience properties. You can use `prompt` or `messages` as you prefer.

```python
from ollama_think import Client
client = Client()

# Make a non-streaming call
response: ThinkResponse = client.call(
    model="qwen3",         # The model to use
    prompt="Hello, world!" # A single user message
    messages = None,       # or a list of messages
    tools = None,          # A list of tools available
    think = True,          # Enable thinking mode
    format = None,         # The format to return a response in: None | 'json' | your_obj.model_json_schema()
    options = None,        # Additional model parameter dict, such as {'temperature': 0.1, 'num_ctx': 8192}
    keep_alive = None,     # Controls how long the model will stay loaded in memory following the request.
    use_cache = True)      # If True, attempts to retrieve the response from cache.

# The response object contains all the original data from the Ollama ChatResponse
print(response)
# ThinkResponse(
#     model='qwen3',
#     created_at='2025-07-03T14:16:05.8452406Z',
#     done=True,
#     done_reason='stop',
#     total_duration=2461619200,
#     load_duration=2111438400,
#     prompt_eval_count=20,
#     prompt_eval_duration=78409600,
#     eval_count=16,
#     eval_duration=271104600,
#     message=Message(role='assistant', content='Hello, world! How can I assist you today?', thinking='...',
#                     images=None, tool_calls=None))

# For convenience, you can access the content and thinking as properties
print(response.thinking)
# '...'
print(response.content)
# 'Hello, world! ...'

# The response object can be used as a string which will show just the 'content'
print(f"The model said: {response}")  # same as response.content
# The model said: Hello, world! ...

# or unpack the response into thinking and content for single line access
thinking, content = response
print(f"Thinking: {thinking}, Content: {content}")
```

### Streaming

The `stream` method provides a strongly typed access to the underlying `Chat` method in streaming mode. It returns a an iterator of `ThinkResponse` chunks

```python
from ollama_think import Client
client = Client()

stream = client.stream(model="qwen3", prompt="Tell me a short story about italian chimpanzees and bananas", think=True)
for thinking, content in stream:
    print(thinking, end="")
    print(content, end="")  # empty until thinking is finished for most models
```

### Thinking Mode

The `think` parameter tells ollama to enable thinking for models that support this. For other models that use non-standard ways of enabling thinking we do the neccesary. [Why hack?](why_hack.md) Default config: [src/ollama_think/config.yaml](src/ollama_think/config.yaml) Results: [model_capabilities.md](model_capabilities.md)

Some models will think, even without 'enabling' thinking. This output is separated out of the `content` into `thinking`.

Note: Not all models officially or unofficially support thinking. They will throw a `400` error if you try to enable thinking.


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
from ollama_think import Client
client = Client()

prompt="Describe the earth to an alien who has just arrived."
options={'num_ctx': 8192, 'temperature': 0.9}

print("Using prompt:", prompt)
print("Using options:", options)

thinking, content = client.call(model="qwen3", prompt=prompt, think=True, options=options)
print(f"Thinking: {thinking}, Content: {content}")
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
from ollama_think import Client
import json

client = Client()

text_json = client.call(
    model="qwen3",
    prompt="Design a json representation of a spiral galaxy",
    format="json",
).content

my_object = json.loads(text_json)  # might explode if invalid json was returned
```

You can use pydantic models to describe more exactly the format you want.

```python
from ollama_think import Client
from pydantic import BaseModel, Field
client = Client()

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

Since the `ollama_think` is a thin wrapper around the `ollama.client`, you can still access the all the underlying ollama client methods.

```python
from ollama_think import Client
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
from ollama_think import Client

client = Client()
# the prompt parameter in `call` and `stream` is just a shortcut for
prompt = 'Why is the sky blue?'
message =  {'role': 'user', 'content': prompt}
client.call(model='llama3.2', messages=[message])  # shortcut
client.call(model='llama3.2', prompt=prompt)       # same thing
```

## Credit to

- ollama [https://ollama.com/](https://ollama.com/)
- ollama-python [https://github.com/ollama/ollama-python](https://github.com/ollama/ollama-python)
- diskcache [https://github.com/grantjenks/python-diskcache/](https://github.com/grantjenks/python-diskcache/)
- pydantic [https://pydantic-docs.helpmanual.io/](https://pydantic-docs.helpmanual.io/)

## Reference docs

- Ollama Thinking - [https://ollama.com/blog/thinking](https://ollama.com/blog/thinking)
- Ollama Tool support - [https://ollama.com/blog/tool-support](https://ollama.com/blog/tool-support)
- Ollama Structured Outputs - [https://ollama.com/blog/structured-outputs](https://ollama.com/blog/structured-outputs)
- Ollama Options - [https://github.com/ollama/ollama-python/blob/main/ollama/_types.py](https://github.com/ollama/ollama-python/blob/main/ollama/_types.py)

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

### Development Setup

This project uses `uv` for package management, but pip should work too.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/ollama-think.git
    cd ollama-think
    ```

2.  **Create a virtual environment and install dependencies:**
    This command creates a virtual environment in `.venv` and installs all dependencies, including development tools.
    ```bash
    uv sync --extra dev
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
    uv run pytest
    ```
  - To run the full test suite, including `slow` integration tests that require a running Ollama instance:
    ```bash
    uv run pytest -m "slow or not slow"
    ```
  - To pass a custom host to the integration tests:
    ```bash
    uv run pytest -m "slow or not slow" --host http://localhost:11434
    ```


- **Testing new models:**

  ```python
  # edit /src/ollama_think/config.yaml
  # check the output from non-streaming and streaming
  uv run ./tests/test_hacks.py --host http://localhost:11434 --model "model_name"

  # check that this makes a difference
  uv run pytest ./tests/test_model_capabilities.py --host http://localhost:11434 -m "slow" --model "model_name"

  # re-generate doc
  uv run tests/generate_model_capabilities_report.py

  # submit a PR
  ```
 
## License

This project is licensed under the MIT License.
