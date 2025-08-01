# Ollama-Think

## Overview

Ollama-Think is a wrapper around ollama-python that adds caching, thinking mode support, and cleaner APIs. It extends base Ollama functionality with response caching, "thinking" output separation, and syntax sugar.

## Core Classes

### Client

Enhanced Ollama client with automatic caching and response processing.

**Constructor:**

```python
Client(host=None, cache_dir=".ollama_cache", clear_cache=False)
```

**Key Methods:**

- `call()` - Non-streaming chat, returns ThinkResponse
- `stream()` - Streaming chat, yields ThinkResponse chunks  
- `stop()` - Unload model from memory
- `close()` - Clean up cache

### ThinkResponse

Extends ollama.ChatResponse with convenience properties for accessing thinking and content separately.

**Properties:**

- `.thinking` - Model's reasoning process (empty string if None)
- `.content` - Final response content
- Can unpack: `thinking, content = response`
- String conversion returns content only

## Quick Start

```python
from ollama_think import Client

# Initialize with caching
client = Client(host="http://localhost:11434")

# Non-streaming call
thinking, content = client.call(
    model="qwen3",
    prompt="Why is the sky blue?",
    think=True
)

# Streaming call  
for thinking, content in client.stream(model="qwen3", prompt="Tell me a story", think=True):
    print(thinking, end="")
    print(content, end="")
```

## Parameters

All methods support these parameters:

- `model` - Model name (required)
- `prompt` - Single user message (converted to messages list)
- `messages` - List of conversation messages  
- `tools` - Available tools for model to call
- `think` - Enable thinking mode (default: False for call, True for stream)
- `format` - Response format: None, "json", or schema
- `options` - Model params dict (temperature, num_ctx, etc.)
- `keep_alive` - Memory retention time
- `use_cache` - Enable response caching (default: True)

## Key Features

### Caching

- Automatic disk-based caching using DiskCache
- Keyed by request hash + host
- Can be disabled per-call with `use_cache=False`
- Clear cache: `Client(clear_cache=True)`

### Thinking Mode

- Separates model reasoning from final output
- Enables thinking for unsupported models via hacks
- Access via `.thinking` and `.content` properties

### Response Formats

```python
# JSON format
response = client.call(model="qwen3", prompt="Design JSON", format="json")

# Pydantic schema
from pydantic import BaseModel
class MySchema(BaseModel):
    reasoning: str
    temperature: float

response = client.call(
    model="qwen3", 
    prompt="How hot?",
    format=MySchema.model_json_schema()
)
```

### Tool Calling

Pass callable functions or tool dictionaries via `tools` parameter. Model can request tool execution in response.

### Options

Common model parameters:

- `temperature` - Creativity (0.1-1.0)
- `num_ctx` - Context length (default 2048)

```python
client.call(
    model="qwen3",
    prompt="Describe Earth",
    options={'num_ctx': 8192, 'temperature': 0.9}
)
```

## Context Management

```python
# Automatic cleanup
with Client() as client:
    response = client.call(model="llama3", prompt="Hello")

# Manual cleanup
client = Client()
# ... use client ...
client.close()
```

## Async Support

AsyncClient provides same interface with async/await:

```python
async with AsyncClient() as client:
    response = await client.call(model="qwen3", prompt="Hello")
    async for chunk in client.stream(model="qwen3", prompt="Story"):
        print(chunk.content, end="")
```

## Access to Base Ollama

Full ollama.Client methods remain available:

```python
client = Client()
response = client.chat(model='llama3.2', messages=[...])  # Direct ollama call
```
