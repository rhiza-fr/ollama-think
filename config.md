# Model Configuration Details

Use `config.yaml` settings to enable and parse "thinking" output from models that do not officially support the standard `think=True` parameter.

When a request is made, the library checks if the requested model name starts with one of the names listed in the `models` section of `config.yaml`. The first match is used.

## Default Behavior

If a model is not explicitly listed, no alteration is made.

- **`enable_thinking: false`**: The library will not send `think=True` to the Ollama API, preventing errors for models that don't support it.
- **`add_message: null`**: No special message is injected into the prompt.
- **`content_parsers`**: A default regex `"<think>(?P<thinking>.*?)</think>(?P<content>.*)"` is used to attempt to find `<think>` tags in the response content.

--- 

## Model-Specific Configurations

Here are the details for each model with custom settings:

### `cogito`

- **`enable_thinking: false`**: Forces `think=False` to prevent Ollama from complaing

- **`add_message`**: Injects the message `{'role': 'system', 'content': 'Enable deep thinking subroutine.'}`.

- **`content_parsers`**: A default regex `"<think>(?P<thinking>.*?)</think>(?P<content>.*)"`

### `granite3.2-vision`

- **`content_parsers: []`**: This model does not produce thinking output, so all content parsers are disabled to prevent incorrect parsing.

### `granite3.2`

- **`add_message`**: Injects a `control` role message with the content `"thinking"`.
- **`content_parsers`**: This model has inconsistent output order, especially between streaming and non-streaming modes. Two parsers are provided to handle both cases:
    1.  Looks for the response first, then the thought process.
    2.  Looks for the thought process first, then the response.
    For streaming, only the second parser is used.

### `granite3.3`

- **`add_message`**: Injects a `control` role message with the content `"thinking"`.
- **`content_parsers`**: Uses a specific XML-style parser that looks for `<think>` and `<response>` tags: `"<think>(?P<thinking>.*?)</think><response>(?P<content>.*?)</response>"`.

### `reflection`

- **`content_parsers`**: Uses a specific XML-style parser that looks for `<thinking>` and `<output>` tags: `"<thinking>(?P<thinking>.*?)</thinking><output>(?P<content>.*?)</output>"`.

### Models Using Default Hacks

The following models are explicitly listed but use the default hack configuration (a regex parser for `<think>` tags). They are listed to ensure the hack is applied to them, as they are known to sometimes produce thinking output without being officially supported.

- `phi4-reasoning`
- `phi4-mini-reasoning`
- `deepscaler`
- `deepcoder`
