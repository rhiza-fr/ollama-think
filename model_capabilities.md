# Model Capability Report

This report compares model capabilities with and without `ollama-think`'s compatibility hacks.
A `❌` &rarr; `✅` indicates that the hack fixed a previously failing capability.
A `❗` indicates invalid JSON, on one test without specific encouragement.

| Model | JSON Format | Pydantic Format | Tool Calls | Thinking output | No thinking in Content |
|:---| :--- | :--- | :--- | :--- | :--- |
| `cogito:14b` | ✅ | ✅ | ✅ | [❌](## "Does not support thinking") &rarr; ✅ | ✅ |
| `qwen3:0.6b` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `qwen3:1.7b` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `qwen3:14b` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `qwen3:latest` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `deepcoder:latest` | ✅ | ✅ | [❌](## "Does not support tools") | [❌](## "Does not support thinking") &rarr; ✅ | [❌](## "Thinking outputed to content when think=False &#x27;&lt;th") &rarr; ✅ |
| `deepscaler:latest` | ✅ | ✅ | [❌](## "Does not support tools") | [❌](## "Does not support thinking") &rarr; ✅ | [❌](## "Thinking outputed to content when think=False &#x27;&lt;th") &rarr; ✅ |
| `deepseek-r1:1.5b` | ✅ | ✅ | [❌](## "Does not support tools") | ✅ | ✅ |
| `deepseek-r1:14b` | ✅ | ✅ | [❌](## "Does not support tools") | ✅ | ✅ |
| `deepseek-r1:7b` | ✅ | ✅ | [❌](## "Does not support tools") | ✅ | ✅ |
| `deepseek-r1:8b` | ✅ | ✅ | [❌](## "Does not support tools") | ✅ | ✅ |
| `deepseek-r1:latest` | ✅ | ✅ | [❌](## "Does not support tools") | ✅ | ✅ |
| `granite3.2:latest` | ✅ | ✅ | [❌](## "Expected tool call, received &#x27;5&#x27;") | [❌](## "Does not support thinking") &rarr; ✅ | ✅ |
| `granite3.3:latest` | ✅ | ✅ | [❌](## "Expected tool call, received &#x27;&#x27;") | [❌](## "Does not support thinking") &rarr; ✅ | ✅ |
| `llama3.1:latest` | ✅ | ✅ | ✅ | [❌](## "Does not support thinking") | ✅ |
| `llama3.2:latest` | ✅ | ✅ | ✅ | [❌](## "Does not support thinking") | ✅ |
| `mistral:latest` | ✅ | ✅ | ✅ | [❌](## "Does not support thinking") | ✅ |
| `phi4-reasoning:plus` | ✅ | ✅ | [❌](## "Does not support tools") | [❌](## "Does not support thinking") &rarr; ✅ | [❌](## "Thinking outputed to content when think=False &#x27;&lt;th") &rarr; ✅ |
| `qwen3:4b` | ✅ | ✅ | [❌](## "Expected tool call, received &#x27;The result of 2 + 3 ") | ✅ | ✅ |
| `codeqwen:7b-chat` | ✅ | ✅ | [❌](## "Does not support tools") | [❌](## "Does not support thinking") | ✅ |
| `deepseek-coder-v2:16b` | ✅ | ✅ | [❌](## "Does not support tools") | [❌](## "Does not support thinking") | ✅ |
| `dolphin-llama3:8b` | ✅ | ✅ | [❌](## "Does not support tools") | [❌](## "Does not support thinking") | ✅ |
| `ebdm/gemma3-enhanced:12b` | ✅ | ✅ | [❌](## "Expected tool call, received &#x27;2 + 3 = 5&#x27;") | [❌](## "Does not support thinking") | ✅ |
| `exaone-deep:latest` | ✅ | ✅ | [❌](## "Does not support tools") | [❌](## "Does not support thinking") | ✅ |
| `falcon3:10b` | ✅ | ✅ | [❌](## "Does not support tools") | [❌](## "Does not support thinking") | ✅ |
| `gemma2:latest` | ✅ | ✅ | [❌](## "Does not support tools") | [❌](## "Does not support thinking") | ✅ |
| `gemma3:12b` | ✅ | ✅ | [❌](## "Does not support tools") | [❌](## "Does not support thinking") | ✅ |
| `gemma3:latest` | ✅ | ✅ | [❌](## "Does not support tools") | [❌](## "Does not support thinking") | ✅ |
| `gemma3n:latest` | ✅ | ✅ | [❌](## "Does not support tools") | [❌](## "Does not support thinking") | ✅ |
| `granite3.1-dense:8b` | ✅ | ✅ | [❌](## "Expected tool call, received &#x27;&lt;tool_call&gt;[{&quot;argume") | [❌](## "Does not support thinking") | ✅ |
| `granite3.2-vision:latest` | ✅ | ✅ | [❌](## "Expected tool call, received &#x27; 5&#x27;") | [❌](## "Thinking supported but empty") | ✅ |
| `llama3.2-vision:latest` | ✅ | ✅ | [❌](## "Does not support tools") | [❌](## "Does not support thinking") | ✅ |
| `marco-o1:latest` | ✅ | ✅ | [❌](## "Does not support tools") | [❌](## "Does not support thinking") | ✅ |
| `minicpm-v:latest` | ✅ | ✅ | [❌](## "Does not support tools") | [❌](## "Does not support thinking") | ✅ |
| `phi3.5:latest` | ✅ | ✅ | [❌](## "Does not support tools") | [❌](## "Does not support thinking") | ✅ |
| `phi3:14b` | ✅ | ✅ | [❌](## "Does not support tools") | [❌](## "Does not support thinking") | ✅ |
| `phi4-mini-reasoning:latest` | [❗](## "Invalid JSON") | ✅ | [❌](## "Does not support tools") | [❌](## "Does not support thinking") &rarr; ✅ | [❌](## "Thinking outputed to content when think=False &#x27;&lt;th") &rarr; ✅ |
| `phi4-mini:latest` | ✅ | ✅ | [❌](## "Expected tool call, received &#x27;[\{ &quot;status&quot;: &quot;succe") | [❌](## "Does not support thinking") | ✅ |
| `phi4:latest` | ✅ | ✅ | [❌](## "Does not support tools") | [❌](## "Does not support thinking") | ✅ |
| `qwen2.5-coder:1.5b` | ✅ | ✅ | [❌](## "Expected tool call, received &#x27;```json {   &quot;name&quot;: ") | [❌](## "Does not support thinking") | ✅ |
| `qwen2.5-coder:14b` | ✅ | ✅ | [❌](## "Expected tool call, received &#x27;{&quot;name&quot;: &quot;addTwoInts") | [❌](## "Does not support thinking") | ✅ |
| `qwen2.5-coder:7b-base` | ✅ | ✅ | [❌](## "Does not support tools") | [❌](## "Does not support thinking") | ✅ |
| `qwen2.5-coder:latest` | ✅ | ✅ | [❌](## "Expected tool call, received &#x27;{&quot;name&quot;: &quot;addTwoInts") | [❌](## "Does not support thinking") | ✅ |
| `reader-lm:latest` | ✅ | ✅ | [❌](## "Does not support tools") | [❌](## "Does not support thinking") | ✅ |
| `smallthinker:latest` | ✅ | ✅ | [❌](## "Does not support tools") | [❌](## "Does not support thinking") | ✅ |
| `starcoder2:7b` | ✅ | ✅ | [❌](## "Does not support tools") | [❌](## "Does not support thinking") | ✅ |
| `wizardlm2:7b` | [❌](## "Unterminated string starting at: line 1 column 23 ") | ✅ | [❌](## "Does not support tools") | [❌](## "Does not support thinking") | ✅ |
| `deepseek-r1:14` | [❌](## "model &quot;deepseek-r1:14&quot; not found, try pulling it f") | [❌](## "model &quot;deepseek-r1:14&quot; not found, try pulling it f") | [❌](## "model &quot;deepseek-r1:14&quot; not found, try pulling it f") | [❌](## "model &quot;deepseek-r1:14&quot; not found, try pulling it f") | ✅ |
