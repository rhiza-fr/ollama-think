

# Corrected import: httpx uses ConnectError for connection issues
from typing import Literal

import pytest

from ollama_think import Client


@pytest.mark.slow
def test_gptoss():
    c = Client(host="http://localhost:11434")
    model = "gpt-oss"
    prompt = "What is 2 + 6^2 / 0.11? Show your working"

    levels : list[bool | Literal['low', 'medium', 'high']] = [False, True, 'low', 'medium', 'high']

    for level in levels:
        print(f"Calling {model} with think={level} prompt='{prompt}'")
        res = c.call(model="gpt-oss", prompt=prompt, think=level)
        print("Thinking:", res.thinking)
        print("Content:", res.content)
        print("-" * 50)
        assert len(res.thinking) > 0  # gpt-oss does not respect think=False !
        assert len(res.content) > 0



if __name__ == "__main__":
    test_gptoss()