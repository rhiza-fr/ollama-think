from typing import TypedDict


class ThinkingHacks(TypedDict):
    enable_thinking: bool
    add_message: dict[str, str] | None
    content_parsers: list[str]


THINKING_HACKS: dict[str, ThinkingHacks] = {
    "cogito": {
        "enable_thinking": False,  # don't tell ollama that we think we can think
        "add_message": {
            "role": "system",
            "content": "Enable deep thinking subroutine.",
        },
        "content_parsers": ["<think>(?P<thinking>.*?)</think>(?P<content>.*)"],
    },
    "granite3.2": {
        "enable_thinking": False,
        "add_message": {"role": "control", "content": "thinking"},
        "content_parsers": [
            # yes, any order is correct!
            "Here is my response:\n(?P<content>.*?)Here is my thought process:\n(?P<thinking>.*)",
            "Here is my thought process:\n(?P<thinking>.*?)Here is my response:\n(?P<content>.*)",
        ],
    },
    "granite3.3": {
        "enable_thinking": False,
        "add_message": {"role": "control", "content": "thinking"},
        "content_parsers": [
            "<think>(?P<thinking>.*?)</think><response>(?P<content>.*?</response>)"
        ],
    },
    "phi4-reasoning": {
        "enable_thinking": False,
        "add_message": None,
        "content_parsers": ["<think>(?P<thinking>.*?)</think>(?P<content>.*)"],
    },
    "phi4-mini-reasoning": {
        "enable_thinking": False,
        "add_message": None,
        "content_parsers": ["<think>(?P<thinking>.*?)</think>(?P<content>.*)"],
    },
    "reflection": {
        "enable_thinking": False,
        "add_message": None,
        "content_parsers": [
            # untested
            "<thinking>(?P<thinking>.*?)</thinking>\\s*<output>(?P<content>.*?)</output>"
        ],
    },
    "deepscaler": {
        "enable_thinking": False,
        "add_message": None,
        "content_parsers": ["<think>(?P<thinking>.*?)</think>(?P<content>.*)"],
    },
    "deepcoder": {
        "enable_thinking": False,
        "add_message": None,
        "content_parsers": ["<think>(?P<thinking>.*?)</think>(?P<content>.*)"],
    },
}
