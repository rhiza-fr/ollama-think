import json
from pathlib import Path

import pytest
from pydantic import BaseModel, Field
from rich import print

from ollama_think.client import Client

client = Client(host="http://localhost:11434")
prompt = "what is 2 + 3?"


def _thinking_mode(model: str, think: bool = True) -> tuple[bool, str]:
    try:
        tr = client.call(model=model, prompt=prompt, think=think)
        if not tr.thinking:
            return False, "Thinking supported but empty"
        return (True, "")
    except Exception as e:
        err = f"Model {model}, THINK, think={think} failed: {e}"
        return (False, err)


def _content_no_thinking(model: str) -> tuple[bool, str]:
    try:
        tr = client.call(model=model, prompt=prompt, think=False)
        if tr.content.find("<think>") > -1:
            return (
                False,
                f"Thinking outputed to content when think=False {tr.content}",
            )
        return (True, "")
    except Exception as e:
        err = f"Model {model}, CONTENT THINK, failed: {e}"
        return (True, err)


def _json_format(model: str, think: bool = True) -> tuple[bool, str]:
    r = None
    try:
        r = client.call(model=model, prompt=prompt, format="json", think=think)
        _ = json.loads(r.content)
        return True, ""
    except Exception as e:
        err = f"Model {model}, JSON, think={think} failed: {e}"
        if r:
            err = err + f" '{r}'"
        return False, err


class ResponseObj(BaseModel):
    """A specially crafted response object to capture an iterpretation of heat"""

    addition_result: int = Field(..., description="the result of the addition")


def _pydantic_format(model: str, think: bool = True) -> tuple[bool, str]:
    r = None
    try:
        r = client.call(
            model=model,
            prompt=prompt,
            format=ResponseObj.model_json_schema(),
            think=think,
        )
        _ = ResponseObj.model_validate_json(r.content)
        return True, ""
    except Exception as e:
        err = f"Model {model}, PYDANTIC, think={think} failed: {e}"
        if r:
            err = err + f" '{r}'"
        return False, err


def addTwoInts(a: int, b: int) -> int:
    """
    Add two numbers

    Args:
        a (int): The first number
        b (int): The second number

    Returns:
        int: The sum of the two numbers
    """
    return int(a) + int(b)


def _tool_calling(model: str, think: bool = True) -> tuple[bool, str]:
    r = None
    try:
        r = client.call(model=model, prompt=prompt, tools=[addTwoInts], think=think)
        if r.message.tool_calls:
            if r.message.tool_calls[0].function.name == "addTwoInts":
                return True, ""
        return False, f"Expected tool call, received '{r}'"
    except Exception as e:
        err = f"Model {model}, TOOL, think={think} failed: {e}"
        if r:
            err = err + f" '{r}'"
        return False, err


def _get_model_names():
    return [m["model"] for m in client.list()["models"]]

@pytest.mark.slow
def test_model_capabilities():
    model_names = _get_model_names()
    blacklisted_models = [
        "mxbai-embed-large:latest",
        "granite-embedding:278m",
        "nomic-embed-text:latest",
        "tazarov/all-minilm-l6-v2-f32:latest",
        "stable-code:3b-code-q4_0",
    ]
    models = [m for m in model_names if m not in blacklisted_models]

    results = {}
    for name in models:
        print(f"Testing model: {name}")
        can_think = _thinking_mode(model=name, think=True)
        can_json = _json_format(model=name, think=False)
        can_pydantic = _pydantic_format(model=name, think=False)
        can_tool_call = _tool_calling(model=name, think=False)
        content_no_thinking = _content_no_thinking(model=name)

        if can_think[0] is True:
            can_json_think = _json_format(model=name, think=True)
            can_pydantic_think = _pydantic_format(model=name, think=True)
            can_tool_call_think = _tool_calling(model=name, think=True)

        else:
            can_json_think = False, "Thinking not supported"
            can_pydantic_think = False, "Thinking not supported"
            can_tool_call_think = False, "Thinking not supported"

        results[name] = {
            "can_think": can_think,
            "can_json": can_json,
            "can_pydantic": can_pydantic,
            "can_json_think": can_json_think,
            "can_pydantic_think": can_pydantic_think,
            "can_tool_call": can_tool_call,
            "can_tool_call_think": can_tool_call_think,
            "content_no_thinking": content_no_thinking,
        }
        outpath = Path("model_capabilies.json")
        outpath.write_text(json.dumps(results, indent=4, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    test_model_capabilities()
