import json
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

try:
    from rich import print
except ImportError:
    from builtins import print

from ollama_think import Client

prompt = "what is 2 + 3?"


def _thinking_mode(client: Client, model: str, think: bool = True) -> tuple[bool, str]:
    try:
        tr = client.call(model=model, prompt=prompt, think=think)
        if not tr.thinking:
            return False, "Thinking supported but empty"
        return (True, "")
    except Exception as e:
        if str(e).find("does not support thinking") > -1:
            err = "Does not support thinking"
            return (False, err)
        err = f"{e}"
        return (False, err)


def _content_no_thinking(client: Client, model: str) -> tuple[bool, str]:
    try:
        tr = client.call(model=model, prompt=prompt, think=False)
        if tr.content.find("<think>") > -1 or tr.content.find("Here is my thought process") > -1:
            return (
                False,
                f"Thinking outputed to content when think=False '{tr.content}'",
            )
        return (True, "")
    except Exception as e:
        err = f"{e}"
        return (True, err)


def _json_format(client: Client, model: str, think: bool = True) -> tuple[bool, str]:
    r = None
    try:
        r = client.call(model=model, prompt=prompt, format="json", think=think)
        _ = json.loads(r.content)
        return True, ""
    except Exception as e:
        err = f"{e}"
        if r:
            err = err + f" '{r}'"
        return False, err


class ResponseObj(BaseModel):
    """A specially crafted response object to capture an iterpretation of heat"""

    addition_result: int = Field(..., description="the result of the addition")


def _pydantic_format(client: Client, model: str, think: bool = True) -> tuple[bool, str]:
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
        err = f"{e}"
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


def _tool_calling(client: Client, model: str, think: bool = True) -> tuple[bool, str]:
    r = None
    try:
        r = client.call(model=model, prompt=prompt, tools=[addTwoInts], think=think)
        if r.message.tool_calls:
            if r.message.tool_calls[0].function.name == "addTwoInts":
                return True, ""
        return False, f"Expected tool call, received '{r}'"
    except Exception as e:
        if str(e).find("does not support tools") > 0:
            err = "Does not support tools"
            return False, err
        err = f"{e}"
        if r:
            err = err + f" '{r}'"
        return False, err


def _get_model_names(client: Client):
    return [m["model"] for m in client.list()["models"]]


def pytest_generate_tests(metafunc):
    if "test_spec" in metafunc.fixturenames:
        host = metafunc.config.getoption("--host")
        client = Client(host=host)

        selected_model = metafunc.config.getoption("--model")
        if selected_model:
            models = [selected_model]
        else:
            model_names = _get_model_names(client)
            blacklisted_models = [
                "mxbai-embed-large:latest",
                "granite-embedding:278m",
                "nomic-embed-text:latest",
                "tazarov/all-minilm-l6-v2-f32:latest",
                "stable-code:3b-code-q4_0",
                "stable-code:3b-code-q4_0",
                "qwen3:4b-nothink",
            ]
            models = [m for m in model_names if m not in blacklisted_models]

        test_specs = []
        for model_name in models:
            for hacks_enabled in [True, False]:
                test_specs.append({"model_name": model_name, "hacks_enabled": hacks_enabled})

        ids = [
            f"{spec['model_name']}-hacks_{'on' if spec['hacks_enabled'] else 'off'}"
            for spec in test_specs
        ]
        metafunc.parametrize("test_spec", test_specs, ids=ids)


@pytest.mark.slow
def test_model_capabilities(client: Client, test_spec: dict):
    model_name = test_spec["model_name"]
    hacks_enabled = test_spec["hacks_enabled"]

    if hacks_enabled:
        client.config.enable_hacks = True
        output_dir = Path("test_output/hacks_enabled")
    else:
        client.config.enable_hacks = False
        output_dir = Path("test_output/hacks_disabled")

    print(f"Testing model: {model_name} (hacks={'on' if hacks_enabled else 'off'})")

    can_think = _thinking_mode(client, model=model_name, think=True)
    can_json = _json_format(client, model=model_name, think=False)
    can_pydantic = _pydantic_format(client, model=model_name, think=False)
    can_tool_call = _tool_calling(client, model=model_name, think=False)
    content_no_thinking = _content_no_thinking(client, model=model_name)

    if can_think[0] is True:
        can_json_think = _json_format(client, model=model_name, think=True)
        can_pydantic_think = _pydantic_format(client, model=model_name, think=True)
        can_tool_call_think = _tool_calling(client, model=model_name, think=True)
    else:
        can_json_think = (False, "Thinking not supported")
        can_pydantic_think = (False, "Thinking not supported")
        can_tool_call_think = (False, "Thinking not supported")

    results = {
        "can_think": can_think,
        "can_json": can_json,
        "can_pydantic": can_pydantic,
        "can_json_think": can_json_think,
        "can_pydantic_think": can_pydantic_think,
        "can_tool_call": can_tool_call,
        "can_tool_call_think": can_tool_call_think,
        "content_no_thinking": content_no_thinking,
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    sanitized_model_name = model_name.replace(":", "_").replace("/", "_")

    outpath = output_dir / f"{sanitized_model_name}.json"
    outpath.write_text(
        json.dumps({model_name: results}, indent=4, sort_keys=True), encoding="utf-8"
    )
