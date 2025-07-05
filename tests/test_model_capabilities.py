import html
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
        if str(e).find("does not support thinking") > -1:
            err = "Does not support thinking"
            return (False, err)
        err = f"{e}"
        return (False, err)


def _content_no_thinking(model: str) -> tuple[bool, str]:
    try:
        tr = client.call(model=model, prompt=prompt, think=False)
        if tr.content.find("<think>") > -1 or tr.content.find("Here is my thought process") > -1:
            return (
                False,
                f"Thinking outputed to content when think=False \'{tr.content}\'",
            )
        return (True, "")
    except Exception as e:
        err = f"{e}"
        return (True, err)


def _json_format(model: str, think: bool = True) -> tuple[bool, str]:
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


def _tool_calling(model: str, think: bool = True) -> tuple[bool, str]:
    r = None
    try:
        r = client.call(model=model, prompt=prompt, tools=[addTwoInts], think=think)
        if r.message.tool_calls:
            if r.message.tool_calls[0].function.name == "addTwoInts":
                return True, ""
        return False, f"Expected tool call, received '{r}'"
    except Exception as e:
        if str(e).find('does not support tools') > 0:
            err = "Does not support tools"
            return False, err
        err = f"{e}"
        if r:
            err = err + f" '{r}'"
        return False, err


def _get_model_names():
    return [m["model"] for m in client.list()["models"]]

@pytest.mark.slow
def test_model_capabilities(output_path: str = "model_capabilies.json"):
    model_names = _get_model_names()
    blacklisted_models = [
        "mxbai-embed-large:latest",
        "granite-embedding:278m",
        "nomic-embed-text:latest",
        "tazarov/all-minilm-l6-v2-f32:latest",
        "stable-code:3b-code-q4_0",
        "stable-code:3b-code-q4_0",
        "qwen3:4b-nothink"
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
        outpath = Path(output_path)
        outpath.write_text(json.dumps(results, indent=4, sort_keys=True), encoding="utf-8")


def generate_markdown_report(
    no_hacks: str = "model_capabilies_no_hacks.json",
    hacks: str = "model_capabilies.json",
    output_path: str = "model_capabilies.md",
):
    """
    Generates a Markdown report comparing model capabilities with and without hacks.

    It reads two JSON files containing test results, one with hacks enabled and one
    without. It then generates a Markdown table that summarizes the capabilities
    of each model, highlighting any improvements gained from using the hacks.

    Args:
        no_hacks: Path to the JSON file with results when hacks are disabled.
        hacks: Path to the JSON file with results when hacks are enabled.
        output_path: Path to write the output Markdown file.
    """
    no_hacks_path = Path(no_hacks)
    hacks_path = Path(hacks)

    if not no_hacks_path.exists():
        print(f"WARNING: '{no_hacks_path}' not found. Skipping report generation.")
        return
    if not hacks_path.exists():
        print(f"WARNING: '{hacks_path}' not found. Skipping report generation.")
        return

    no_hacks_data = json.loads(no_hacks_path.read_text(encoding="utf-8"))
    hacks_data = json.loads(hacks_path.read_text(encoding="utf-8"))

    # Get all model names and sort them
    all_models = sorted(list(set(no_hacks_data.keys()) | set(hacks_data.keys())))

    capabilities = [
        "can_json",
        "can_pydantic",
        "can_tool_call",
        "can_think",
        "content_no_thinking",
        # "can_json_think",
        # "can_pydantic_think",
        # "can_tool_call_think",
    ]
    header_map = {
        "can_think": "Thinking output",
        "content_no_thinking": "No thinking in Content",
        "can_json": "JSON Format",
        "can_pydantic": "Pydantic Format",
        "can_tool_call": "Tool Calls",
        # "can_json_think": "JSON (T)",
        # "can_pydantic_think": "Pydantic (T)",
        # "can_tool_call_think": "Tool Call (T)",
    }

    # Build Markdown table
    md_lines = [
        "# Model Capability Report",
        "",
        "This report compares model capabilities with and without `ollama-think`'s compatibility hacks.",
        "A `❌` &rarr; `✅` indicates that the hack fixed a previously failing capability.",
        "",
    ]

    # Header
    header = "| Model | " + " | ".join(header_map[c] for c in capabilities) + " |"
    md_lines.append(header)
    # Separator
    separator = "|:---| " + " | ".join([":---:"] * len(capabilities)) + " |"
    md_lines.append(separator)

    def format_icon(res):
        ok = res[0]
        desc = html.escape( res[1][:50].replace('\n'," ").replace('|',''))
        if ok:
            return "✅"
        return f"[❌](## \"{desc}\")"

    def format_cell(no_hacks_res, hacks_res):
        no_hacks_ok = no_hacks_res[0] if no_hacks_res else False
        hacks_ok = hacks_res[0] if hacks_res else False

        no_hacks_icon = format_icon(no_hacks_res)
        hacks_icon = format_icon(hacks_res)

        if no_hacks_ok != hacks_ok:
            return f"{no_hacks_icon} &rarr; {hacks_icon}"
        return hacks_icon

    # Rows
    for model in all_models:
        row = [f"`{model}`"]
        no_hacks_results = no_hacks_data.get(model, {})
        hacks_results = hacks_data.get(model, {})

        for cap in capabilities:
            no_hacks_res = no_hacks_results.get(cap)
            hacks_res = hacks_results.get(cap)
            row.append(format_cell(no_hacks_res, hacks_res))

        md_lines.append("| " + " | ".join(row) + " |")

    # Write to file
    output_file = Path(output_path)
    output_file.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Markdown report generated at: {output_path}")


if __name__ == "__main__":
    test_model_capabilities()
    client.config.enable_hacks = False
    test_model_capabilities(output_path="model_capabilies_no_hacks.json")
    generate_markdown_report()
