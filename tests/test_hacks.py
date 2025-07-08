import pytest
from rich import print

from ollama_think.client import Client


def pytest_generate_tests(metafunc):
    if "test_case" in metafunc.fixturenames:
        host = metafunc.config.getoption("--host")
        client = Client(host=host)
        hacks = client.config.models

        # all model names on the server
        selected_model = metafunc.config.getoption("--model")
        if selected_model:
            models = [selected_model]
        else:
         models = [m["model"] for m in client.list()["models"]]

        # refine to models that have hacks
        test_cases = []
        for model in models:
            for hack_model in hacks.keys():
                if model.startswith(hack_model):
                    if len(hacks[hack_model].get("content_parsers", [])) == 0:
                        break  # granite3.2-vision is skipped
                    test_cases.append({"model": model, "hack": hacks[hack_model]})
                    break
        metafunc.parametrize("test_case", test_cases)


@pytest.mark.slow
def test_models(client, test_case):
    model = test_case["model"]
    hack = test_case["hack"]
    prompt = "What is 12 + 12 / 3?"

    print(f"{'=' * 20} Model: {model} {'=' * 20}")
    print(hack)

    thinking, content = client.call(model=model, prompt=prompt, think=True)
    print(f"--- Thinking ---\n[i]{thinking}[/i]")
    print(f"\n--- Content ---\n{content}")
    print("-" * 50)

    assert len(thinking) > 0
    assert len(content) > 0

    print(f"{'=' * 15} Model: {model} (STREAMING) {'=' * 15}")
    print(hack)
    stream = client.stream(model=model, prompt=prompt, think=True)
    thinking_chunks = []
    content_chunks = []
    chunks = []
    for chunk in stream:
        thinking_chunks.append(chunk.thinking)  # we don't need to see the streaming come in
        content_chunks.append(chunk.content)
        chunks.append(chunk)  # for debug
    print(f"--- Thinking ---\n[i]{''.join(thinking_chunks)}[/i]")
    print(f"\n--- Content ---\n{''.join(content_chunks)}")
    assert len(thinking) > 0
    assert len(content) > 0
    print("-" * 50)
