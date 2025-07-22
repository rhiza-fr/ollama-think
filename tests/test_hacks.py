import pytest

try:
    from rich import print  # type: ignore
except ImportError:
    from builtins import print

from ollama_think import Client


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

    print(f"{'=' * 20} Model: {model} think=True {'=' * 20}")
    print(hack)

    thinking, content = client.call(model=model, prompt=prompt, think=True)
    print(f"--- Thinking ---\n{thinking}")
    print(f"\n--- Content ---\n{content}")
    print("-" * 50)

    assert len(thinking) > 0
    assert len(content) > 0

    print(f"{'=' * 15} Model: {model} think=True (STREAMING) {'=' * 15}")
    print(hack)
    stream = client.stream(model=model, prompt=prompt, think=True)
    thinking_chunks = []
    content_chunks = []
    chunks = []
    for chunk in stream:
        thinking_chunks.append(chunk.thinking)  # we don't need to see the streaming come in
        content_chunks.append(chunk.content)
        chunks.append(chunk)  # for debug
    thinking = ''.join(thinking_chunks)
    content = ''.join(content_chunks)
    print(f"--- Thinking ---\n{thinking}")
    print(f"\n--- Content ---\n{content}")

    assert len(thinking) > 0
    assert len(content) > 0
    print("-" * 50)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host")
    parser.add_argument("--model", help="Run a single model")
    args = parser.parse_args()

    class MockConfig:
        def __init__(self, host, model):
            self.host = host
            self.model = model

        def getoption(self, name):
            if name == "--host":
                return self.host
            if name == "--model":
                return self.model
            return None

    class MockMetafunc:
        def __init__(self, config):
            self.fixturenames = ["test_case"]
            self.config = config
            self.test_cases = []

        def parametrize(self, name, test_cases):
            self.test_cases = test_cases

    mock_config = MockConfig(args.host, args.model)
    mock_metafunc = MockMetafunc(mock_config)

    pytest_generate_tests(mock_metafunc)

    client = Client(host=args.host)

    if not mock_metafunc.test_cases:
        print("No models with hacks found to test.")
        sys.exit(0)

    for test_case in mock_metafunc.test_cases:
        try:
            test_models(client, test_case)
        except Exception as e:
            print(f"Error testing model {test_case['model']}: {e}")
