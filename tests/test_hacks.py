
import pytest
from rich import print

from ollama_think.client import Client


@pytest.mark.slow
def test_models():
    client = Client(host="http://localhost:11434") #  clear_cache=True
    hacks = client.config.models
    prompt = "What is 12 + 12 / 3?"

    # all model names on the server
    all_models = [m["model"] for m in client.list()["models"]]

    # refine to models that have hacks
    test_models = {}
    for model in all_models:
        for hack_model in hacks.keys():
            if model.startswith(hack_model):
                test_models[model] = hacks[hack_model]
                break
                
    for model, hack in test_models.items():
        if len(hack.get('content_parsers', [])) == 0:
            continue # granite3.2-vision is skipped

        print(f"{'='*20} Model: {model} {'='*20}")
        print(hack)

        thinking, content = client.call(model=model, prompt=prompt, think=True)
        print(f"--- Thinking ---\n[i]{thinking}[/i]")
        print(f"\n--- Content ---\n{content}")
        print('-'*50)
        
        assert(len(thinking) > 0)
        assert(len(content) > 0)

        print(f"{'='*15} Model: {model} (STREAMING) {'='*15}")
        print(hack)
        stream = client.stream(model=model, prompt=prompt, think=True)
        thinking_chunks = []
        content_chunks = []
        chunks = []
        for chunk in stream:
            thinking_chunks.append(chunk.thinking) # we don't need to see the streaming come in
            content_chunks.append(chunk.content)
            chunks.append(chunk) # for debug
        print(f"--- Thinking ---\n[i]{''.join(thinking_chunks)}[/i]")
        print(f"\n--- Content ---\n{''.join(content_chunks)}")
        assert(len(thinking) > 0)
        assert(len(content) > 0)    
        print('-'*50)

if __name__ == "__main__":
    test_models()