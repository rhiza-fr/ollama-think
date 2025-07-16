try:
    from rich import print  # type: ignore
except ImportError:
    from builtins import print

from ollama_think import Client


def main():
    client = Client(host="http://localhost:11434")

    prompt = "Describe the earth to an alien who has just arrived."
    options = {"num_ctx": 8192, "temperature": 0.9}

    print("Using prompt:", prompt)
    print("Using options:", options)

    thinking, content = client.call(model="qwen3", prompt=prompt, think=True, options=options)
    print("Thinking:", thinking)
    print("Content:", content)


if __name__ == "__main__":
    main()

# From https://github.com/ollama/ollama-python/blob/main/ollama/_types.py

# load time options
#   numa: Optional[bool] = None
#   num_ctx: Optional[int] = None
#   num_batch: Optional[int] = None
#   num_gpu: Optional[int] = None
#   main_gpu: Optional[int] = None
#   low_vram: Optional[bool] = None
#   f16_kv: Optional[bool] = None
#   logits_all: Optional[bool] = None
#   vocab_only: Optional[bool] = None
#   use_mmap: Optional[bool] = None
#   use_mlock: Optional[bool] = None
#   embedding_only: Optional[bool] = None
#   num_thread: Optional[int] = None

# runtime options
#   num_keep: Optional[int] = None
#   seed: Optional[int] = None
#   num_predict: Optional[int] = None
#   top_k: Optional[int] = None
#   top_p: Optional[float] = None
#   tfs_z: Optional[float] = None
#   typical_p: Optional[float] = None
#   repeat_last_n: Optional[int] = None
#   temperature: Optional[float] = None
#   repeat_penalty: Optional[float] = None
#   presence_penalty: Optional[float] = None
#   frequency_penalty: Optional[float] = None
#   mirostat: Optional[int] = None
#   mirostat_tau: Optional[float] = None
#   mirostat_eta: Optional[float] = None
#   penalize_newline: Optional[bool] = None
#   stop: Optional[Sequence[str]] = None
