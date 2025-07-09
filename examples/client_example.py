from ollama_think import Client


def main():
    client = Client(
        host="http://localhost:11434",
        cache_dir=".ollama_cache",
        clear_cache=False,
    )
    # do something
    client.close()  # optional, but polite: close the cache

    # or
    with Client(
        host="http://localhost:11434",
        cache_dir=".ollama_cache",
        clear_cache=False,
    ) as client:
        print(
            f"Response.content as string: {client.call(model='qwen3', prompt='Hello, world!', think=True)}"
        )

        # list running models
        print("\nRunning models:", client.ps())

        # stop a model
        client.stop("qwen3")

        # show available models
        print("\nAvailable models:\n", client.list())


if __name__ == "__main__":
    main()
