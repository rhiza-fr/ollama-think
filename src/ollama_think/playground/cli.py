from typing import Annotated

import typer
from rich import print

from ollama_think import Client


def main(
    prompt: Annotated[str, typer.Argument(help="The prompt to send to the model.")],
    model: Annotated[str, typer.Option("--model", "-m", help="The model to use.")] = "qwen3",
    think: Annotated[
        bool,
        typer.Option(help="Enable thinking mode. Use --no-think to disable."),
    ] = True,
    host: Annotated[str, typer.Option(help="The Ollama host URL.")] = "http://localhost:11434",
) -> None:
    """
    A simple command-line interface for ollama-think.
    """
    with Client(host=host) as client:
        seen_first_content = False
        try:
            stream = client.stream(model=model, prompt=prompt, think=think)
            if think:
                print("[dim]Thinking...[/dim]")
            for thinking, content in stream:
                if thinking:
                    print(f"[i]{thinking}[/i]", end="")
                elif think and seen_first_content is False:
                    seen_first_content = True
                    print("[dim]Content...[/dim]")
                print(content, end="")
        except Exception as e:
            print("[bold red]Error:[/bold red] Maybe use --host http://localhost:11434 ?")
            print(f"The underlying error was: {e}")


def entrypoint():
    typer.run(main)


if __name__ == "__main__":
    entrypoint()
