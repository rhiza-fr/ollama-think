from typer import run

from ollama_think.client import Client


def main():
    # default arg is prompt
    # should take --model model, --think --no-think --host host --chat
    # verbosity 
    #   0) silent
    #   1) prompt -> content
    #   2) prompt -> thinking, content
    #   3) A transient waiting anim, transient streaming thinking then response
    #   4) A rich pannel with streaming markup data and full stats
    with Client(host="http://localhost:11434") as client:
        # do something
        pass


if __name__ == "__main__":
    run(main)

