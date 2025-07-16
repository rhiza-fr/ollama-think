import json

from pydantic import BaseModel, Field

try:
    from rich import print  # type: ignore
except ImportError:
    from builtins import print

from ollama_think import Client


# Used by format=MyResponseObject.model_json_schema()
class MyResponseObject(BaseModel):
    """A specially crafted response object to capture an iterpretation of heat"""

    how_hot_is_the_world: str = Field(default=..., description="your reasoning for the response")
    average_temperature: float = Field(..., description="average temperature")
    confidence: int = Field(
        default=...,
        description="how confident you are about your respponse. 0-10",
    )


def main():
    with Client(host="http://localhost:11434") as client:
        # implicit format=None
        print("Default format is string:")
        print(f"{client.call(model='qwen3', prompt='Hello, who made you?', think=True)}")

        # explicit json format
        print("\njson format: response as text")
        text_json = client.call(
            model="qwen3",
            prompt="Design a json representation of a spiral galaxy",
            format="json",
        ).content
        print(text_json)

        # try to parse it
        try:
            print("Parsed json:")
            print(json.loads(text_json))
        except Exception as e:
            print("Failed to parse json", e)

        # explicit constrained format
        print("\nExplicit object format:")
        text_obj = client.call(
            model="qwen3",
            prompt="How hot is the world?",
            format=MyResponseObject.model_json_schema(),
        ).content
        print("Content as text:")
        print(text_obj)

        # try to parse using pydantic
        try:
            print("\nParsed object:")
            print(MyResponseObject.model_validate_json(text_obj))
        except Exception as e:
            print("Failed to parse object", e)
            # it might still be valid-ish json
            # See https://github.com/Qarj/fix-busted-json
            # pip install fix-busted-json
            # from fix_busted_json import repair_json
            # fixed_json = repair_json(text_obj)
            # print(MyResponseObject.model_validate_json(fixed_json))


if __name__ == "__main__":
    main()
