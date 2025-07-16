try:
    from rich import print  # type: ignore
except ImportError:
    from builtins import print

from ollama_think import Client


# this is an example function to call
def add_two_ints(a: int, b: int) -> int:
    """
    Add two numbers

    Args:
        a (int): The first number
        b (int): The second number

    Returns:
        int: The sum of the two numbers
    """
    return int(a) + int(b)


def main():
    client = Client(host="http://localhost:11434")
    messages = [
        {
            "role": "user",
            "content": "What is 3 + 4?",
        }
    ]
    registry = {"add_two_ints": add_two_ints}

    print("Calling model with tools...")
    response = client.call(model="qwen3", messages=messages, tools=[add_two_ints])
    print("\nRAW first response:")
    print(response.to_dict())

    # if there are tool_calls we would normally have hidden the first result
    # we will call all the tools, add these as message and send back to the model
    if response.message.tool_calls:
        for tool_call in response.message.tool_calls:
            name = tool_call.function.name
            args = tool_call.function.arguments
            if name in registry:
                print(f"Calling tool: {name} with args: {args}")
                result = registry[name](**args)
                print("Result:", result)
                messages.append({"role": "tool", "name": name, "content": str(result)})
        if len(messages) > 1:
            second_response = client.call(model="qwen3", messages=messages)
            print("\nRAW second response:")
            print(second_response.to_dict())


if __name__ == "__main__":
    main()
