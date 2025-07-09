from ollama_think import Client


def main():
    client = Client(host="http://localhost:11434")

    # printing the raw response
    print("Calling model...")
    response = client.call(
        model="qwen3",
        prompt="Is concentration of capital a good thing?",
        think=True,
    )
    print(response)

    # printing the response interpreted as response.message.content
    response = client.call(model="qwen3", prompt="Why is the meaning of life 42?", think=True)
    print(f"{response}")

    # separating thinking and content
    thinking, content = client.call(
        model="qwen3",
        prompt="The future is here it is just not evenly distributed: Discuss...",
        think=True,
    )
    print(f"Thinking: {thinking}")
    print(f"\nContent: {content}")

    # using a traditional message
    messages = [
        {
            "role": "user",
            "content": "Make a system prompt for feng shui analysis",
        }
    ]
    response = client.call(model="qwen3", messages=messages, think=True)
    print(f"{response}")


if __name__ == "__main__":
    main()
