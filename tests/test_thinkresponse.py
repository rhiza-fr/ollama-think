import unittest

from ollama import ChatResponse, Message

from ollama_think.thinkresponse import ThinkResponse


class TestThinkResponse(unittest.TestCase):
    def test_str_with_content(self):
        message = Message(role="assistant", content="Hello, world!")
        chat_response = ChatResponse(
            model="llama2",
            created_at="2023-08-04T19:22:45.499127Z",
            message=message,
            done=True,
        )
        think_response = ThinkResponse(chat_response)
        self.assertEqual(str(think_response), "Hello, world!")

    def test_str_without_content(self):
        message = Message(role="assistant", content=None)
        chat_response = ChatResponse(
            model="llama2",
            created_at="2023-08-04T19:22:45.499127Z",
            message=message,
            done=True,
        )
        think_response = ThinkResponse(chat_response)
        self.assertEqual(str(think_response), "")

    def test_repr(self):
        message = Message(
            role="assistant", content="Hello!", thinking="I am thinking..."
        )
        chat_response = ChatResponse(
            model="llama2",
            created_at="2023-08-04T19:22:45.499127Z",
            message=message,
            done=True,
        )
        think_response = ThinkResponse(chat_response)
        expected_repr = "ThinkResponse(thinking='I am thinking...', content='Hello!')"
        self.assertEqual(repr(think_response), expected_repr)

    def test_iter(self):
        message = Message(
            role="assistant", content="Hello!", thinking="I am thinking..."
        )
        chat_response = ChatResponse(
            model="llama2",
            created_at="2023-08-04T19:22:45.499127Z",
            message=message,
            done=True,
        )
        think_response = ThinkResponse(chat_response)
        thinking, content = think_response
        self.assertEqual(thinking, "I am thinking...")
        self.assertEqual(content, "Hello!")

    def test_thinking_property(self):
        message = Message(role="assistant", content="", thinking="Thinking aloud.")
        chat_response = ChatResponse(
            model="llama2",
            created_at="2023-08-04T19:22:45.499127Z",
            message=message,
            done=True,
        )
        think_response = ThinkResponse(chat_response)
        self.assertEqual(think_response.thinking, "Thinking aloud.")

    def test_content_property(self):
        message = Message(role="assistant", content="This is the content.")
        chat_response = ChatResponse(
            model="llama2",
            created_at="2023-08-04T19:22:45.499127Z",
            message=message,
            done=True,
        )
        think_response = ThinkResponse(chat_response)
        self.assertEqual(think_response.content, "This is the content.")


if __name__ == "__main__":
    unittest.main()