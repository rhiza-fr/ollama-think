import json
import unittest

from ollama import ChatResponse, Message

from ollama_think.thinkresponse import ThinkResponse


class TestThinkResponse(unittest.TestCase):
    def _make_response(self, thinking="", content="Hello, world!"):
        message = Message(role="assistant", thinking=thinking, content=content)
        chat_response = ChatResponse(
            model="llama2",
            created_at="2023-08-04T19:22:45.499127Z",
            message=message,
            done=True,
        )
        return ThinkResponse(chat_response)

    def test_str_with_content(self):
        think_response = self._make_response()
        self.assertEqual(str(think_response), "Hello, world!")

    def test_str_without_content(self):
        think_response = self._make_response(content="")
        self.assertEqual(str(think_response), "")

    def test_repr(self):
        self.maxDiff = None
        think_response = self._make_response(content="Hello!", thinking="I am thinking...")
        expected_repr = "ThinkResponse(model='llama2', created_at='2023-08-04T19:22:45.499127Z', done=True, done_reason=None, total_duration=None, load_duration=None, prompt_eval_count=None, prompt_eval_duration=None, eval_count=None, eval_duration=None, message=Message(role='assistant', content='Hello!', thinking='I am thinking...', images=None, tool_name=None, tool_calls=None))"
        self.assertEqual(repr(think_response), expected_repr)

    def test_iter(self):
        think_response = self._make_response(content="Hello!", thinking="I am thinking...")
        thinking, content = think_response
        self.assertEqual(thinking, "I am thinking...")
        self.assertEqual(content, "Hello!")

    def test_thinking_property(self):
        think_response = self._make_response(content="Hello!", thinking="Thinking aloud.")
        self.assertEqual(think_response.thinking, "Thinking aloud.")

    def test_content_property(self):
        think_response = self._make_response(content="This is the content.")
        self.assertEqual(think_response.content, "This is the content.")

    def test_is_serializable(self):
        think_response = self._make_response()
        json_data = think_response.model_dump_json()
        self.assertIsInstance(json_data, str)
        self.assertEqual(json_data, think_response.to_json())

    def test_is_nearly_json_dumpable(self):
        think_response = self._make_response()
        json_data = json.dumps(think_response.to_dict())  # cheat with to_dict()
        self.assertIsInstance(json_data, str)


if __name__ == "__main__":
    unittest.main()
