import unittest
from unittest.mock import patch

from ollama import ChatResponse, Message

from ollama_think.client import Client


class TestOllamaClient(unittest.TestCase):
    def setUp(self):
        """Set up mocks for Cache and the upstream client before each test."""
        self.cache_patcher = patch("ollama_think.client.Cache")
        self.MockCache = self.cache_patcher.start()
        self.mock_cache_instance = self.MockCache.return_value
        self.mock_cache_instance.get.return_value = (
            None  # Default to a cache miss
        )

        self.chat_patcher = patch("ollama_think.client.OllamaClient.chat")
        self.mock_chat = self.chat_patcher.start()

        self.addCleanup(self.cache_patcher.stop)
        self.addCleanup(self.chat_patcher.stop)

    def test_clear_cache_on_init(self):
        """Test if the cache is cleared when clear_cache=True."""
        Client(clear_cache=True)
        self.mock_cache_instance.clear.assert_called_once()

    def test_close_method(self):
        """Test that the explicit close method calls the cache's close method."""
        client = Client()
        client.close()
        self.mock_cache_instance.close.assert_called_once()

    def test_call_with_prompt_and_cache_miss(self):
        """Test a standard call that results in a cache miss and stores the result."""
        self.mock_chat.return_value = ChatResponse(
            model="llama2",
            created_at="",
            message=Message(role="assistant", content="Hello, world!"),
            done=True,
        )
        client = Client()
        response = client.call(model="llama2", prompt="Hello, world!")

        self.assertEqual(response.content, "Hello, world!")
        self.mock_cache_instance.get.assert_called_once()
        self.mock_chat.assert_called_once()
        self.mock_cache_instance.set.assert_called_once()

    def test_call_with_cache_hit(self):
        """Test that a call retrieves a response from the cache."""
        cached_response = ChatResponse(
            model="llama2",
            created_at="",
            message=Message(role="assistant", content="Cached response"),
            done=True,
        )
        self.mock_cache_instance.get.return_value = cached_response
        client = Client()

        response = client.call(model="llama2", prompt="Cache me")

        self.assertEqual(response.content, "Cached response")
        self.mock_cache_instance.get.assert_called_once()
        self.mock_chat.assert_not_called()
        self.mock_cache_instance.set.assert_not_called()

    def test_call_with_use_cache_false(self):
        """Test that use_cache=False makes an API call and does not save to cache."""
        self.mock_chat.return_value = ChatResponse(
            model="llama2",
            created_at="",
            message=Message(role="assistant", content="Fresh response"),
            done=True,
        )
        client = Client()
        client.call(model="llama2", prompt="No cache", use_cache=False)

        self.mock_cache_instance.get.assert_not_called()  # The key is still checked
        self.mock_chat.assert_called_once()
        self.mock_cache_instance.set.assert_not_called()  # But the result is not stored

    def test_stream_with_cache_miss(self):
        """Test a standard stream that results in a cache miss and stores the result."""
        self.mock_chat.return_value = iter(
            [
                ChatResponse(
                    message=Message(role="assistant", content="Hello, "),
                    done=False,
                    model="l2",
                    created_at="",
                ),
                ChatResponse(
                    message=Message(role="assistant", content="world!"),
                    done=True,
                    model="l2",
                    created_at="",
                ),
            ]
        )
        client = Client()
        responses = list(client.stream(model="llama2", prompt="Hello, world!"))

        self.assertEqual(len(responses), 2)
        self.assertEqual(responses[0].content, "Hello, ")
        self.mock_cache_instance.get.assert_called_once()
        self.mock_chat.assert_called_once()
        self.mock_cache_instance.set.assert_called_once()

    def test_stream_with_cache_hit(self):
        """Test that a stream retrieves a response from the cache."""
        cached_stream = [
            ChatResponse(
                message=Message(role="assistant", content="Cached "),
                done=False,
                model="l2",
                created_at="",
            ),
            ChatResponse(
                message=Message(role="assistant", content="stream"),
                done=True,
                model="l2",
                created_at="",
            ),
        ]
        self.mock_cache_instance.get.return_value = cached_stream
        client = Client()

        responses = list(client.stream(model="llama2", prompt="Cache me"))

        self.assertEqual(len(responses), 2)
        self.assertEqual(responses[0].content, "Cached ")
        self.mock_cache_instance.get.assert_called_once()
        self.mock_chat.assert_not_called()
        self.mock_cache_instance.set.assert_not_called()


if __name__ == "__main__":
    unittest.main()
