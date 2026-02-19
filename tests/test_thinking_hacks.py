# test_thinking_hacks.py
from unittest.mock import MagicMock

from ollama import ChatResponse, Message
from ollama._types import ChatRequest

from ollama_think.config import ThinkingHacks
from ollama_think.stream_parser import StreamingParser
from ollama_think.thinking_hacks import (
    hack_request,
    hack_response,
    hack_stream_chunk,
    setup_stream_parser,
)
from ollama_think.thinkresponse import ThinkResponse

THINK_PATTERN = "<think>(?P<thinking>.*?)</think>(?P<content>.*)"


def _hacks(**overrides) -> ThinkingHacks:
    base: ThinkingHacks = {"enable_thinking": False, "add_message": None, "content_parsers": [THINK_PATTERN]}
    base.update(overrides)
    return base


def _tr(**kwargs) -> ThinkResponse:
    kwargs.setdefault("model", "test")
    kwargs.setdefault("created_at", "")
    kwargs.setdefault("done", False)
    kwargs.setdefault("message", Message(role="assistant", content=""))
    return ThinkResponse(ChatResponse(**kwargs))


# --- hack_request ---


def test_hack_request_disables_thinking_when_enable_thinking_false():
    cr = ChatRequest(model="test", messages=[Message(role="user", content="hi")], think=True)
    result = hack_request(cr, _hacks(enable_thinking=False))
    assert result.think is False


def test_hack_request_leaves_thinking_when_enable_thinking_true():
    cr = ChatRequest(model="test", messages=[Message(role="user", content="hi")], think=True)
    result = hack_request(cr, _hacks(enable_thinking=True))
    assert result.think is True


def test_hack_request_prepends_add_message():
    add_msg = {"role": "system", "content": "Enable deep thinking subroutine."}
    cr = ChatRequest(model="test", messages=[Message(role="user", content="hi")])
    result = hack_request(cr, _hacks(add_message=add_msg))
    assert len(result.messages) == 2
    assert result.messages[0]["content"] == "Enable deep thinking subroutine."
    assert result.messages[1].role == "user"


def test_hack_request_no_add_message_leaves_messages_unchanged():
    cr = ChatRequest(model="test", messages=[Message(role="user", content="hi")])
    result = hack_request(cr, _hacks(add_message=None))
    assert len(result.messages) == 1


# --- hack_response ---


def test_hack_response_extracts_thinking_and_content():
    tr = _tr(message=Message(role="assistant", content="<think>a thought</think>the answer"))
    result = hack_response(tr, _hacks())
    assert result.thinking == "a thought"
    assert result.content == "the answer"


def test_hack_response_no_match_leaves_unchanged():
    tr = _tr(message=Message(role="assistant", content="no tags here"))
    result = hack_response(tr, _hacks())
    assert result.content == "no tags here"
    assert result.thinking == ""


def test_hack_response_empty_parsers_leaves_unchanged():
    tr = _tr(message=Message(role="assistant", content="some content"))
    result = hack_response(tr, _hacks(content_parsers=[]))
    assert result.content == "some content"


# --- setup_stream_parser ---


def test_setup_stream_parser_returns_none_when_no_hacks():
    assert setup_stream_parser("any-model", None) is None


def test_setup_stream_parser_returns_parser_when_parsers_exist():
    result = setup_stream_parser("cogito", _hacks())
    assert isinstance(result, StreamingParser)


def test_setup_stream_parser_returns_none_when_empty_parsers():
    result = setup_stream_parser("granite3.2-vision", _hacks(content_parsers=[]))
    assert result is None


# --- hack_stream_chunk ---


def test_hack_stream_chunk_non_done_yields_result():
    sp = MagicMock()
    sp.process_chunk.return_value = iter([("a thought", "an answer")])
    tr = _tr(message=Message(role="assistant", content="raw chunk"), done=False)
    result = hack_stream_chunk(tr, sp)
    assert result is not None
    assert result.message.thinking == "a thought"
    assert result.message.content == "an answer"


def test_hack_stream_chunk_non_done_no_output_returns_none():
    sp = MagicMock()
    sp.process_chunk.return_value = iter([])
    tr = _tr(message=Message(role="assistant", content="<think>"), done=False)
    assert hack_stream_chunk(tr, sp) is None


def test_hack_stream_chunk_done_finalizes_response():
    sp = MagicMock()
    sp.finalize.return_value = iter([("final thought", "final answer")])
    tr = _tr(message=Message(role="assistant", content=""), done=True)
    result = hack_stream_chunk(tr, sp)
    assert result is tr
    assert result.message.thinking == "final thought"
    assert result.message.content == "final answer"


def test_hack_stream_chunk_done_no_finalize_returns_tr_unchanged():
    sp = MagicMock()
    sp.finalize.return_value = iter([])
    tr = _tr(message=Message(role="assistant", content=""), done=True)
    result = hack_stream_chunk(tr, sp)
    assert result is tr
