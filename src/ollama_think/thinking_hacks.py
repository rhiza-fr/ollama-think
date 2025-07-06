import re

from ollama._types import ChatRequest

from ollama_think.config import ThinkingHacks
from ollama_think.stream_parser import StreamingParser
from ollama_think.thinkresponse import ThinkResponse


def hack_request(cr: ChatRequest, hacks: ThinkingHacks) -> ChatRequest:
    """
    Modify a ChatRequest object to enable thinking hacks based on the model name.

    Args:
        cr: A ChatRequest object to be modified.
        hacks: Hacks relevant to this model
    Returns:
        A modified ChatRequest object with thinking hacks applied.
    """
    if hacks.get("enable_thinking", False) is False:
        cr.think = False
    new_message = hacks.get("add_message", "")
    if new_message and cr.messages:
        messages = [new_message]
        for message in cr.messages:
            messages.append(message)  # type: ignore hmmm
        cr.messages = messages
    return cr


def hack_response(tr: ThinkResponse, hacks: ThinkingHacks) -> ThinkResponse:
    """
    Parse a ThinkResponse object to extract thinking content based on the model name.

    Args:
        tr: A ThinkResponse object to be processed.
        hacks: Hacks relevant to this model
    Returns:
        A ThinkResponse object with extracted thinking content.
    """
    regexes: list[str] = hacks.get("content_parsers", [])
    if regexes:
        for regex in regexes:
            match = re.search(pattern=str(regex), string=tr.content, flags=re.DOTALL)
            if match:
                tr.message.thinking = match.group("thinking").strip()
                tr.message.content = match.group("content").strip()
                break
    return tr


def setup_stream_parser(model: str, hacks: ThinkingHacks | None) -> StreamingParser | None:
    if not hacks:
        return None

    regexes: list[str] = hacks.get("content_parsers", [])
    if regexes:
        # Prefer the last stream parse (granite3.2 inverses the order when streaming !?)
        return StreamingParser(format_pattern=str(regexes[-1]))
    return None


def hack_stream_chunk(tr: ThinkResponse, sp: StreamingParser) -> ThinkResponse | None:
    """
    Processes a raw chunk from the stream using the StreamingParser.

    Takes a raw ThinkResponse chunk, passes its content to the parser,
    and if the parser yields a complete part, it returns a new ThinkResponse object
    with the parsed 'thinking' and 'content' fields populated.

    If a chunk is consumed but does not result in a complete part (e.g., it's just
    a marker like `<think>`), this function returns None.
    """
    if tr.done:
        # Finalize the stream. The parser will yield at most one final tuple.
        for thinking, content in sp.finalize():
            tr.message.thinking = (tr.message.thinking or "") + thinking
            tr.message.content = content if content else ""
        return tr  # Return the final, completed response object
    else:
        # Process the chunk. The parser will yield at most one tuple.
        for thinking, content in sp.process_chunk(str(tr.message.content)):
            # A part was completed. Create a new response object with the parsed data.
            new_tr = tr.model_copy(deep=True)
            new_tr.message.thinking = thinking
            new_tr.message.content = content
            return new_tr
    return None
