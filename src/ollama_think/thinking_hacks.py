import re

from ollama._types import ChatRequest

from ollama_think.hack_config import THINKING_HACKS
from ollama_think.stream_parser import StreamingParser
from ollama_think.thinkresponse import ThinkResponse


def hack_request(cr: ChatRequest) -> ChatRequest:
    """
    Modify a ChatRequest object to enable thinking hacks based on the model name.

    Args:
        cr: A ChatRequest object to be modified.

    Returns:
        A modified ChatRequest object with thinking hacks applied.
    """
    for key in THINKING_HACKS.keys():
        if cr.model.startswith(key):
            if THINKING_HACKS[key].get("enable_thinking", False) is False:
                cr.think = False
            new_message = THINKING_HACKS[key].get("add_message", "")
            if new_message and cr.messages:
                messages = [new_message]
                for message in cr.messages:
                    messages.append(message)
                cr.messages = messages
            break
    return cr


def hack_response(model: str, tr: ThinkResponse) -> ThinkResponse:
    """
    Parse a ThinkResponse object to extract thinking content based on the model name.

    Args:
        model: The name of the model.
        tr: A ThinkResponse object to be processed.

    Returns:
        A ThinkResponse object with extracted thinking content.
    """
    for key in THINKING_HACKS.keys():
        if model.startswith(key):
            regexes: list[str] = THINKING_HACKS[key].get("content_parsers", [])
            if regexes:
                for regex in regexes:
                    match = re.search(pattern=str(regex), string=tr.content, flags=re.DOTALL)
                    if match:
                        tr.message.thinking = match.group("thinking").strip()
                        tr.message.content = match.group("content").strip()
                        break
            break
    return tr


def setup_stream_parser(model: str) -> StreamingParser | None:
    for key in THINKING_HACKS.keys():
        if model.startswith(key):
            regexes: list[str] = THINKING_HACKS[key].get("content_parsers", [])
            if regexes:
                # this will fail half the time for granite3.2 !
                return StreamingParser(format_pattern=str(regexes[0]))
            break
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
