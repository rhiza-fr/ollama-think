import pytest

from ollama_think.stream_parser import StreamingParser

# A list of chunk sizes to test against for each scenario.
# 1: Extreme case, tests every character boundary.
# 7: Awkward prime number size, likely to split markers.
# 25: A medium size.
# 1000: A large size, likely to process the whole response in one chunk.
CHUNK_SIZES_TO_TEST = [1, 3, 7, 25, 1000]

# --- Test Scenarios ---
# Each tuple contains:
# 1. A unique test ID/name.
# 2. The format pattern string.
# 3. The full response string to be streamed.
# 4. The expected final 'thinking' string.
# 5. The expected final 'content' string.
TEST_CASES = [
    (
        "standard_xml",
        r"<thinking>(?P<thinking>.*?)</thinking><output>(?P<content>.*?)</output>",
        "<thinking>First, I plan.</thinking><output>This is the final output.</output>",
        "First, I plan.",
        "This is the final output.",
    ),
    (
        "abbreviated_xml_capture_to_end",
        r"<think>(?P<thinking>.*?)</think>(?P<content>.*)",
        "<think>Short thought.</think>The rest of this is all content.",
        "Short thought.",
        "The rest of this is all content.",
    ),
    (
        "plain_text_content_first",
        r"Here is my response:\n(?P<content>.*?)Here is my thought process:\n(?P<thinking>.*)",
        "Here is my response:\nThis is the main answer.\nHere is my thought process:\nI decided to answer first.",
        "I decided to answer first.",
        "This is the main answer.\n",
    ),
    (
        "plain_text_thinking_first",
        r"Here is my thought process:\n(?P<thinking>.*?)Here is my response:\n(?P<content>.*)",
        "Here is my thought process:\nFirst I plan, then I write.\nHere is my response:\nThis is the final response.",
        "First I plan, then I write.\n",
        "This is the final response.",
    ),
    (
        "only_content",
        r"Final Answer: (?P<content>.*)",
        "Final Answer: The only thing here is the answer.",
        "",
        "The only thing here is the answer.",
    ),
    (
        "only_thinking",
        r"My Thoughts:\n(?P<thinking>.*)",
        "My Thoughts:\nThis is just a thought process, no final output.",
        "This is just a thought process, no final output.",
        "",
    ),
    (
        "empty_content",
        r"<thinking>(?P<thinking>.*?)</thinking><content>(?P<content>.*?)</content>",
        "<thinking>This is my thought</thinking><content></content>",
        "This is my thought",
        "",
    ),
    (
        "empty_thinking",
        r"<thinking>(?P<thinking>.*?)</thinking><content>(?P<content>.*?)</content>",
        "<thinking></thinking><content>Content without thought</content>",
        "",
        "Content without thought",
    ),
    (
        "no_prologue",
        r"(?P<content>.*?)---(?P<thinking>.*)",
        "This is content right away.---And this is the thought.",
        "And this is the thought.",
        "This is content right away.",
    ),
    (
        "premature_stream_end",
        r"<thinking>(?P<thinking>.*?)</thinking><content>(?P<content>.*?)</content>",
        "<thinking>This thought is written",
        "This thought is written",
        "",
    ),
    (
        "no_match_at_all",
        r"<thinking>(?P<thinking>.*?)</thinking>",
        "This is just some plain text.",
        "",
        "This is just some plain text.",
    ),
]


@pytest.mark.parametrize(
    "test_id, format_pattern, full_response, expected_thinking, expected_content",
    TEST_CASES,
    ids=[case[0] for case in TEST_CASES],
)
def test_streaming_parser_scenarios(
    test_id, format_pattern, full_response, expected_thinking, expected_content
):
    """
    Tests the StreamingParser against a variety of scenarios and chunk sizes.
    This test is updated for the API that yields (thinking, content) tuples.
    """
    for chunk_size in CHUNK_SIZES_TO_TEST:
        parser = StreamingParser(format_pattern)

        stream = (
            full_response[i : i + chunk_size] for i in range(0, len(full_response), chunk_size)
        )

        actual_thinking = ""
        actual_content = ""

        for chunk in stream:
            for thinking_part, content_part in parser.process_chunk(chunk):
                actual_thinking += thinking_part
                actual_content += content_part

        # Finalize the stream to flush any remaining buffers
        for thinking_part, content_part in parser.finalize():
            actual_thinking += thinking_part
            actual_content += content_part

        assert actual_thinking == expected_thinking, (
            f"Failed thinking for chunk size {chunk_size} in test '{test_id}' with text '{full_response}' and pattern '{format_pattern}'"
        )
        assert actual_content == expected_content, (
            f"Failed content for chunk size {chunk_size} in test '{test_id}' with text '{full_response}' and pattern '{format_pattern}'"
        )


def test_instantiation_failures():
    """
    Tests that the parser raises ValueError for invalid format patterns.
    """
    # Pattern with an unsupported capture group name
    with pytest.raises(ValueError, match="Unsupported capture group name: 'other'"):
        StreamingParser(r"(?P<other>.*?)")


def test_reset_functionality():
    """
    Tests that the reset() method correctly clears the parser's state.
    This test is updated to reflect the internal state of the final parser version.
    """
    format_pattern = r"<t>(?P<thinking>.*?)</t>"
    parser = StreamingParser(format_pattern)

    list(parser.process_chunk("<t>some partial"))

    assert parser._plan_index == 1  # Moved past the initial 'FIND' step
    parser.reset()
    assert parser._buffer == ""
    assert parser._plan_index == 0


def test_empty_stream():
    """
    Tests that the parser handles an empty stream gracefully.
    """
    parser = StreamingParser(r"<t>(?P<thinking>.*)</t>")
    results_process = list(parser.process_chunk(""))
    results_finalize = list(parser.finalize())
    assert results_process == []
    assert results_finalize == []
