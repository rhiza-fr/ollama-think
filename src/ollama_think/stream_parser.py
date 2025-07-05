import re
from collections.abc import Generator


class StreamingParser:
    """
    A generalized, continuously-streaming parser for LLM responses.

    Frankly a lot of work to support some older models in streaming mode.
    Uses a state machine that handles complex format patterns that are defined
    as a regex with named capture groups.
    """

    def __init__(self, format_pattern: str):
        self.format_pattern = format_pattern
        self.plan = self._compile_format(format_pattern)
        self.reset()

    def reset(self):
        self._buffer = ""
        self._plan_index = 0
        self._capturing_name = None

    @staticmethod
    def _get_boundary_char(regex_str: str) -> str | None:
        """
        Find the first literal character in a regex,
        """
        i = 0
        while i < len(regex_str):
            char = regex_str[i]
            if char == "\\":
                # The next character is an escaped literal.
                if i + 1 < len(regex_str):
                    return regex_str[i + 1]
                i += 2
            # These are common regex metacharacters to skip over.
            elif char in ".^$*+?{}|()":
                i += 1
            # We found a literal character.
            else:
                return char
        return None

    def _compile_format(self, pattern: str) -> list[tuple[str, ...]]:
        """
        Splits separators between capture groups into an 'end_marker' for the
        current capture and a 'find_marker' for the next.
        """
        group_pattern = r"\(\?P<([a-zA-Z_][a-zA-Z0-9_]*)>.*?\)"
        parts = re.split(f"({group_pattern})", pattern)
        plan = []

        if not parts:
            return []

        if parts[0]:
            plan.append(("FIND", parts[0]))

        for i in range(1, len(parts), 3):
            name = parts[i + 1]
            if name not in ["thinking", "content"]:
                raise ValueError(f"Unsupported capture group name: '{name}'")

            separator = parts[i + 2] if (i + 2) < len(parts) else ""
            end_marker = separator
            next_find_marker = ""

            # Heuristic to split a separator like '</think>\s*<response>'
            # We look for what looks like the start of the *next* tag.
            if separator:
                next_tag_start = self._get_boundary_char(separator)
                if next_tag_start:
                    # Find the last occurrence of the boundary char's start.
                    # This handles separators like '</tag1></tag2>'
                    split_indices = [
                        m.start() for m in re.finditer(re.escape(next_tag_start), separator)
                    ]
                    if len(split_indices) > 1:
                        # If separator is '</t>\s*<r>', next_tag_start is '<'.
                        # There are two '<'. We want to split before the second one.
                        split_point = split_indices[-1]
                        end_marker = separator[:split_point]
                        next_find_marker = separator[split_point:]

            plan.append(("CAPTURE", name, end_marker))
            if next_find_marker:
                plan.append(("FIND", next_find_marker))

        return plan

    def _internal_processor(
        self, chunk: str | None = None
    ) -> Generator[tuple[str, str], None, None]:
        """
        The core state machine logic for parsing the stream.

        This generator function processes the internal buffer, advancing through a
        pre-compiled 'plan' of parsing steps (finding markers and capturing
        content). It yields captured data as `(name, text)` tuples, where `name`
        is either 'thinking' or 'content'.

        It's designed to be called repeatedly. On each call, it consumes as much
        of the buffer as possible. If it can't make progress (e.g., waiting for
        an end-marker to appear in the stream), it returns, preserving its state
        for the next call.

        Args:
            chunk: An optional string chunk to add to the buffer before processing.
                   If None, it processes the existing buffer (used by `finalize`).

        Yields:
            Tuples of (str, str) representing `(capture_group_name, captured_text)`.
        """
        # Add the new data chunk to our internal buffer.
        if chunk:
            self._buffer += chunk

        # Loop continuously as long as we can make progress through the buffer.
        # This allows consuming multiple plan steps from a single large chunk.
        while True:
            # Check if we have completed all steps in the parsing plan.
            if self._plan_index >= len(self.plan):
                # If the plan is finished but we are in a "capture-to-end" state,
                # yield any remaining data in the buffer.
                if self._capturing_name and self._buffer:
                    yield self._capturing_name, self._buffer
                    self._buffer = ""
                return  # Parsing is complete.

            # Get the current instruction (action and details) from the plan.
            action, *details = self.plan[self._plan_index]
            # Record buffer size to detect if we're stalled.
            original_buffer_len = len(self._buffer)

            # --- STATE: FIND ---
            # In this state, we are looking for a literal separator/marker.
            if action == "FIND":
                self._capturing_name = None  # Not capturing content in this state.
                (marker_regex,) = details

                # Search for the marker at the very beginning of the buffer.
                # re.DOTALL allows markers like `\s*` to match newlines.
                match = re.search("^" + marker_regex, self._buffer, re.DOTALL)
                if match:
                    # If found, consume the marker from the buffer.
                    self._buffer = self._buffer[match.end() :]
                    # Advance to the next step in our plan.
                    self._plan_index += 1
                else:
                    # Marker not found. We need more data. Exit the loop and
                    # wait for the next chunk.
                    return

            # --- STATE: CAPTURE ---
            # In this state, we are capturing text into a named group.
            elif action == "CAPTURE":
                capture_name, end_marker_regex = details
                self._capturing_name = capture_name

                # Handle "capture-to-end" case where no end marker is defined.
                if not end_marker_regex:
                    if self._buffer:
                        # Yield everything currently in the buffer.
                        yield capture_name, self._buffer
                        self._buffer = ""
                    # We can't know if the stream is over, so we wait for more
                    # chunks or for `finalize()` to be called.
                    return

                # Search for the end marker anywhere in the current buffer.
                match = re.search(end_marker_regex, self._buffer, re.DOTALL)
                if match:
                    # --- End marker found: The capture is complete. ---
                    # The text to yield is everything up to the start of the marker.
                    text_to_yield = self._buffer[: match.start()]
                    if text_to_yield:
                        yield capture_name, text_to_yield

                    # Consume the captured text AND the end marker from the buffer.
                    self._buffer = self._buffer[match.end() :]
                    # Advance to the next step in the plan.
                    self._plan_index += 1
                    self._capturing_name = None  # Exit capture state.
                else:
                    # --- End marker not found: Stream is in progress. ---
                    # We must yield what we can without yielding a partial marker.
                    # Heuristic 1: Find a "boundary character" of the end marker.
                    boundary_char = self._get_boundary_char(end_marker_regex)

                    if boundary_char:
                        # Find the last occurrence of this boundary char. This is a
                        # conservative split point.
                        split_pos = self._buffer.rfind(boundary_char)
                        if split_pos != -1:
                            # Yield everything before this potential start of a marker.
                            text_to_yield = self._buffer[:split_pos]
                            # Keep the rest in the buffer.
                            self._buffer = self._buffer[split_pos:]
                            if text_to_yield:
                                yield capture_name, text_to_yield
                            # We've yielded what we can for now. Wait for more data.
                            return

                    # Heuristic 2 (Fallback): If no boundary char, keep a small
                    # safety margin at the end of the buffer.
                    # This prevents yielding the start of a marker like `</th` from `</think>`.
                    yield_point = max(0, len(self._buffer) - (len(end_marker_regex) + 2))
                    text_to_yield = self._buffer[:yield_point]
                    self._buffer = self._buffer[yield_point:]
                    if text_to_yield:
                        yield capture_name, text_to_yield
                    # We've yielded what we can. Wait for more data.
                    return

            # If the loop completes an iteration without consuming any data from
            # the buffer, it means we are stalled. Break the loop and wait for
            # the next chunk to arrive.
            if len(self._buffer) == original_buffer_len:
                break

    def process_chunk(self, chunk: str) -> Generator[tuple[str, str], None, None]:
        """
        Processes a new chunk of text from the stream.

        This method consumes the internal generator and accumulates all fully
        parsed 'thinking' and 'content' parts that can be determined from the
        given chunk. It yields at most one tuple of (thinking, content).

        Args:
            chunk: The string chunk to process.

        Yields:
            A single tuple of (str, str) containing all the thinking and content
            that was completed by processing this chunk.
        """
        thinking_parts = []
        content_parts = []
        for name, text in self._internal_processor(chunk):
            if name == "thinking":
                thinking_parts.append(text)
            elif name == "content":
                content_parts.append(text)

        if thinking_parts or content_parts:
            yield "".join(thinking_parts), "".join(content_parts)

    def finalize(self) -> Generator[tuple[str, str], None, None]:
        """Flushes any remaining text from the buffer and yields a final result."""
        thinking_parts = []
        content_parts = []
        for name, text in self._internal_processor():
            if name == "thinking":
                thinking_parts.append(text)
            elif name == "content":
                content_parts.append(text)

        if self._capturing_name and self._buffer:
            if self._capturing_name == "thinking":
                thinking_parts.append(self._buffer)
            elif self._capturing_name == "content":
                content_parts.append(self._buffer)
            self._buffer = ""

        # If no parts were ever captured and there's still data in the buffer,
        # assume it's all content. This handles patterns that don't match at all.
        if not thinking_parts and not content_parts and self._buffer:
            content_parts.append(self._buffer)
            self._buffer = ""

        if thinking_parts or content_parts:
            yield "".join(thinking_parts), "".join(content_parts)
