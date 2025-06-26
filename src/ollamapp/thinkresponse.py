from typing import Iterator, override

from ollama import ChatResponse


class ThinkResponse(ChatResponse):
    """
    A wrapper around ChatResponse that adds string behavior, a thinking property,
    and supports addition of ThinkResponse objects.
    """

    def __init__(self, cr: ChatResponse) -> None:
        super().__init__(**cr.__dict__)

    def __str__(self) -> str:
        """
        Returns the message content as a string.
        """
        return self.message.content or ""

    def __repr__(self) -> str:
        """
        Returns a string representation of the object, using the message content.
        """
        return f"ThinkResponse(thinking={self.message.thinking!r}, content={self.message.content!r})"

    @override
    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        return iter((self.thinking, self.content))

    @property
    def thinking(self) -> str:
        """
        Returns the thinking content from the message.
        """
        return self.message.thinking or ""

    @property
    def content(self) -> str:
        """
        Returns the content from the message.
        """
        return self.message.content or ""
