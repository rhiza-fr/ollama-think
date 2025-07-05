from collections.abc import Iterator
from typing import override

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

    def to_json(self) -> str:
        """
        Returns a JSON representation of the object.
        """
        return self.model_dump_json()

    def to_dict(self) -> dict:
        """
        Returns a dictionary representation of the object.
        """
        return self.model_dump()

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
