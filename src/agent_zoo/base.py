"""Internal shared base interface for agent implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar


class BaseAgent(ABC):
    """Small internal contract shared by concrete agent wrappers."""

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    @abstractmethod
    def ask(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the agent's primary request flow."""
        raise NotImplementedError
