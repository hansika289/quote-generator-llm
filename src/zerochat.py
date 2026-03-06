from dataclasses import dataclass, field
from typing import List

from .config import TrainingConfig
from .generate import generate_quote


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str


@dataclass
class ZeroChatSession:
    """
    Minimal ZeroChat-style wrapper around the quote generator.

    It keeps a simple history of messages and, for each new user message,
    produces a motivational quote-style reply using `generate_quote`.
    """

    cfg: TrainingConfig = field(default_factory=TrainingConfig)
    history: List[Message] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        self.history.append(Message(role="user", content=content))

    def add_assistant_message(self, content: str) -> None:
        self.history.append(Message(role="assistant", content=content))

    def reply(self, user_message: str) -> str:
        """
        Add a user message and return an assistant reply based on the prompt.
        """
        self.add_user_message(user_message)
        response = generate_quote(prompt=user_message, cfg=self.cfg)
        self.add_assistant_message(response)
        return response

