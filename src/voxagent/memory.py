"""
In-memory conversation store.

Keeps per-session message history in a plain Python dict. This is
v1-appropriate: ephemeral, fast, simple. For production, swap in a
Postgres-backed or Redis-backed implementation behind the same
interface — agent.py and main.py don't need to know which backend
is active.

The get() method returns a COPY of the internal list to prevent
callers from mutating state without going through add(). Callers
that want to send the history to an LLM can use the returned list
directly; callers that want to modify history must use add().
"""
from voxagent.llm import Message


class ConversationMemory:
    """Session-keyed in-memory conversation history.

    Not thread-safe for concurrent writes to the same session_id, but
    FastAPI's async model means requests are serialized within a
    single worker anyway. If we ever add multi-worker deployment
    (see production roadmap), this needs to move to Postgres or
    Redis behind the same interface.
    """

    def __init__(self) -> None:
        self._store: dict[str, list[Message]] = {}

    def get(self, session_id: str) -> list[Message]:
        """Return a copy of the message history for this session.

        Empty list if the session has no history. Returns a copy so
        callers can iterate or modify the list without affecting
        memory's internal state.
        """
        return self._store.get(session_id, []).copy()

    def add(self, session_id: str, role: str, content: str) -> None:
        """Append a message to this session's history.

        Creates the session's history list on first call.
        """
        if session_id not in self._store:
            self._store[session_id] = []
        self._store[session_id].append(
            Message(role=role, content=content)
        )

    def clear(self, session_id: str) -> None:
        """Remove all history for this session."""
        self._store.pop(session_id, None)


# Module-level singleton — import this, don't instantiate your own.
memory = ConversationMemory()
