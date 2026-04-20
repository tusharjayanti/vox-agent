"""Tests for ConversationMemory — the in-memory session store."""
import pytest
from voxagent.memory import ConversationMemory
from voxagent.llm import Message


@pytest.fixture
def mem():
    """Fresh ConversationMemory instance for each test.

    We don't use the module-level singleton here because tests would
    share state and order would matter. Each test gets its own clean
    instance.
    """
    return ConversationMemory()


class TestGet:
    def test_empty_session_returns_empty_list(self, mem):
        assert mem.get("no-such-session") == []

    def test_get_returns_copy_not_reference(self, mem):
        """Mutating the returned list must not affect internal state."""
        mem.add("session-1", "user", "hello")
        history = mem.get("session-1")
        history.append(Message(role="assistant", content="tampered"))
        # Internal state should be untouched
        assert len(mem.get("session-1")) == 1


class TestAdd:
    def test_add_user_message(self, mem):
        mem.add("session-1", "user", "How long to return?")
        history = mem.get("session-1")
        assert len(history) == 1
        assert history[0].role == "user"
        assert history[0].content == "How long to return?"

    def test_add_preserves_order(self, mem):
        mem.add("session-1", "user", "first")
        mem.add("session-1", "assistant", "second")
        mem.add("session-1", "user", "third")
        history = mem.get("session-1")
        contents = [m.content for m in history]
        assert contents == ["first", "second", "third"]

    def test_add_assigns_timestamp(self, mem):
        mem.add("session-1", "user", "hello")
        message = mem.get("session-1")[0]
        assert message.timestamp is not None


class TestSessionIsolation:
    """Critical: one session's history must never bleed into another."""

    def test_separate_sessions_have_separate_history(self, mem):
        mem.add("alice", "user", "alice's question")
        mem.add("bob", "user", "bob's question")

        alice_history = mem.get("alice")
        bob_history = mem.get("bob")

        assert len(alice_history) == 1
        assert len(bob_history) == 1
        assert alice_history[0].content == "alice's question"
        assert bob_history[0].content == "bob's question"

    def test_adding_to_one_does_not_affect_another(self, mem):
        mem.add("alice", "user", "hi")
        mem.add("bob", "user", "hello")
        mem.add("alice", "assistant", "alice response")

        assert len(mem.get("alice")) == 2
        assert len(mem.get("bob")) == 1


class TestClear:
    def test_clear_removes_history(self, mem):
        mem.add("session-1", "user", "hello")
        mem.add("session-1", "assistant", "hi there")
        assert len(mem.get("session-1")) == 2

        mem.clear("session-1")
        assert mem.get("session-1") == []

    def test_clear_nonexistent_session_is_safe(self, mem):
        # Should not raise
        mem.clear("never-existed")

    def test_clear_does_not_affect_other_sessions(self, mem):
        mem.add("alice", "user", "hi")
        mem.add("bob", "user", "hello")

        mem.clear("alice")

        assert mem.get("alice") == []
        assert len(mem.get("bob")) == 1


class TestSingleton:
    """The module-level `memory` singleton is importable and usable."""

    def test_singleton_is_importable(self):
        from voxagent.memory import memory
        assert isinstance(memory, ConversationMemory)

    def test_singleton_persists_across_imports(self):
        from voxagent.memory import memory as mem1
        from voxagent.memory import memory as mem2
        assert mem1 is mem2
