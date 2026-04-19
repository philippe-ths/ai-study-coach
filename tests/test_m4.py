import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capstone import (  # noqa: E402
    KNOWLEDGE_BASE,
    Student,
    run_coach_chat,
)


def test_knowledge_base_has_at_least_eight_chunks():
    assert len(KNOWLEDGE_BASE) >= 8
    assert all(isinstance(chunk, str) and chunk.strip() for chunk in KNOWLEDGE_BASE)


def test_run_coach_chat_without_api_key(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    student = Student(name="Ada", course="AI Eng", goals="pass")
    run_coach_chat(student)
    out = capsys.readouterr().out
    assert "Coach Chat" in out
    assert "OPENAI_API_KEY not set" in out


def test_run_coach_chat_prints_section_header(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    student = Student(name="Bo", course="AI Eng", goals="-")
    run_coach_chat(student)
    out = capsys.readouterr().out
    assert "=== Coach Chat (LangChain) ===" in out
