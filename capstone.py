"""AI Study Coach.

Run:
    python capstone.py

The script runs sequentially through five sections:
    1. Input collection         (implemented)
    2. Data summary and charts  (M2, not yet implemented)
    3. SVM risk prediction      (M3, not yet implemented)
    4. Coach Chat (LangChain)   (M4, not yet implemented)
    5. Docstring and packaging  (M5)

API key:
    The LangChain sections (M4) will use the OPENAI_API_KEY environment
    variable. If it is not set, those sections will print a message and
    the script will still complete.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date


# ---------------------------------------------------------------------------
# Section 1: Domain model
# ---------------------------------------------------------------------------


class Person:
    """Base class for anyone using the coach. Demonstrates inheritance."""

    def __init__(self, name: str, course: str) -> None:
        self.name = name
        self.course = course

    def introduce(self) -> str:
        return f"{self.name} studying {self.course}"


@dataclass
class StudySession:
    """A single study session. Difficulty is a 1-5 rating."""

    date: date
    duration_minutes: int
    topic: str
    difficulty: int

    def __str__(self) -> str:
        return (
            f"{self.date.isoformat()}  "
            f"{self.duration_minutes:>3d} min  "
            f"difficulty {self.difficulty}/5  "
            f"{self.topic}"
        )


class Student(Person):
    """A student using the coach. Extends Person and encapsulates quiz scores."""

    def __init__(self, name: str, course: str, goals: str) -> None:
        super().__init__(name, course)
        self.goals = goals
        self._quiz_scores: list[float] = []
        self.study_sessions: list[StudySession] = []

    @property
    def quiz_scores(self) -> list[float]:
        # Return a copy so callers cannot mutate the private list.
        return list(self._quiz_scores)

    def add_quiz_score(self, score: float) -> None:
        if not 0 <= score <= 100:
            raise ValueError("quiz score must be between 0 and 100")
        self._quiz_scores.append(score)

    def add_session(self, session: StudySession) -> None:
        self.study_sessions.append(session)


# ---------------------------------------------------------------------------
# Section 1: Input helpers
# ---------------------------------------------------------------------------


def prompt_non_empty_string(prompt: str) -> str:
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("  Please enter a non-empty value.")


def prompt_int(prompt: str, minimum: int, maximum: int) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
        except ValueError:
            print(f"  Please enter a whole number between {minimum} and {maximum}.")
            continue
        if not minimum <= value <= maximum:
            print(f"  Please enter a whole number between {minimum} and {maximum}.")
            continue
        return value


def prompt_float(prompt: str, minimum: float, maximum: float) -> float:
    while True:
        raw = input(prompt).strip()
        try:
            value = float(raw)
        except ValueError:
            print(f"  Please enter a number between {minimum} and {maximum}.")
            continue
        if not minimum <= value <= maximum:
            print(f"  Please enter a number between {minimum} and {maximum}.")
            continue
        return value


def prompt_date(prompt: str) -> date:
    while True:
        raw = input(prompt).strip()
        try:
            return date.fromisoformat(raw)
        except ValueError:
            print("  Please enter a date in YYYY-MM-DD format (e.g. 2026-04-18).")


def prompt_yes_no(prompt: str) -> bool:
    while True:
        raw = input(prompt).strip().lower()
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("  Please answer 'y' or 'n'.")


# ---------------------------------------------------------------------------
# Section 1: Input collection flow
# ---------------------------------------------------------------------------


def collect_student() -> Student:
    print("=" * 60)
    print("AI Study Coach - Tell me about yourself")
    print("=" * 60)

    name = prompt_non_empty_string("Your name: ")
    course = prompt_non_empty_string("Course you are studying: ")
    goals = prompt_non_empty_string("What are your study goals? ")

    student = Student(name=name, course=course, goals=goals)

    print()
    print("-- Quiz scores ------------------------------------------")
    i = 1
    while prompt_yes_no(
        "Add a quiz score? (y/n): " if i == 1 else "Add another quiz score? (y/n): "
    ):
        score = prompt_float(f"  Quiz {i} score (0-100): ", 0, 100)
        student.add_quiz_score(score)
        i += 1

    print()
    print("-- Recent study sessions --------------------------------")
    i = 1
    while prompt_yes_no(
        "Add a study session? (y/n): "
        if i == 1
        else "Add another study session? (y/n): "
    ):
        print(f"  Session {i}:")
        session_date = prompt_date("    Date (YYYY-MM-DD): ")
        duration = prompt_int("    Duration in minutes: ", 1, 24 * 60)
        topic = prompt_non_empty_string("    Topic: ")
        difficulty = prompt_int("    Difficulty (1=easy, 5=hard): ", 1, 5)
        student.add_session(
            StudySession(
                date=session_date,
                duration_minutes=duration,
                topic=topic,
                difficulty=difficulty,
            )
        )
        i += 1

    return student


def print_summary(student: Student) -> None:
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Name:   {student.name}")
    print(f"Course: {student.course}")
    print(f"Goals:  {student.goals}")

    scores = student.quiz_scores
    if scores:
        avg = sum(scores) / len(scores)
        print(f"Quiz scores ({len(scores)}): {scores}  avg={avg:.1f}")
    else:
        print("Quiz scores: none captured")

    if student.study_sessions:
        total_min = sum(s.duration_minutes for s in student.study_sessions)
        print(
            f"Study sessions ({len(student.study_sessions)}): "
            f"total {total_min} min"
        )
        for s in student.study_sessions:
            print(f"  - {s}")
    else:
        print("Study sessions: none captured")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Later sections (not yet implemented)
# ---------------------------------------------------------------------------


def run_data_and_charts(student: Student) -> None:
    # M2: Pandas DataFrame, summary metrics, Matplotlib charts.
    pass


def run_svm_prediction(student: Student) -> None:
    # M3: Train SVC on CSV, print accuracy + classification report, predict.
    pass


def run_coach_chat(student: Student) -> None:
    # M4: LangChain conversational chain with memory + FAISS RAG.
    # Graceful degradation when OPENAI_API_KEY is unset.
    pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    try:
        student = collect_student()
    except (EOFError, KeyboardInterrupt):
        print()
        print("Input cancelled. Exiting.")
        return 1

    print_summary(student)

    run_data_and_charts(student)
    run_svm_prediction(student)
    run_coach_chat(student)
    return 0


if __name__ == "__main__":
    sys.exit(main())
