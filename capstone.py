"""AI Study Coach.

A single-file terminal program that collects a student's study data,
summarises and visualises it, predicts study risk with an SVM classifier,
and answers study questions via a LangChain Coach Chat with memory and
FAISS-backed retrieval.

How to run
----------
    python capstone.py

The script expects `study_risk.csv` to sit next to `capstone.py` (same
directory). It runs sequentially through four interactive sections:

    1. Input collection        - name, course, goals, quiz scores,
                                 study sessions (with validation and
                                 re-prompting on bad input).
    2. Data summary and charts - Pandas DataFrame, summary metrics, and
                                 two Matplotlib chart types (bar + line).
    3. SVM risk prediction     - SVC trained on study_risk.csv with a
                                 train/test split; prints accuracy, a
                                 classification report, and a single
                                 prediction for the current student.
    4. Coach Chat (LangChain)  - ConversationBufferMemory + ConversationChain
                                 (prints memory.buffer) plus a FAISS-backed
                                 RetrievalQA over 10 in-code study-tip
                                 chunks, answering two questions.

How to set the API key
----------------------
Section 4 calls the OpenAI API via LangChain. Set your key in the shell
before running the script:

    export OPENAI_API_KEY=sk-...
    python capstone.py

If `OPENAI_API_KEY` is not set, section 4 prints a clear message and
exits cleanly; sections 1-3 still run end-to-end without error.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

SVM_CSV_PATH = Path(__file__).resolve().parent / "study_risk.csv"
SVM_FEATURE_COLUMNS = [
    "total_hours",
    "avg_quiz_score",
    "avg_difficulty",
    "session_count",
]
SVM_LABEL_COLUMN = "label"

LLM_MODEL = "gpt-4o-mini"

KNOWLEDGE_BASE: list[str] = [
    "Active recall beats passive re-reading. Close the book, write what you remember, then check what you missed.",
    "Spaced repetition reviews material at expanding intervals (1 day, 3 days, 7 days, 14 days). Anki and similar flashcard tools automate this.",
    "The Pomodoro technique uses 25 minutes of focused study followed by a 5-minute break. After four rounds, take a 15 to 30 minute break.",
    "Sleep consolidates learning. Aim for 7 to 9 hours; pulling an all-nighter before an exam reduces recall the next day.",
    "The Feynman technique: explain a concept in plain language as if to a twelve-year-old. The gaps in your explanation reveal what you do not yet understand.",
    "Interleaving mixes topics within a single session rather than blocking one topic for hours. It improves long-term retention even though it feels harder in the moment.",
    "A consistent, distraction-free study environment cues focus. Phones in another room outperform phones placed face-down on the desk.",
    "Exam technique: read every question before answering, allocate time by marks available, and tackle the questions you know first.",
    "Exercise boosts cognition. Even a 20-minute walk before a study session improves focus for roughly two hours afterward.",
    "Goals should be specific and time-boxed. 'Study more' fails; 'Finish chapter 4 exercises by Friday' succeeds.",
]


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
# Section 2: Data and charts
# ---------------------------------------------------------------------------


@dataclass
class StudyMetrics:
    total_hours: float
    average_duration_minutes: float
    average_quiz_score: float | None


def sessions_dataframe(student: Student) -> pd.DataFrame:
    columns = ["date", "duration_minutes", "topic", "difficulty"]
    if not student.study_sessions:
        return pd.DataFrame({c: pd.Series(dtype=t) for c, t in zip(
            columns, ["datetime64[ns]", "int64", "object", "int64"]
        )})
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp(s.date),
                "duration_minutes": s.duration_minutes,
                "topic": s.topic,
                "difficulty": s.difficulty,
            }
            for s in student.study_sessions
        ],
        columns=columns,
    )


def compute_metrics(student: Student, df: pd.DataFrame) -> StudyMetrics:
    if df.empty:
        total_hours = 0.0
        avg_duration = 0.0
    else:
        total_hours = float(df["duration_minutes"].sum()) / 60.0
        avg_duration = float(df["duration_minutes"].mean())
    scores = student.quiz_scores
    avg_quiz = float(sum(scores) / len(scores)) if scores else None
    return StudyMetrics(
        total_hours=total_hours,
        average_duration_minutes=avg_duration,
        average_quiz_score=avg_quiz,
    )


def bar_chart_minutes_by_topic(df: pd.DataFrame):
    totals = df.groupby("topic")["duration_minutes"].sum().sort_values(ascending=False)
    fig, ax = plt.subplots()
    ax.bar(totals.index.tolist(), totals.values)
    ax.set_title("Total study minutes by topic")
    ax.set_xlabel("Topic")
    ax.set_ylabel("Minutes")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, ax


def line_chart_minutes_over_time(df: pd.DataFrame):
    ordered = df.sort_values("date")
    fig, ax = plt.subplots()
    ax.plot(ordered["date"], ordered["duration_minutes"], marker="o")
    ax.set_title("Study minutes per session over time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Minutes")
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, ax


def run_data_and_charts(student: Student) -> None:
    print()
    print("=" * 60)
    print("Data and charts")
    print("=" * 60)

    df = sessions_dataframe(student)
    if df.empty:
        print("No study sessions captured - skipping data and charts.")
        return

    print("Study sessions (DataFrame):")
    print(df.to_string(index=False))

    metrics = compute_metrics(student, df)
    print()
    print(f"Total study time:        {metrics.total_hours:.2f} hours")
    print(f"Average session length:  {metrics.average_duration_minutes:.1f} minutes")
    if metrics.average_quiz_score is None:
        print("Average quiz score:      none captured")
    else:
        print(f"Average quiz score:      {metrics.average_quiz_score:.1f}")

    bar_chart_minutes_by_topic(df)
    line_chart_minutes_over_time(df)
    plt.show()


# ---------------------------------------------------------------------------
# Section 3: SVM risk prediction
# ---------------------------------------------------------------------------


def load_training_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    X = df[SVM_FEATURE_COLUMNS]
    y = df[SVM_LABEL_COLUMN]
    return X, y


def student_feature_vector(student: Student) -> np.ndarray | None:
    sessions = student.study_sessions
    scores = student.quiz_scores
    if not sessions or not scores:
        return None
    total_hours = sum(s.duration_minutes for s in sessions) / 60.0
    avg_quiz = sum(scores) / len(scores)
    avg_difficulty = sum(s.difficulty for s in sessions) / len(sessions)
    session_count = len(sessions)
    return np.array(
        [[total_hours, avg_quiz, avg_difficulty, session_count]],
        dtype=float,
    )


def train_svm(
    X: pd.DataFrame, y: pd.Series
) -> tuple[SVC, float, str]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, zero_division=0)
    return model, accuracy, report


def run_svm_prediction(student: Student) -> None:
    print()
    print("=" * 60)
    print("SVM risk prediction")
    print("=" * 60)

    try:
        X, y = load_training_data(SVM_CSV_PATH)
    except FileNotFoundError:
        print(f"Training data not found: {SVM_CSV_PATH.name} — skipping SVM section.")
        return
    print(f"Loaded {len(X)} training rows from {SVM_CSV_PATH.name}")

    model, accuracy, report = train_svm(X, y)
    print(f"Test accuracy: {accuracy:.2%}")
    print()
    print("Classification report:")
    print(report)

    vector = student_feature_vector(student)
    if vector is None:
        print(
            "Not enough data to predict for this student "
            "(need at least one study session and one quiz score)."
        )
        return

    features_df = pd.DataFrame(vector, columns=SVM_FEATURE_COLUMNS)
    prediction = model.predict(features_df)[0]
    print(f"Prediction for {student.name}: {prediction}")


# ---------------------------------------------------------------------------
# Section 4: Coach Chat (LangChain memory + RAG)
# ---------------------------------------------------------------------------


def _import_langchain():
    # Cross-version imports: the course tutorial uses the pre-v1 layout
    # (langchain.memory / langchain.chains), current installs split those
    # into langchain-classic. Try both so the submission runs on either.
    try:
        from langchain.memory import ConversationBufferMemory
        from langchain.chains import ConversationChain, RetrievalQA
    except ImportError:
        from langchain_classic.memory import ConversationBufferMemory
        from langchain_classic.chains import ConversationChain, RetrievalQA
    try:
        from langchain_community.vectorstores import FAISS
    except ImportError:
        from langchain.vectorstores import FAISS
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    except ImportError:
        from langchain.chat_models import ChatOpenAI
        from langchain.embeddings import OpenAIEmbeddings
    return (
        ConversationBufferMemory,
        ConversationChain,
        RetrievalQA,
        FAISS,
        ChatOpenAI,
        OpenAIEmbeddings,
    )


def run_coach_chat(student: Student) -> None:
    print()
    print("=== Coach Chat (LangChain) ===")

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; skipping Coach Chat.")
        print("Set the environment variable and rerun to see the LLM sections.")
        return

    (
        ConversationBufferMemory,
        ConversationChain,
        RetrievalQA,
        FAISS,
        ChatOpenAI,
        OpenAIEmbeddings,
    ) = _import_langchain()

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    print()
    print("-- Memory chain --")
    memory = ConversationBufferMemory()
    chat = ConversationChain(llm=llm, memory=memory)
    first = chat.predict(
        input=(
            f"My name is {student.name} and I am studying {student.course}. "
            "Give me one concrete study tip in a single sentence."
        )
    )
    print(f"Coach: {first}")
    follow_up = chat.predict(input="What was my name?")
    print(f"Coach: {follow_up}")
    print()
    print("memory.buffer:")
    print(memory.buffer)

    print()
    print("-- Retrieval QA over the coach knowledge base --")
    vector_store = FAISS.from_texts(KNOWLEDGE_BASE, OpenAIEmbeddings())
    rag = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())
    for question in (
        "What is a good way to memorise material?",
        "How should I structure a focused study session?",
    ):
        answer = rag.invoke({"query": question})["result"]
        print(f"Q: {question}")
        print(f"A: {answer}")
        print()


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
