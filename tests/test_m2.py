import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capstone import (  # noqa: E402
    Student,
    StudySession,
    bar_chart_minutes_by_topic,
    compute_metrics,
    line_chart_minutes_over_time,
    run_data_and_charts,
    sessions_dataframe,
)


def _student_with_sessions() -> Student:
    s = Student(name="Ada", course="AI Eng", goals="pass the course")
    s.add_quiz_score(80)
    s.add_quiz_score(90)
    s.add_session(StudySession(date(2026, 4, 10), 60, "Python", 3))
    s.add_session(StudySession(date(2026, 4, 11), 45, "Python", 2))
    s.add_session(StudySession(date(2026, 4, 12), 90, "Stats", 4))
    return s


def test_sessions_dataframe_shape_and_columns():
    student = _student_with_sessions()
    df = sessions_dataframe(student)
    assert list(df.columns) == ["date", "duration_minutes", "topic", "difficulty"]
    assert len(df) == 3
    assert df["duration_minutes"].sum() == 195
    assert set(df["topic"]) == {"Python", "Stats"}


def test_sessions_dataframe_empty_when_no_sessions():
    student = Student(name="Bo", course="AI Eng", goals="-")
    df = sessions_dataframe(student)
    assert df.empty
    assert list(df.columns) == ["date", "duration_minutes", "topic", "difficulty"]


def test_compute_metrics_values():
    student = _student_with_sessions()
    df = sessions_dataframe(student)
    metrics = compute_metrics(student, df)
    assert metrics.total_hours == pytest.approx(195 / 60)
    assert metrics.average_duration_minutes == pytest.approx(65.0)
    assert metrics.average_quiz_score == pytest.approx(85.0)


def test_compute_metrics_no_quiz_scores():
    student = Student(name="Cai", course="AI Eng", goals="-")
    student.add_session(StudySession(date(2026, 4, 10), 30, "Python", 1))
    df = sessions_dataframe(student)
    metrics = compute_metrics(student, df)
    assert metrics.average_quiz_score is None
    assert metrics.total_hours == pytest.approx(0.5)


def test_bar_chart_builds_figure():
    df = sessions_dataframe(_student_with_sessions())
    fig, ax = bar_chart_minutes_by_topic(df)
    assert isinstance(fig, plt.Figure)
    assert ax.get_title() == "Total study minutes by topic"
    # Two distinct topics -> two bars
    assert len(ax.patches) == 2


def test_line_chart_builds_figure():
    df = sessions_dataframe(_student_with_sessions())
    fig, ax = line_chart_minutes_over_time(df)
    assert isinstance(fig, plt.Figure)
    assert ax.get_title() == "Study minutes per session over time"
    lines = ax.get_lines()
    assert len(lines) == 1
    assert len(lines[0].get_xdata()) == 3


def test_run_data_and_charts_smoke(monkeypatch, capsys):
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    student = _student_with_sessions()
    run_data_and_charts(student)
    out = capsys.readouterr().out
    assert "Data and charts" in out
    assert "Total study time" in out
    assert "Average quiz score:      85.0" in out


def test_run_data_and_charts_empty(monkeypatch, capsys):
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    student = Student(name="Dev", course="AI Eng", goals="-")
    run_data_and_charts(student)
    out = capsys.readouterr().out
    assert "skipping data and charts" in out
