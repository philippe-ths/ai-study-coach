import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capstone import (  # noqa: E402
    SVM_CSV_PATH,
    SVM_FEATURE_COLUMNS,
    Student,
    StudySession,
    load_training_data,
    run_svm_prediction,
    student_feature_vector,
    train_svm,
)


def _student_with_data() -> Student:
    s = Student(name="Ada", course="AI Eng", goals="pass")
    s.add_quiz_score(75)
    s.add_quiz_score(85)
    s.add_session(StudySession(date(2026, 4, 10), 60, "Python", 3))
    s.add_session(StudySession(date(2026, 4, 11), 90, "Stats", 4))
    return s


def test_training_csv_exists_and_loads():
    assert SVM_CSV_PATH.exists(), "study_risk.csv must ship next to capstone.py"
    X, y = load_training_data(SVM_CSV_PATH)
    assert list(X.columns) == SVM_FEATURE_COLUMNS
    assert len(X) == len(y) > 0
    assert set(y.unique()) == {"at risk", "on track"}


def test_student_feature_vector_shape_and_values():
    student = _student_with_data()
    vec = student_feature_vector(student)
    assert vec is not None
    assert vec.shape == (1, 4)
    assert vec[0, 0] == pytest.approx(2.5)       # 150 minutes / 60
    assert vec[0, 1] == pytest.approx(80.0)      # avg of 75, 85
    assert vec[0, 2] == pytest.approx(3.5)       # avg difficulty
    assert vec[0, 3] == 2                        # session count


def test_student_feature_vector_none_when_no_sessions():
    s = Student(name="Bo", course="AI Eng", goals="-")
    s.add_quiz_score(80)
    assert student_feature_vector(s) is None


def test_student_feature_vector_none_when_no_quiz_scores():
    s = Student(name="Cai", course="AI Eng", goals="-")
    s.add_session(StudySession(date(2026, 4, 10), 60, "Python", 3))
    assert student_feature_vector(s) is None


def test_train_svm_returns_fitted_model_and_metrics():
    X, y = load_training_data(SVM_CSV_PATH)
    model, accuracy, report = train_svm(X, y)
    assert isinstance(model, SVC)
    assert 0.0 <= accuracy <= 1.0
    assert "precision" in report
    # Predicts on a known healthy feature vector — exercises the fitted model.
    pred = model.predict(
        pd.DataFrame(
            [[25.0, 85.0, 3.0, 10]],
            columns=SVM_FEATURE_COLUMNS,
        )
    )
    assert pred[0] in {"at risk", "on track"}


def test_run_svm_prediction_happy_path(capsys):
    student = _student_with_data()
    run_svm_prediction(student)
    out = capsys.readouterr().out
    assert "SVM risk prediction" in out
    assert "Test accuracy:" in out
    assert "Classification report:" in out
    assert "Prediction for Ada:" in out


def test_run_svm_prediction_insufficient_data(capsys):
    student = Student(name="Dev", course="AI Eng", goals="-")
    run_svm_prediction(student)
    out = capsys.readouterr().out
    assert "Test accuracy:" in out
    assert "Not enough data to predict" in out


def test_run_svm_prediction_missing_csv(monkeypatch, capsys):
    import capstone

    monkeypatch.setattr(capstone, "SVM_CSV_PATH", Path("/nonexistent/study_risk.csv"))
    student = Student(name="Dev", course="AI Eng", goals="-")
    run_svm_prediction(student)
    out = capsys.readouterr().out
    assert "study_risk.csv" in out
    assert "skipping SVM section" in out
    assert "Test accuracy:" not in out
