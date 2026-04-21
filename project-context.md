# Project Context: AI Study Coach

Version: 1.0.0

## Product Summary

- AI Study Coach is a single-file Python terminal program that collects a student's study data, summarises it with charts, predicts study risk with an SVM classifier, and answers study questions via a LangChain Coach Chat.
- The primary users are developers and evaluators who run the script to see five Python capability areas in action.
- The core flow is a sequential terminal session: input collection → Pandas summary + Matplotlib charts → SVM prediction → LangChain Coach Chat (memory + RAG).

## Domain Concepts

- `Person` is a base class holding name and course; `Student` inherits from it, adding goals, quiz scores, and study sessions.
- Quiz scores are encapsulated as a private list with a read-only property and a validating mutator.
- `StudySession` is a single study event with a date, duration in minutes, topic, and a numeric difficulty rating (1–5).
- `StudyMetrics` aggregates a student's sessions into summary values used as the SVM feature vector.
- A `Student` owns zero or more `StudySession` records; aggregated metrics become the SVM input for a single prediction.

## Scope

- Terminal-based input collection with validation and re-prompting on invalid input.
- OOP domain model: `Person`, `Student`, `StudySession` with encapsulation and inheritance.
- Pandas DataFrame of study sessions with computed summary metrics printed to the terminal.
- Two Matplotlib chart types: bar chart of minutes by topic and line chart of minutes over time.
- SVM classifier (SVC) trained on `study_risk.csv` with train/test split, accuracy, classification report, and a per-student prediction.
- LangChain `ConversationChain` with `ConversationBufferMemory` (prints `memory.buffer`) and a FAISS-backed `RetrievalQA` over 10 in-code text chunks answering two questions.
- Graceful degradation: all non-LLM sections complete successfully without an `OPENAI_API_KEY`.
- No web UI, no database, no persistent storage, no authentication, no multi-user support.

## Important Constraints

- Submission is a single `capstone.py` plus `study_risk.csv`, zipped as `PM_python_coding.zip`; no other files are included.
- The script must run end-to-end without crashing on bad input or a missing API key.
- The tech stack is fixed: Python 3.x, Pandas, NumPy, Matplotlib, scikit-learn SVC, LangChain, FAISS, OpenAI.
- All training data is synthetic; no real student data is used.

## Architecture Summary

- Single-file procedural script with sequential sections and no menus or loops back to the start.
- One runtime layer: a terminal process calling local libraries and optionally the OpenAI API.
- Primary data flow: user input → domain objects → Pandas DataFrame + charts → SVM prediction → LangChain Coach Chat.
- The only external service boundary is the OpenAI API, reached via LangChain, guarded by an `OPENAI_API_KEY` check.

## Key Dependencies

- `pandas`: structures study sessions as a DataFrame and computes summary metrics.
- `numpy`: numerical support for Pandas and scikit-learn operations.
- `matplotlib`: renders bar and line charts in the terminal session.
- `scikit-learn`: provides SVC, train/test split, and evaluation metrics.
- `langchain`, `langchain-openai`, `langchain-community`: builds the conversational memory chain and the retrieval QA chain.
- `faiss-cpu`: backs the vector store for the RAG knowledge base.
- `openai`: LLM provider used via LangChain for chat and RAG answers.

## Project Structure

- `capstone.py`: single-file script containing all five sections and the module docstring.
- `study_risk.csv`: synthetic labelled dataset read by the SVM section; columns are `total_hours`, `avg_quiz_score`, `avg_difficulty`, `session_count`, `label` (`"at risk"` or `"on track"`).
- `tests/`: Pytest suite covering M2 (DataFrame, charts), M3 (SVM), and M4 offline paths; excluded from the submission zip.
- `tests/conftest.py`: forces the `Agg` matplotlib backend so tests are non-interactive.
- `requirements-dev.txt`: pinned local dev dependencies; excluded from the submission zip.
- `README.md`: setup and usage instructions for the public repo.
- `project-context.md` (this file): current implementation truth; replaces `project-spec.md`.
- `ai-workflow.md`: AI coding workflow rules imported by `CLAUDE.md`.
- `.claude/skills/`: local skill definitions used by the AI agent.
- `.ai-policy/`, `.githooks/`, `.github/`: policy scripts and CI scaffolding; excluded from the submission zip.

## Testing Overview

- Pytest suite runs via `.venv/bin/pytest tests/`.
- M2 coverage: DataFrame shape and columns, summary metric values, bar and line chart construction, smoke and empty-data paths for `run_data_and_charts`.
- M3 coverage: CSV load shape, student feature-vector values and insufficient-data cases, SVC training returns a fitted model with in-range accuracy, happy-path and insufficient-data runs of `run_svm_prediction`.
- M4 coverage (offline only): knowledge-base size (≥8) and shape, graceful-degradation path of `run_coach_chat` when `OPENAI_API_KEY` is unset.
- M1 (input collection) has no automated tests; M4 LLM-calling path is intentionally manual-only to avoid API calls in CI.

## Maintenance Checklist

- Update this file when the data model, section flow, dependencies, or test coverage change.
- Keep entries aligned with the current `capstone.py` and CSV contents, not planned features.
- Bump the `Version` header on every material change.
