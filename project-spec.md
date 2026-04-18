# Project Spec: AI Study Coach

Version: 0.3.0
Brief: [assignment-2-brief.md](assignment-2-brief.md)
Spec: [assignment-2-spec.md](assignment-2-spec.md)

This file documents the current implementation truth for the AI Study Coach project.
It is a first draft based on the brief and spec; it will be updated as code lands.

## Product Summary
- AI Study Coach is a single-file Python script that demonstrates five areas of Python competency for an Oxford AI Engineering course assignment.
- The primary users are the course assessors, who run the script to verify each required capability is present and functional.
- The core flow is a sequential terminal session that collects student data, summarises it with charts, predicts risk with an SVM classifier, and answers study questions via a LangChain Coach Chat.

## Domain Concepts
- A `Student` holds a name, course, goals, quiz scores, and a collection of study sessions.
- A `StudySession` is a single study event with a date, duration, topic, and difficulty rating.
- The SVM training dataset is a separate CSV of synthetic records labelled "at risk" or "on track".
- The RAG knowledge base is a list of 8+ short text chunks hardcoded in the script.
- A `Student` owns many `StudySession` records; the student's aggregated metrics become the feature vector fed into the SVM for a single prediction.

## Scope
- Terminal-based input collection with validation and re-prompting on invalid input.
- Object-oriented domain model for students and study sessions, including encapsulation and inheritance.
- Pandas DataFrame of study sessions with computed summary metrics printed to the terminal.
- At least two Matplotlib chart types visualising the student's study data.
- SVM classifier trained on a CSV with a train/test split, accuracy, and classification report.
- LangChain conversational chain with memory, demonstrating at least one exchange and printing `memory.buffer`.
- LangChain RAG using FAISS over 8+ in-code text chunks, answering at least two questions.
- Graceful degradation: all non-LLM sections complete successfully without an OpenAI API key.
- No web UI, no database, no persistent storage, no authentication, no multi-user support, no deployment.

## Important Constraints
- Grading is binary (Complete / Not Complete) with one submission attempt, due 25 April 2026.
- Submission must be a single `capstone.py` plus the SVM training CSV, zipped as `PM_python_coding.zip`.
- The script must run end-to-end without crashing on bad input or a missing API key.
- The tech stack is fixed by the course modules: Python 3.x, Pandas, NumPy, Matplotlib, scikit-learn SVC, LangChain, FAISS, OpenAI.
- Implementation patterns should stay consistent with the course tutorial files (`python_basics`, `oop`, `ml_libraries`, `svm`, `langchain`).
- All training data is synthetic; no real student data is used.

## Architecture Summary
- Single-file procedural script with sequential sections, no menus and no loops back to the start.
- One runtime layer: a terminal process that runs Python, calling local libraries and optionally the OpenAI API.
- Primary data flow: user input -> domain objects -> Pandas DataFrame + charts -> SVM prediction -> LangChain Coach Chat.
- The only external service boundary is the OpenAI API, reached via LangChain, guarded by an API key check.

## Key Dependencies
- pandas: Structure study sessions as a DataFrame and compute summary metrics.
- numpy: Numerical support for Pandas and scikit-learn operations.
- matplotlib: Render the required chart types in the terminal session.
- scikit-learn: Provide the SVC classifier, train/test split, and evaluation metrics.
- langchain: Build the conversational memory chain and the retrieval QA chain.
- faiss: Back the vector store for the RAG knowledge base.
- openai: LLM provider used via LangChain for chat and RAG answers.

## Project Structure
- `capstone.py`: Single Python file containing all five sections and the top-level docstring. (Not yet created.)
- SVM training CSV: Synthetic labelled dataset read by the SVM section. (Not yet created; filename to be decided.)
- `assignment-2-brief.md`: Living brief describing the assignment goal.
- `assignment-2-spec.md`: Living detailed spec covering outcomes, scope, constraints, architecture, milestones, and acceptance criteria.
- `project-spec-template.md`: Template this spec was derived from (kept local, not part of the submission).
- `.ai-policy/`, `.agents/`, `.claude/`, `.codex/`, `.gemini/`, `.githooks/`, `.github/`, `ai-workflow.md`, `CLAUDE.md`: AI workflow scaffolding, kept local and excluded from the submission zip.

## Testing Overview
- No automated test framework is in place; the brief does not require unit tests.
- Validation today is manual: run the script end-to-end and check each acceptance criterion (AC1-AC9) is visibly satisfied.
- Major gap: no automated way to catch regressions in input validation, chart rendering, SVM accuracy, or LangChain chain behaviour.

## Versioning
- `project-spec.md` is the living "latest" document and is the only file CLAUDE.md imports.
- Living documents (`project-spec.md`, `assignment-2-spec.md`, `assignment-2-brief.md`) use unversioned filenames.
- The current version is recorded in the `Version: X.Y.Z` header at the top of each living document.
- When releasing a new minor or major version, create a Git tag on the merge commit using the format `<doc-slug>/X.Y.Z` (e.g. `spec/2.1.0`, `project-spec/0.2.0`).
- Git history and tags are the canonical record of past versions; do not create physical snapshot files for living docs.
- Versioned filenames may still be used for frozen documents that will not be updated.
- Historical versions are viewed via `git show <tag>:<path>` or GitHub's browse-at-tag UI; link to historical content with tag-pinned URLs rather than inline filenames.
- Use semantic-ish versioning: patch for wording, minor for added/removed sections or scope changes, major for breaking direction changes.

## Maintenance Checklist
- Update this file when the data model, section flow, dependencies, or acceptance criteria change.
- Keep this file aligned with the current `capstone.py` and CSV contents, not planned features.
- Keep entries factual and concise; defer narrative rationale to the brief and spec.
- Follow the Versioning rule above on every material change: bump the Version header, edit, and tag the merge commit for minor or major bumps.
