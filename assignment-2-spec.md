# Assignment 2 Spec: AI Study Coach

Version: 2.5.0

Audience: course assessors and AI coding agents working on the project.

Brief: [assignment-2-brief.md](assignment-2-brief.md)
Project spec: [project-spec.md](project-spec.md)

---

## Purpose

This document is the reproduce-from-this-alone source of truth for the AI Study Coach.
A reader with this file, the brief, and the fixed tech stack should be able to reconstruct the project's deliverable shape — outcomes, scope, constraints, architecture, acceptance criteria, milestones, and prior decisions — without reading any other project file.
Any scope change, decision, or deliverable-affecting choice made during implementation must be reflected here before the corresponding pull request is merged.

---

## 1. Outcomes

A student can:

- Run a single Python script and be prompted for their name, course, study hours, quiz scores, and recent study sessions.
- See their study data summarised in the terminal with at least two visual charts.
- Receive a machine-learning-based prediction of whether they are "at risk" or "on track", along with the model's reported accuracy.
- Ask natural language questions to a "Coach Chat" and get answers grounded in a small knowledge base, with conversation memory preserved across turns.
- Run the script without an OpenAI API key and still see all non-LLM sections work without crashing.

---

## 2. In-scope

- Terminal-based input collection with validation
- Object-oriented domain model for students and study sessions
- Pandas-based data tracking with computed summary metrics
- Matplotlib visualisations: a bar chart of total study minutes by topic and a line chart of study minutes per session over time (two distinct chart types)
- SVM classification trained on a small CSV dataset
- LangChain conversational chain with memory
- LangChain RAG using FAISS over a local knowledge base of 8+ text chunks
- Graceful degradation when no API key is present

## Out-of-scope

- Web UI or GUI of any kind
- Persistent storage or database
- User authentication
- Multi-user support
- Deployment or hosting
- Unit tests as part of the submission (a local pytest scaffold exists under `tests/` for development convenience but is excluded from the submission zip)
- OAuth or third-party API integrations beyond OpenAI
- Real student data (all training data is synthetic)

---

## 3. Constraints

**Tech stack (fixed by course modules):**

- Python 3.x, single file
- Pandas and NumPy for data handling
- Matplotlib for visualisation
- scikit-learn (SVC) for classification
- LangChain for LLM chain, memory, and RAG
- FAISS for vector store
- OpenAI as LLM provider

**Patterns to follow:**

- The course provided tutorial files for each module: `python_basics/tutorial_code.py`, `oop/tutorial_code.py`, `ml_libraries/tutorial_code.py`, `svm/tutorial_code.py`, `langchain/tutorial_code.py`.
- The assessors expect to see patterns consistent with what was taught in these modules.

**Submission:**

- Single Python file: `capstone.py`
- One supporting data file: CSV for the SVM training data
- Zipped as: `PM_python_coding.zip`
- Excluded from the zip: `tests/`, `.venv/`, `requirements-dev.txt`, and any other local development scaffolding (AI workflow files, living docs, editor configs).
- One submission attempt, no resubmission.
- Deadline: 25 April 2026.

---

## 4. Architecture / Design Notes

The program runs sequentially through five sections in a single terminal session.
There are no menus and no loops back to the start.
Each section builds on data collected or created in the previous one.

**Pattern:**

This program follows two named patterns.

- **Batch Sequential / Pipeline** (architecture level): the stages input collection, object creation, DataFrame + charts, SVM prediction, and Coach Chat run one after another. Each stage completes before the next begins. Data flows one way through the pipeline.
- **Transaction Script** (code-structure level, per Martin Fowler): the program is organised as a top-to-bottom procedural script. Classes represent the domain model internally. The script itself is the orchestration; there is no object graph or service layer.

**Flow:**

```
Input collection -> Object creation -> DataFrame + charts -> SVM prediction -> Coach Chat (if API key present)
```

**Data model (conceptual):**

- A student has a name, course, goals, and a collection of study sessions.
- Each study session has a date, duration, topic, and difficulty rating.
- Quiz scores are collected as part of the student profile.
- The SVM training data is a separate CSV of synthetic records with features like weekly hours, quiz scores, difficulty, and session frequency, labelled as "at risk" or "on track".

**Knowledge base for RAG:**

A list of 8+ short text strings defined in-code.
Topics should cover study techniques, course logistics, exam rules, and similar.
These are not loaded from an external file; they are hardcoded as the "knowledge base" the coach draws from.

**Key integration point:**

The student's own data (collected in Section 1, structured in Section 2, analysed in Section 3) feeds into the SVM prediction in Section 4.
The model is trained on the CSV, then the current student's metrics are used as input for a single prediction.

---

## 5. Acceptance Criteria

**AC1 - Input handling:**
The script prompts for all required student data. Invalid input (wrong type, out of range, empty strings) is caught and re-prompted. The script never crashes on bad input.

**AC2 - Object model:**
Student and study session data is represented using classes. At least one attribute uses encapsulation (private attribute with a property or getter). An inheritance example is present.

**AC3 - Data and visualisation:**
Study sessions are stored in a Pandas DataFrame. Summary metrics (total hours, average duration, average quiz score) are printed. At least two different chart types are displayed using Matplotlib.

**AC4 - SVM prediction:**
An SVM classifier is trained on data loaded from a CSV file. A train/test split is used. Accuracy and a classification report are printed. The current student's data is fed through the model and a prediction is printed.

**AC5 - LangChain memory:**
A conversational chain is created with memory. At least one exchange is run through it. `memory.buffer` is printed to demonstrate memory is working.

**AC6 - LangChain RAG:**
A FAISS vector store is built from at least 8 text chunks. At least 2 questions are asked using the retrieval QA chain. Answers are printed.

**AC7 - Graceful degradation:**
If no OpenAI API key is set, the script prints a clear message for the LangChain section and completes all other sections without error.

**AC8 - Docstring:**
The file begins with a docstring explaining how to run the script, what it does, and how to set the API key.

**AC9 - Submission format:**
The zip is named `PM_python_coding.zip` and contains `capstone.py` plus the SVM training CSV.

---

## 6. Milestones

**Milestone convention (applies from v2.1.0 onward):**

Every milestone should document the following mandatory sections:

- **Intent:** what the milestone delivers, in user-visible terms.
- **Context:** background, prior decisions, and related systems relevant to the milestone.
- **Scope:** what is in scope, with an explicit **out-of-scope** list (treated as at least as important as the in-scope list).
- **Success criteria:** testable conditions that define "done".
- **Constraints:** technical or process limits that apply. Milestone-level Constraints are additive to §3; if no milestone-specific constraints exist, state that §3 applies and nothing else.
- **Open questions:** ambiguities to resolve; mark with `[NEEDS CLARIFICATION]` when stakeholder input is needed.

Milestones must not contain implementation detail such as tech stack choices, API shapes, code structure, data models, algorithms, or step-by-step procedures.
Those belong in the implementation plan produced at Step 3 of the AI workflow.

Optional sections may be added where useful, for example:

- **Verification:** how each success criterion will be checked (test, manual observation, metric), distinct from the behavioural success criteria themselves.
- **Links:** related issues, prior research, or upstream dependencies.

### M1 - Skeleton + input collection

- **Intent:** The student can launch the script and complete input collection for name, course, study hours, quiz scores, and recent study sessions. The script prints a summary confirming what it captured.
- **Context:** First milestone in the sequential flow. Input collection is the entry point for every downstream section; no data flows without it. Builds on the OOP patterns taught in the `oop/` tutorial.
- **Scope:**
  - In: prompts for all required fields, input validation with re-prompting on bad input, object construction for `Student` and `StudySession`, a printed summary.
  - Out: charts, metrics, ML, LangChain.
- **Success criteria:** AC1 (input handling, no crash on bad input) and AC2 (object model with encapsulation and inheritance) are satisfied.
- **Constraints:** §3 applies; no milestone-specific additions.
- **Open questions:** None identified.

### M2 - Data and charts

- **Intent:** The student sees their study data rendered as summary metrics and at least two distinct chart types.
- **Context:** Depends on M1 for the `Student` and `StudySession` objects. Demonstrates Pandas and Matplotlib competency, building on the `ml_libraries/` tutorial.
- **Scope:**
  - In: Pandas DataFrame built from session objects, computed totals and averages, at least two chart types displayed via Matplotlib.
  - Out: SVM, LangChain, persistence of generated plots.
- **Success criteria:** AC3 (DataFrame, summary metrics, two chart types) is satisfied.
- **Constraints:** §3 applies. Charts render in the terminal session; a pop-up window is acceptable.
- **Open questions:** None identified.

### M3 - SVM prediction

- **Intent:** The student receives a risk classification ("at risk" / "on track") based on their collected data, alongside the model's accuracy on a held-out test split.
- **Context:** Depends on M1/M2 for the student data. Demonstrates scikit-learn competency, building on the `svm/` tutorial. The training CSV is a separate file in the submission zip.
- **Scope:**
  - In: synthetic labelled CSV, SVC training, train/test split, accuracy and classification report printed, single prediction for the current student's metrics.
  - Out: LangChain, hyperparameter tuning beyond what the tutorial demonstrates, cross-validation.
- **Success criteria:** AC4 (train/test split, accuracy, report, per-student prediction) is satisfied.
- **Constraints:** §3 applies. The CSV's feature set must align with metrics derivable from the student's collected data, so the current student's vector can be fed to the model directly.
- **Open questions:** None identified.

### M4 - LangChain integration

- **Intent:** The student can ask questions via Coach Chat and receive memory-aware answers grounded in the in-code knowledge base. The script completes cleanly when no API key is present.
- **Context:** Self-contained relative to M1-M3 for data flow. Demonstrates LangChain memory, RAG, and graceful-degradation competency, building on the `langchain/` tutorial. The only section that touches the OpenAI API.
- **Scope:**
  - In: LangChain conversational chain with memory (`memory.buffer` printed), FAISS vector store over 8+ in-code chunks, at least two RAG questions answered, graceful fallback when `OPENAI_API_KEY` is unset.
  - Out: streaming responses, tool use, agents, multi-turn RAG memory coupling.
- **Success criteria:** AC5 (memory chain), AC6 (RAG), and AC7 (graceful degradation) are satisfied.
- **Constraints:** §3 applies; no milestone-specific additions.
- **Open questions:** None identified.

### M5 - Polish and package

- **Intent:** A submission-ready zip exists, containing the final script and the SVM CSV, with a top-of-file docstring explaining how to run it.
- **Context:** Final milestone. All earlier sections must be working end-to-end before packaging. One submission attempt means this must be correct first time.
- **Scope:**
  - In: top-of-file docstring (what the script does, how to run it, how to set the API key), end-to-end run confirmation, zip named `PM_python_coding.zip` containing `capstone.py` and the training CSV.
  - Out: README file, separate setup script, `requirements.txt` (not required by the brief); local development scaffolding — `tests/`, `.venv/`, `requirements-dev.txt`, AI workflow files, and living docs — must not be included in the submission zip.
- **Success criteria:** AC8 (docstring) and AC9 (zip naming and contents) are satisfied.
- **Constraints:** §3 applies; no milestone-specific additions.
- **Open questions:** None identified.

---

## 7. Prior Decisions and Rationale

| Decision | Rationale |
|---|---|
| Use Case 3 (AI Study Coach) chosen over Use Cases 1 and 2 | Most comprehensive option. Covers all five course modules in one script. |
| Generic student theme, not personalised to AI engineering | Keeps it simple and aligned with the brief template. No need to overthink the domain. |
| CSV for SVM data instead of in-code synthetic generation | Demonstrates file I/O. Slightly more realistic. Adds one file to the zip but that is acceptable. |
| OpenAI as LLM provider | Explicitly named in the brief. Matches the course tutorial code. |
| Knowledge base hardcoded in-code, not loaded from file | The brief asks for "at least 8 short documents/chunks" defined for RAG. Hardcoding avoids an extra file and keeps the knowledge base visible in the script. |
| Sequential flow, no menu system | The brief describes a linear program, not an interactive app. A menu would be scope creep. |
| Named architecture patterns (Batch Sequential / Pipeline at architecture level; Transaction Script at code-structure level) | Gives readers a handle they can look up; makes the intent explicit rather than implied by description alone. Added in v2.1.0. |
| Milestone convention (Intent / Context / Scope / Success / Constraints / Open questions) | Aligns with current spec-driven development practice (GitHub Spec Kit, Addy Osmani, Augment Code, Fowler). Keeps implementation detail out of milestones. Added in v2.1.0. |
| Git-tag-based versioning for living docs, unversioned filenames | Git history and tags already provide version tracking. Physical snapshot files are redundant and risk drift. Added in v2.1.0. |
| M2 chart pair: bar (total study minutes by topic) + line (study minutes per session over time) | Two tutorial-familiar chart types that satisfy AC3 while answering distinct questions: aggregation across a categorical dimension, and a trend across time. Added in v2.2.0. |
| Local pytest scaffold under `tests/` with pinned dev deps in `requirements-dev.txt`, excluded from the submission zip | Gives the agent a repeatable validation signal during development without expanding the submission surface beyond what the brief requires. Added in v2.2.0. |
| M3 SVM training CSV: `study_risk.csv` with columns `total_hours, avg_quiz_score, avg_difficulty, session_count, label` (labels `"at risk"` / `"on track"`) | Feature columns map 1:1 to quantities derivable from the student's collected data, so the runtime feature vector can be fed to the model without transformation. Filename kept short and label-descriptive. Added in v2.3.0. |
| M3 insufficient-data policy: if the current student has no quiz scores or no study sessions, still train and print accuracy + classification report, but print "not enough data to predict" instead of a per-student prediction | Keeps AC4's training-side evidence visible to the assessor even when the student skipped optional inputs, while being honest about the prediction's unreliability. Added in v2.3.0. |
| M4 LangChain stack: `ConversationBufferMemory` + `ConversationChain` for AC5, `FAISS.from_texts` + `RetrievalQA.from_chain_type` for AC6, `ChatOpenAI(model="gpt-4o-mini")` + `OpenAIEmbeddings()` as the LLM / embedding providers | Matches the course `langchain/` tutorial patterns and satisfies AC5's requirement to print `memory.buffer`. `gpt-4o-mini` keeps per-run cost negligible while producing coherent short-form answers. Added in v2.4.0. |
| M4 cross-version import fallback: each LangChain import is wrapped in a try/except that also tries the `langchain_classic` / `langchain_community` / `langchain_openai` paths | LangChain v1 relocated `ConversationBufferMemory`, `ConversationChain`, and `RetrievalQA` into `langchain-classic`, and split out `langchain-community` / `langchain-openai` earlier. The try/except lets a single submitted `capstone.py` run on either the older layout the course tutorial used or a current fresh install. Added in v2.4.0. |
| M4 knowledge base: 10 hardcoded in-code chunks covering active recall, spaced repetition, Pomodoro, sleep, Feynman, interleaving, environment, exam technique, exercise, and goal-setting | Exceeds AC6's 8-chunk minimum with a small buffer and spans topical areas the brief calls out (study techniques, course logistics, exam rules). Keeps the knowledge base visible inside the script rather than in a separate file. Added in v2.4.0. |
| M4 memory-chain exchange: two predictions per run (an opening tip request referencing the student's name and course, then "What was my name?") | A single exchange satisfies AC5's literal requirement, but the follow-up question gives the assessor a visible signal that `memory.buffer` actually carries prior turns. Added in v2.4.0. |
| M4 graceful-degradation message: "OPENAI_API_KEY not set; skipping Coach Chat." followed by a hint to rerun with the variable set | AC7 requires a clear message; surfacing the rerun hint makes the skip feel like a controlled branch rather than an error, and keeps the assessor's exit-code-0 expectation intact. Added in v2.4.0. |
| M5 submission zip built on demand with `zip -j PM_python_coding.zip capstone.py study_risk.csv` and gitignored rather than committed | The zip is a derived artefact of two tracked files; keeping it out of git prevents drift between the submitted zip and the source files. `-j` flattens directory prefixes so both files land at the zip root as AC9 requires. Added in v2.5.0. |
