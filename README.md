# AI Study Coach

Single-file Python terminal app that collects a student's study data, summarises and visualises it, predicts study risk with an SVM classifier, and answers questions via a LangChain coach chat with memory and RAG.

> **Status:** in progress. `capstone.py` is in the repo with the input-collection section and the data / charts section implemented; the SVM training CSV and the SVM / LangChain sections are not yet complete. This README describes the intended runtime behaviour as defined in [`project-spec.md`](project-spec.md) and will be updated to match implementation truth as code lands.

## What it does

1. **OOP domain model** — `Student` and `StudySession` classes with encapsulation and inheritance; terminal input with validation and re-prompting.
2. **Pandas / NumPy** — study sessions structured as a DataFrame with computed summary metrics.
3. **Matplotlib** — at least two chart types visualising the student's study data.
4. **scikit-learn SVM** — SVC classifier trained on a synthetic CSV with train/test split, accuracy, and classification report; predicts whether the student is "at risk" or "on track".
5. **LangChain** — conversational chain with memory (prints `memory.buffer`) plus a FAISS-backed RAG chain over 8+ in-code text chunks.

## Requirements

- Python 3.x
- `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `langchain`, `faiss`, `openai`
- OpenAI API key in the `OPENAI_API_KEY` environment variable (optional — see below)

## How to run

Set up a local virtual environment and install the dev dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Then run the script:

```bash
python capstone.py
```

The script runs as a sequential terminal session. It prompts for student data, then walks through each section in order.

### Running without an API key

All non-LLM sections (input, OOP model, Pandas summary, Matplotlib charts, SVM prediction) complete successfully without an OpenAI API key. If `OPENAI_API_KEY` is not set, the LangChain coach chat and RAG sections degrade gracefully and the script continues to finish without crashing.

## Session flow

1. Collect student profile and study sessions from the terminal.
2. Build domain objects (`Student`, `StudySession`).
3. Convert sessions to a Pandas DataFrame and print summary metrics.
4. Render Matplotlib charts of the student's study data.
5. Run the SVM classifier and print accuracy + classification report for a risk prediction.
6. Launch the LangChain coach chat (memory) for at least one exchange, then print `memory.buffer`.
7. Run the FAISS RAG chain against the in-code knowledge base for at least two questions.

## File layout

- `capstone.py` — single-file script containing all five sections. *(Sections 1 and 2 implemented; sections 3-5 are placeholder stubs.)*
- SVM training CSV — synthetic labelled dataset read by the SVM section. *(Not yet created; filename TBD.)*
- [`project-spec.md`](project-spec.md) — living implementation-truth spec for this project.
