# Assignment 2 Spec: AI Study Coach

Version: 2.0.0
Brief: [assignment-2-brief.md](assignment-2-brief-1.0.0.md)

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
- Matplotlib visualisations (at least 2 distinct chart types)
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
- Unit tests (not required by brief, though useful)
- OAuth or third-party API integrations beyond OpenAI
- Real student data - all training data is synthetic

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

- The course provided tutorial files for each module: `python_basics/tutorial_code.py`, `oop/tutorial_code.py`, `ml_libraries/tutorial_code.py`, `svm/tutorial_code.py`, `langchain/tutorial_code.py`. The assessors expect to see patterns consistent with what was taught in these modules.

**Submission:**

- Single Python file: `capstone.py`
- One supporting data file: CSV for the SVM training data
- Zipped as: `PM_python_coding.zip`
- One submission attempt. No resubmission.

---

## 4. Architecture / Design Notes

The program runs sequentially through five sections in a single terminal session. No menus, no loops back to start. Each section builds on data collected or created in the previous one.

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

A list of 8+ short text strings defined in-code. Topics should cover study techniques, course logistics, exam rules, and similar. These are not loaded from an external file - they are hardcoded as the "knowledge base" the coach draws from.

**Key integration point:**

The student's own data (collected in Section 1, structured in Section 2, analysed in Section 3) feeds into the SVM prediction in Section 4. The model is trained on the CSV, then the current student's metrics are used as input for a single prediction.

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

**M1 - Skeleton + input collection**
Script runs, collects all inputs with validation, creates objects, prints a summary.

**M2 - Data and charts**
DataFrame built from session objects. Metrics computed and printed. Two charts display correctly.

**M3 - SVM prediction**
CSV created with synthetic data. Model trains, evaluates, and predicts for the current student.

**M4 - LangChain integration**
Memory chain works. FAISS knowledge base built. RAG answers questions. Graceful fallback without API key.

**M5 - Polish and package**
Docstring written. End-to-end run confirmed. Zip created with correct naming.

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
