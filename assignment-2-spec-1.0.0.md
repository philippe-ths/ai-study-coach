# Assignment 2 Spec: AI Study Coach (capstone.py)

## What this is

A single Python file (`capstone.py`) that acts as a small AI Study Coach.
It collects student info, tracks study sessions, predicts risk of falling behind, and answers questions using a knowledge base.

Runs with: `python capstone.py`

## Submission

- File: `PM_python_coding.zip`
- Contains: `capstone.py` + `study_risk_data.csv`
- One submission attempt only
- Due: 25 April 2026

---

## Section 1: Student Profile + Inputs (Python Basics)

**What it does:** Collects student info and recent study data via terminal input.

**Requirements:**

- Collect: name, course, weekly study hours, recent quiz scores
- Validate all inputs (no crashes on bad values)
- Use functions to organize logic (no giant top-level script)

**Inputs to collect:**

- Student name (string, non-empty)
- Course name (string, non-empty)
- Target weekly study hours (positive number)
- Recent quiz scores (list of numbers, 0-100)
- 3-5 study sessions: date, duration in hours, topic, self-rated difficulty (1-5)

---

## Section 2: OOP Class Model

**What it does:** Structures the domain using classes.

**Required classes:**

- `Student` - name, course, goals, study history
  - Method to print a summary
- `StudySession` - date, duration, topic, self-rated difficulty
  - Stored as a list inside Student

**Should also include:**

- At least one example of encapsulation (private attribute with getter/property)
- Optional: one inheritance example (e.g. `STEMStudent(Student)` or `OnlineStudent(Student)`)

---

## Section 3: Data Tracking + Charts (Pandas / Matplotlib)

**What it does:** Stores session data in a DataFrame and visualises it.

**Requirements:**

- Store study sessions in a Pandas DataFrame
- Compute and print summary metrics:
  - Total hours studied
  - Average session duration
  - Average quiz score
  - Study streak length (consecutive days)
- Display at least 2 plots:
  - e.g. trend line of hours over time
  - e.g. scatter of difficulty vs duration
  - e.g. histogram of quiz scores
- Use `plt.show()` to display

---

## Section 4: SVM Risk Prediction (scikit-learn)

**What it does:** Predicts whether a student is "at risk" or "on track" using an SVM classifier.

**Requirements:**

- Use SVC from scikit-learn
- Use train/test split
- Report accuracy + classification_report
- Use the trained model to predict the current student's status

**Data approach:** Small CSV file (`study_risk_data.csv`)

- 50-100 rows
- Features: weekly_hours, avg_quiz_score, difficulty_rating, sessions_per_week
- Label: "at_risk" or "on_track"
- Ship this CSV alongside capstone.py in the zip

---

## Section 5: LangChain Q&A with Memory + RAG

**What it does:** A "Coach Chat" that answers study questions using a small knowledge base.

**Requirements:**

- Load OPENAI_API_KEY from environment or .env
- If no key is set, print a clear message and skip this section (no crash)

**Memory:**

- Save at least one input/output pair
- Print `memory.buffer` to demonstrate memory is working

**RAG (Retrieval-Augmented Generation):**

- Build a knowledge base of at least 8 short text chunks covering:
  - Study techniques (e.g. spaced repetition, active recall)
  - Grading policy
  - Office hours info
  - Exam rules
  - Course resources
  - Time management tips
  - Group study guidelines
  - Mental health / wellbeing resources
- Build a FAISS vector store from these chunks
- Ask at least 2 questions using `qa.run(...)`
- Print the answers

**LLM provider:** OpenAI (as specified in the brief)

---

## Docstring (top of file)

Must include:

- How to run the script
- What the program does (2-3 sentences)
- How to set the OpenAI key for the LangChain section

---

## Checklist (for validation before submission)

- [ ] Single file: `capstone.py`
- [ ] Runs with `python capstone.py`
- [ ] Docstring at top of file
- [ ] Section 1: Input collection with validation, organised in functions
- [ ] Section 2: Student and StudySession classes, encapsulation example
- [ ] Section 3: Pandas DataFrame, summary metrics printed, 2+ plots shown
- [ ] Section 4: SVM with train/test split, accuracy + classification_report, predicts current student
- [ ] Section 5: Graceful handling when no API key
- [ ] Section 5: Memory demo with memory.buffer printed
- [ ] Section 5: FAISS knowledge base with 8+ chunks
- [ ] Section 5: At least 2 qa.run() questions answered
- [ ] CSV file: `study_risk_data.csv` included in zip
- [ ] Zip named: `PM_python_coding.zip`
