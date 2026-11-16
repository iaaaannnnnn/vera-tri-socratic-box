# VERA-TRI: Socratic Box & Authenticity Verification Engine

A Python/Streamlit implementation of the **Socratic Box** and **Multimodal Authenticity Verification Engine** based on the VERA-TRI research framework.

This system mitigates student over-reliance on Generative AI by providing a constrained, material-grounded tutoring environment alongside semantic cosine similarity verification for student submissions.

---

## 📌 Features

### 1. Module 1: Socratic Sandbox (Student Mode)
* **Material-Grounded Evaluator:** Constrains AI evaluation strictly to user-provided reference texts, preventing ungrounded general answers.
* **Minimal Feedback Protocol:** Evaluates answers with a clear verdict (`Correct` / `Incorrect`) and a minimal hint, deliberately withholding full solutions to encourage germane cognitive effort.
* **Cynical & Objective Tone:** Maintains a concise, neutral, and dispassionate voice focused on conceptual accuracy.

### 2. Module 2: Multimodal Authenticity Verification
* **TF-IDF & Cosine Similarity Engine:** Converts written documents and student summaries into numerical vectors to calculate semantic similarity.
* **Threshold-Based Validation:** Enforces a minimum coherence threshold (Score $\ge 0.40$) across up to 3 attempts.
* **Speech-to-Text Integration:** Supports live oral summaries via microphone recording alongside text inputs.

### 3. Teacher Mode (Protected)
* **Password-Protected Portal:** Restricts access to assignment auditing tools (Default password: `veratri2026`).
* **Anti-AI Assignment Vulnerability Audit:** Evaluates draft assignment prompts for AI exploitation risks and suggests structural redesigns (e.g., local context, observational data, reflective commentary).

---

## 🛠️ Installation & Requirements

### System Prerequisites (Audio Support)
If you plan to use live microphone voice summaries on macOS or Linux, install the system-level audio dependencies prior to Python packages:

* **macOS:**
  ```bash
  brew install portaudio