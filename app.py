import os
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------
# Optional File Parsing & OpenAI Dependencies
# ---------------------------------------------------------
try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    import docx
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ---------------------------------------------------------
# Helper Function: Extract Text from Uploaded Files
# ---------------------------------------------------------
def read_uploaded_file(uploaded_file) -> str:
    """Extracts text content from .txt, .md, .pdf, .docx, .xlsx, and .csv files."""
    if uploaded_file is None:
        return ""

    file_name = uploaded_file.name.lower()
    
    # 1. Plain Text & Markdown
    if file_name.endswith((".txt", ".md")):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    
    # 2. PDF Documents
    elif file_name.endswith(".pdf"):
        if not HAS_PYPDF:
            st.warning("⚠️ `pypdf` missing. Install via `pip install pypdf` to parse PDFs.")
            return ""
        try:
            pdf_reader = pypdf.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text.strip()
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""

    # 3. Word Documents (.docx)
    elif file_name.endswith(".docx"):
        if not HAS_DOCX:
            st.warning("⚠️ `python-docx` missing. Install via `pip install python-docx` to parse Word docs.")
            return ""
        try:
            doc = docx.Document(uploaded_file)
            full_text = [para.text for para in doc.paragraphs if para.text.strip()]
            return "\n".join(full_text)
        except Exception as e:
            st.error(f"Error reading Word document: {e}")
            return ""

    # 4. Excel & CSV Spreadsheets (.xlsx, .csv)
    elif file_name.endswith((".xlsx", ".csv")):
        if not HAS_PANDAS:
            st.warning("⚠️ `pandas` missing. Install via `pip install pandas openpyxl` to parse spreadsheets.")
            return ""
        try:
            if file_name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            return df.to_string(index=False)
        except Exception as e:
            st.error(f"Error reading spreadsheet: {e}")
            return ""

    else:
        st.error("Unsupported file format.")
        return ""


# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Socratic Box | VERA-TRI",
    page_icon="📦",
    layout="wide"
)

st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E293B; }
    .sub-title { font-size: 1rem; color: #64748B; margin-bottom: 2rem; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "teacher_authenticated" not in st.session_state:
    st.session_state.teacher_authenticated = False
if "auth_attempts" not in st.session_state:
    st.session_state.auth_attempts = 0


# ---------------------------------------------------------
# Core Engines (Socratic Sandbox & Teacher Assistant)
# ---------------------------------------------------------
class SocraticSandboxEngine:
    """Engine executing the Socratic Box principles for Students & Teachers."""
    def __init__(self, api_key: str = None, language: str = "English"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if (HAS_OPENAI and self.api_key) else None
        self.language = language

        # Language instructions
        lang_map = {
            "English": "Respond strictly in English.",
            "Indonesian": "Jawablah seluruhnya dalam Bahasa Indonesia yang lugas dan baku.",
            "Javanese": "Wangsulana kabeh ing Basa Jawa ingkang trep (Krama/Ngoko lugu)."
        }

        # Student System Prompt: Cold, Cynical, Material-Grounded, Minimal Feedback
        self.student_system_prompt = (
            f"You are the Socratic Box AI Evaluator. You operate under 5 strict core principles:\n"
            f"1. Material-grounded: Judge answers strictly using provided learning resources. Trace every verdict to this material.\n"
            f"2. Minimal feedback: Give ONLY a clear verdict ('Correct' or 'Incorrect') and a tiny hint. NEVER supply the answer or complete solution.\n"
            f"3. Honest uncertainty: Explicitly admit when the provided material is insufficient rather than guessing or using outside knowledge.\n"
            f"4. Independent learning: Help students discover and correct their own mistakes instead of correcting them on their behalf.\n"
            f"5. Cynical personality: Brief, neutral, cold, and unencouraging. Focus on correctness over emotional praise.\n"
            f"LANGUAGE REQUIREMENT: {lang_map.get(language, 'Respond in English.')}"
        )

        # Teacher System Prompt: Collaborative Thinking Partner
        self.teacher_system_prompt = (
            f"You are the Socratic Box Teacher Assistant — a collaborative thinking partner for educators.\n"
            f"Help teachers refine assessment strategies, analyze AI vulnerability in questions, "
            f"suggest higher-order tasks, and design effective lesson plans.\n"
            f"LANGUAGE REQUIREMENT: {lang_map.get(language, 'Respond in English.')}"
        )

    def evaluate_student(self, material: str, submission: str) -> str:
        """Processes Student Mode single combined prompt."""
        if not self.client:
            mock_responses = {
                "English": "**VERDICT: Incorrect**\n\n*Hint:* Re-read paragraph 2 of your material. Focus on the primary variable definition.",
                "Indonesian": "**VONIS: Salah**\n\n*Petunjuk:* Baca kembali paragraf 2 dari materi Anda. Perhatikan definisi variabel utama.",
                "Javanese": "**VONIS: Salah**\n\n*Pituduh:* Waca maneh ing paragraf 2. Gatekna babagan variabel utama."
            }
            return mock_responses.get(self.language, mock_responses["English"])

        combined_prompt = (
            f"--- MANDATORY LEARNING MATERIAL ---\n{material}\n\n"
            f"--- STUDENT QUESTION / ANSWER SUBMISSION ---\n{submission}\n\n"
            f"Provide a brief, cold verdict ('Correct' or 'Incorrect') and a small hint. Do NOT give the full answer."
        )

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.student_system_prompt},
                {"role": "user", "content": combined_prompt}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content

    def assist_teacher(self, mode: str, content: str) -> str:
        """Executes the 3 specific Teacher Mode operational paths."""
        if not self.client:
            return (
                f"**[Mock Teacher Mode Output - {mode}]**\n\n"
                f"Analysis completed. The prompt requires local context integration and multi-step observational reasoning "
                f"to prevent direct AI copy-pasting."
            )

        prompts = {
            "Lesson Planning Suggestions": (
                f"Act as a collaborative thinking partner. Based on this curriculum topic/draft:\n'{content}'\n"
                f"Provide actionable lesson planning suggestions, instructional sequencing, and discussion anchors."
            ),
            "Analysis of Assignments (AI Vulnerability Audit)": (
                f"Analyze this assignment draft:\n'{content}'\n"
                f"Evaluate if this question is too easily solved by Generative AI. Identify weaknesses and recommend "
                f"specific redesigns (e.g., adding local scope, primary data, multi-step reasoning, or personal reflection)."
            ),
            "Ideas for Creative Tasks (Higher-Order Thinking)": (
                f"Based on this subject/material:\n'{content}'\n"
                f"Generate 3 creative assignment ideas that stimulate higher-order cognitive skills (analysis, synthesis, evaluation) "
                f"and resist rote AI completion."
            )
        }

        user_prompt = prompts.get(mode, f"Analyze: {content}")

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.teacher_system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content


class AuthenticityVerifier:
    """Module 2: TF-IDF Cosine Similarity Engine (Text & File driven)."""
    def __init__(self, threshold: float = 0.40):
        self.threshold = threshold

    @staticmethod
    def compute_similarity(text_a: str, text_b: str, language: str = "English") -> float:
        if not text_a.strip() or not text_b.strip():
            return 0.0
        stop_setting = 'english' if language == "English" else None
        vectorizer = TfidfVectorizer(stop_words=stop_setting)
        tfidf = vectorizer.fit_transform([text_a, text_b])
        return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])


# ---------------------------------------------------------
# Sidebar Role Selection & Configuration
# ---------------------------------------------------------
st.sidebar.title("📦 Socratic Box System")
st.sidebar.caption("Material-Grounded AI Evaluation Platform")

# Language Selection
selected_lang = st.sidebar.selectbox(
    "🌐 Interface & Evaluation Language",
    ["English", "Indonesian", "Javanese"],
    help="Determines the operating language for AI verdicts and analysis."
)

# Operational Role Switch
role = st.sidebar.radio(
    "Select Operational Role",
    ["Student Mode", "Teacher Mode (Password-Protected)"],
    help="Selecting a role determines system instructions, access rights, and operational paths."
)

api_key_input = st.sidebar.text_input("OpenAI API Key (Optional)", type="password")
engine = SocraticSandboxEngine(api_key=api_key_input, language=selected_lang)
verifier = AuthenticityVerifier(threshold=0.40)


# ---------------------------------------------------------
# STUDENT MODE
# ---------------------------------------------------------
if role == "Student Mode":
    st.markdown('<div class="main-title">Student Mode — Socratic Sandbox</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Material-grounded evaluation providing minimal feedback to encourage independent thinking.</div>', 
        unsafe_allow_html=True
    )

    tab1, tab2 = st.tabs(["📝 Socratic Sandbox Evaluation", "📄 Written Authenticity Verification"])

    # ----------------------------------
    # TAB 1: SOCRATIC SANDBOX EVALUATION
    # ----------------------------------
    with tab1:
        st.subheader("1. Learning Material (Mandatory Contextual Boundary)")
        st.caption("Upload or paste the learning resources. Evaluation is strictly confined to this scope.")
        
        mat_file = st.file_uploader(
            "Upload Reference Material (.txt, .md, .pdf, .docx, .xlsx, .csv)", 
            type=["txt", "md", "pdf", "docx", "xlsx", "csv"], 
            key="student_mat_file"
        )
        mat_text = read_uploaded_file(mat_file) if mat_file else ""

        learning_material = st.text_area(
            "Learning Resource Content:",
            value=mat_text,
            height=160,
            placeholder="Mandatory field: Upload a file above or paste your study material here to set the AI's boundary..."
        )

        st.subheader("2. Student Submission")
        st.caption("Submit your question or proposed answer for cold, diagnostic feedback.")
        
        sub_file = st.file_uploader(
            "Upload Answer/Submission File (.txt, .md, .pdf, .docx, .xlsx, .csv)", 
            type=["txt", "md", "pdf", "docx", "xlsx", "csv"], 
            key="student_sub_file"
        )
        sub_text = read_uploaded_file(sub_file) if sub_file else ""

        student_submission = st.text_area(
            "Student Question or Attempt:",
            value=sub_text,
            height=120,
            placeholder="Type your response or attempt here..."
        )

        if st.button("Submit for Socratic Verdict", type="primary"):
            if not learning_material.strip():
                st.error("Error: Learning material is mandatory to establish contextual boundaries.")
            elif not student_submission.strip():
                st.warning("Please enter your question or submission before evaluating.")
            else:
                with st.spinner("Evaluating submission against material bounds..."):
                    result = engine.evaluate_student(learning_material, student_submission)
                    st.markdown("### Socratic Box Response")
                    st.info(result)

    # ----------------------------------
    # TAB 2: AUTHENTICITY VERIFICATION
    # ----------------------------------
    with tab2:
        st.subheader("Text Authenticity Validation Engine")
        st.write("Compare full written work against a short summary to calculate TF-IDF Cosine Similarity.")

        st.markdown("#### Step 1: Original Written Work")
        orig_file = st.file_uploader("Upload Assignment File", type=["txt", "md", "pdf", "docx", "xlsx", "csv"], key="orig_file")
        orig_text = read_uploaded_file(orig_file) if orig_file else ""
        
        original_doc = st.text_area(
            "Original Assignment Document:",
            value=orig_text,
            height=140,
            placeholder="Paste complete written assignment..."
        )

        st.markdown("#### Step 2: Written Summary")
        sum_file = st.file_uploader("Upload Summary File", type=["txt", "md", "pdf", "docx", "xlsx", "csv"], key="sum_file")
        sum_text = read_uploaded_file(sum_file) if sum_file else ""

        summary_doc = st.text_area(
            "Written Explanation / Summary:",
            value=sum_text,
            height=100,
            placeholder="Type a concise summary of your work..."
        )

        if st.button("Validate Coherence Score"):
            if not original_doc.strip() or not summary_doc.strip():
                st.error("Both original assignment text and summary are required.")
            else:
                score = verifier.compute_similarity(original_doc, summary_doc, language=selected_lang)
                st.markdown(f"#### Cosine Similarity Score ({selected_lang}): **{score:.4f}**")
                
                if score >= 0.40:
                    st.success("✅ **VERIFICATION PASSED**: Coherence threshold met (Score ≥ 0.40). Assignment authenticated.")
                else:
                    st.session_state.auth_attempts += 1
                    remaining = 3 - st.session_state.auth_attempts
                    st.error(f"❌ **VERIFICATION FAILED**: Low semantic alignment (Score < 0.40). Remaining attempts: {remaining}")
                    
                    if st.session_state.auth_attempts >= 3:
                        st.error("⚠️ **FORMAL WARNING**: Potential AI-assisted plagiarism flagged. Assignment voided.")


# ---------------------------------------------------------
# TEACHER MODE (PASSWORD-PROTECTED)
# ---------------------------------------------------------
else:
    st.markdown('<div class="main-title">Teacher Mode — Collaborative Thinking Partner</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Confidential portal to analyze assignment vulnerability, plan lessons, and design creative tasks.</div>', 
        unsafe_allow_html=True
    )

    # Password Safeguard
    if not st.session_state.teacher_authenticated:
        st.subheader("🔒 Educator Authentication Required")
        pwd = st.text_input("Enter Confidential Teacher Password:", type="password")
        if st.button("Unlock Teacher Portal"):
            if pwd == "veratri2026":
                st.session_state.teacher_authenticated = True
                st.rerun()
            else:
                st.error("Invalid password.")
    else:
        st.success("🔒 Authenticated as Educator")
        if st.button("Logout"):
            st.session_state.teacher_authenticated = False
            st.rerun()

        st.divider()

        # The 3 Specific Options from the Text Specification
        teacher_option = st.selectbox(
            "Select Operational Feature",
            [
                "Lesson Planning Suggestions",
                "Analysis of Assignments (AI Vulnerability Audit)",
                "Ideas for Creative Tasks (Higher-Order Thinking)"
            ]
        )

        st.markdown(f"### Feature: {teacher_option}")
        
        t_file = st.file_uploader(
            "Upload Reference Material / Question Draft (.txt, .md, .pdf, .docx, .xlsx, .csv)", 
            type=["txt", "md", "pdf", "docx", "xlsx", "csv"], 
            key="teacher_file"
        )
        t_file_text = read_uploaded_file(t_file) if t_file else ""

        teacher_input = st.text_area(
            "Paste Draft Question, Topic, or Assignment Content:",
            value=t_file_text,
            height=140,
            placeholder="Enter your assignment draft or lesson topic here for collaborative AI analysis..."
        )

        if st.button("Run Teacher Analysis", type="primary"):
            if not teacher_input.strip():
                st.warning("Please provide input text or upload a file for analysis.")
            else:
                with st.spinner("Generating teacher collaboration output..."):
                    teacher_result = engine.assist_teacher(teacher_option, teacher_input)
                    st.markdown("### Educator Advisory Output")
                    st.info(teacher_result)
