import os
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# PDF extraction support
try:
    import pypdf
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

# OpenAI support
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# ---------------------------------------------------------
# Helper Function: Extract Text from Uploaded Files
# ---------------------------------------------------------
def read_uploaded_file(uploaded_file) -> str:
    """Extracts text content from uploaded .txt, .md, or .pdf files."""
    if uploaded_file is None:
        return ""

    file_name = uploaded_file.name.lower()
    
    if file_name.endswith((".txt", ".md")):
        return uploaded_file.read().decode("utf-8", errors="ignore")
    
    elif file_name.endswith(".pdf"):
        if not HAS_PYPDF:
            st.warning("⚠️ `pypdf` library missing. Install via `pip install pypdf` to read PDF files.")
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
    else:
        st.error("Unsupported file format. Please upload a .txt, .md, or .pdf file.")
        return ""


# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="VERA-TRI | Socratic Box",
    page_icon="🛡️",
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
# Core Logic Engines
# ---------------------------------------------------------
class SocraticSandbox:
    """Module 1: Material-grounded, minimal-feedback cynical tutor."""
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if (HAS_OPENAI and self.api_key) else None

        self.system_prompt = (
            "You are a Socratic Sandbox AI Tutor. You possess a brief, neutral, cold, "
            "and deliberately cynical personality. Your primary goal is to evaluate student answers "
            "STRICTLY using the provided learning materials.\n"
            "OPERATIONAL RULES:\n"
            "1. Material-grounded: Judge answers ONLY using the uploaded resource. If material is insufficient, state honest uncertainty.\n"
            "2. Minimal feedback: Return ONLY a clear verdict ('Correct' or 'Incorrect') and a tiny hint or conceptual gap.\n"
            "3. NEITHER reveal complete solutions NOR rewrite answers for the student under any circumstances.\n"
            "4. Keep output brief, direct, and unencouraging."
        )

    def evaluate(self, material: str, submission: str) -> str:
        if not material.strip():
            return "Error: Learning material is mandatory to establish contextual boundaries."

        if not self.client:
            # Fallback mock response if OpenAI key is not provided
            return (
                "**VERDICT: Incorrect**\n\n"
                "*Hint:* Review the primary mechanisms outlined in the material. "
                "You substituted core concepts with external assumptions."
            )

        prompt = f"--- LEARNING MATERIAL ---\n{material}\n\n--- STUDENT SUBMISSION ---\n{submission}"
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content


class AuthenticityVerifier:
    """Module 2: Cosine Similarity authenticity verification engine (Text/File driven)."""
    def __init__(self, threshold: float = 0.40):
        self.threshold = threshold

    @staticmethod
    def compute_similarity(text_a: str, text_b: str) -> float:
        if not text_a.strip() or not text_b.strip():
            return 0.0
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
        return float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])


# ---------------------------------------------------------
# Sidebar & Navigation
# ---------------------------------------------------------
st.sidebar.title("🛡️ VERA-TRI System")
st.sidebar.caption("Triple-Engine Curriculum Shielding Platform")

role = st.sidebar.radio("Select Operational Role", ["Student Mode", "Teacher Mode (Protected)"])

api_key_input = st.sidebar.text_input("OpenAI API Key (Optional)", type="password")
socratic_engine = SocraticSandbox(api_key=api_key_input)
verifier = AuthenticityVerifier(threshold=0.40)


# ---------------------------------------------------------
# STUDENT MODE
# ---------------------------------------------------------
if role == "Student Mode":
    st.markdown('<div class="main-title">Socratic Sandbox</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Constrained AI tutoring designed for minimal feedback, '
        'material grounding, and metacognitive reflection.</div>', 
        unsafe_allow_html=True
    )

    tab1, tab2 = st.tabs(["📝 Module 1: Guided Socratic Sandbox", "📄 Module 2: Authenticity Verification"])

    # ----------------------------------
    # TAB 1: SOCRATIC SANDBOX
    # ----------------------------------
    with tab1:
        st.subheader("1. Grounding Material (Mandatory)")
        mat_file = st.file_uploader("Upload Learning Material (.txt, .md, .pdf)", type=["txt", "md", "pdf"], key="mat_file")
        
        mat_file_text = read_uploaded_file(mat_file) if mat_file else ""
        
        learning_material = st.text_area(
            "Or paste reference text directly:",
            value=mat_file_text,
            height=160,
            placeholder="Upload a file above or paste study material here..."
        )

        st.subheader("2. Student Submission")
        sub_file = st.file_uploader("Upload Answer Document (.txt, .md, .pdf)", type=["txt", "md", "pdf"], key="sub_file")
        
        sub_file_text = read_uploaded_file(sub_file) if sub_file else ""

        student_input = st.text_area(
            "Or enter your response directly:",
            value=sub_file_text,
            height=120,
            placeholder="Upload an answer file above or type your response here..."
        )

        if st.button("Submit for Socratic Evaluation", type="primary"):
            if not learning_material.strip():
                st.error("Error: Learning material field cannot be empty.")
            elif not student_input.strip():
                st.warning("Please provide your submission text before evaluating.")
            else:
                with st.spinner("Evaluating against material bounds..."):
                    result = socratic_engine.evaluate(learning_material, student_input)
                    st.markdown("### Socratic Box Verdict")
                    st.info(result)

    # ----------------------------------
    # TAB 2: AUTHENTICITY VERIFICATION
    # ----------------------------------
    with tab2:
        st.subheader("Multimodal Authenticity Validation Engine")
        st.write("Compare your original assignment text against a short written summary to verify conceptual ownership.")

        st.markdown("#### Step 1: Original Written Work")
        orig_file = st.file_uploader("Upload Original Assignment (.txt, .md, .pdf)", type=["txt", "md", "pdf"], key="orig_file")
        orig_file_text = read_uploaded_file(orig_file) if orig_file else ""
        
        original_doc = st.text_area(
            "Or paste original assignment text:",
            value=orig_file_text,
            height=140,
            placeholder="Upload your full assignment above or paste text here..."
        )

        st.markdown("#### Step 2: Student Summary")
        sum_file = st.file_uploader("Upload Summary File (.txt, .md, .pdf)", type=["txt", "md", "pdf"], key="sum_file")
        sum_file_text = read_uploaded_file(sum_file) if sum_file else ""

        typed_summary = st.text_area(
            "Or type your summary:",
            value=sum_file_text,
            height=100,
            placeholder="Upload a summary file above or type a short summary here..."
        )

        if st.button("Validate Coherence Score"):
            if not original_doc.strip() or not typed_summary.strip():
                st.error("Both the original document and a summary input are required.")
            else:
                score = verifier.compute_similarity(original_doc, typed_summary)
                st.markdown(f"#### Cosine Similarity Score: **{score:.4f}**")
                
                if score >= 0.40:
                    st.success("✅ **VERIFICATION PASSED**: Conceptual alignment verified (Score ≥ 0.40). Assignment authenticated.")
                else:
                    st.session_state.auth_attempts += 1
                    remaining = 3 - st.session_state.auth_attempts
                    st.error(f"❌ **VERIFICATION FAILED**: Low semantic correlation (Score < 0.40). Remaining attempts: {remaining}")
                    
                    if st.session_state.auth_attempts >= 3:
                        st.error("⚠️ **FORMAL WARNING**: Plagiarism/AI overreliance flagged. Assignment voided (Score: 0).")

# ---------------------------------------------------------
# TEACHER MODE (PASSWORD PROTECTED)
# ---------------------------------------------------------
else:
    st.markdown('<div class="main-title">Teacher Mode & Anti-AI Curriculum Shield</div>', unsafe_allow_html=True)

    if not st.session_state.teacher_authenticated:
        st.subheader("🔒 Authentication Required")
        pwd = st.text_input("Enter Teacher Access Password:", type="password")
        if st.button("Login"):
            if pwd == "veratri2026":  # Default password
                st.session_state.teacher_authenticated = True
                st.rerun()
            else:
                st.error("Invalid password.")
    else:
        st.success("Authenticated as Educator")
        if st.button("Logout"):
            st.session_state.teacher_authenticated = False
            st.rerun()

        st.divider()
        teacher_option = st.selectbox(
            "Select Educator Tool",
            ["AI Vulnerability Audit", "Lesson Planning Suggestions", "Higher-Order Task Ideas"]
        )

        if teacher_option == "AI Vulnerability Audit":
            st.subheader("Assignment Vulnerability Analysis")
            
            t_file = st.file_uploader("Upload Draft Question / Prompt (.txt, .md, .pdf)", type=["txt", "md", "pdf"], key="t_file")
            t_file_text = read_uploaded_file(t_file) if t_file else ""

            draft_question = st.text_area(
                "Or paste draft prompt:",
                value=t_file_text,
                height=120,
                placeholder="Upload file or type draft prompt here..."
            )
            
            if st.button("Analyze Prompt Strength"):
                words = len(draft_question.split())
                score_str = "High Vulnerability" if words < 25 else "Moderate / Low Vulnerability"
                
                st.markdown(f"**Word Count:** {words}")
                st.markdown(f"**Vulnerability Index:** `{score_str}`")
                st.markdown("""
                **Recommendations to Increase AI Resistance:**
                1. Embed local, institution-specific, or contemporary regional context.
                2. Mandate observational data or primary source collection.
                3. Require multi-step reasoning with reflective student commentary.
                """)
