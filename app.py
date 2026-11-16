import os
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import speech_recognition as sr
    HAS_SR = True
except ImportError:
    HAS_SR = False


# Page Configuration & Styling

st.set_page_config(
    page_title="VERA-TRI | Socratic Box",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
    <style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1E293B; }
    .sub-title { font-size: 1rem; color: #64748B; margin-bottom: 2rem; }
    .verdict-box { padding: 1rem; border-radius: 8px; margin-top: 1rem; font-weight: 600; }
    .verdict-correct { background-color: #DCFCE7; color: #166534; border: 1px solid #86EFAC; }
    .verdict-incorrect { background-color: #FEE2E2; color: #991B1B; border: 1px solid #FCA5A5; }
    </style>
""", unsafe_allow_html=True)


# Session State Initialization

if "teacher_authenticated" not in st.session_state:
    st.session_state.teacher_authenticated = False
if "auth_attempts" not in st.session_state:
    st.session_state.auth_attempts = 0


# Core Logic Engines

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
                "*Hint:* Review the primary mechanisms outlined in paragraph 2 of your material. "
                "You substituted the core concept with external assumptions."
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
    """Module 2: Cosine Similarity authenticity verification engine."""
    def __init__(self, threshold: float = 0.40):
        self.threshold = threshold

    @staticmethod
    def compute_similarity(text_a: str, text_b: str) -> float:
        if not text_a.strip() or not text_b.strip():
            return 0.0
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
        return float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])

    @staticmethod
    def record_audio_summary() -> str:
        """Captures voice input via system microphone using SpeechRecognition."""
        if not HAS_SR:
            st.error("SpeechRecognition library is missing. Install via `pip install SpeechRecognition PyAudio`.")
            return ""

        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎙️ Listening... Speak your oral summary clearly (10s max).")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            try:
                audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
                st.success("⌛ Transcribing audio...")
                text = recognizer.recognize_google(audio)
                return text
            except sr.WaitTimeoutError:
                st.warning("Listening timed out. No speech detected.")
            except sr.UnknownValueError:
                st.error("Could not understand audio input.")
            except Exception as e:
                st.error(f"Microphone error: {e}")
        return ""


# Sidebar & Navigation

st.sidebar.title("🛡️ VERA-TRI System")
st.sidebar.caption("Triple-Engine Curriculum Shielding Platform")

role = st.sidebar.radio("Select Operational Role", ["Student Mode", "Teacher Mode (Protected)"])

api_key_input = st.sidebar.text_input("OpenAI API Key (Optional)", type="password")
socratic_engine = SocraticSandbox(api_key=api_key_input)
verifier = AuthenticityVerifier(threshold=0.40)


# STUDENT MODE

if role == "Student Mode":
    st.markdown('<div class="main-title">Socratic Sandbox</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Constrained AI tutoring designed for minimal feedback, '
        'material grounding, and metacognitive reflection.</div>', 
        unsafe_allow_html=True
    )

    tab1, tab2 = st.tabs(["📝 Module 1: Guided Socratic Sandbox", "🎙️ Module 2: Authenticity Verification"])

    with tab1:
        st.subheader("1. Grounding Material (Mandatory)")
        learning_material = st.text_area(
            "Paste your reference text or lesson material here:",
            height=180,
            placeholder="Input study material. The AI will judge your answers strictly based on this scope."
        )

        st.subheader("2. Student Submission")
        student_input = st.text_area(
            "Enter your answer or conceptual explanation:",
            height=120,
            placeholder="Type your response or attempt here..."
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

    with tab2:
        st.subheader("Multimodal Authenticity Validation Engine")
        st.write("Compare your original assignment text against an oral or typed summary to verify conceptual ownership.")

        original_doc = st.text_area("Original Assignment Text:", height=150, placeholder="Paste your complete written assignment...")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Option A: Live Voice Recording**")
            recorded_text = ""
            if st.button("🎙️ Record Voice Summary"):
                recorded_text = verifier.record_audio_summary()
                if recorded_text:
                    st.session_state["oral_summary_text"] = recorded_text

        with col2:
            st.markdown("**Option B: Typed Summary**")
            typed_summary = st.text_area(
                "Manual Summary Input:",
                value=st.session_state.get("oral_summary_text", ""),
                height=100
            )

        final_summary = typed_summary if typed_summary else st.session_state.get("oral_summary_text", "")

        if st.button("Validate Coherence Score"):
            if not original_doc.strip() or not final_summary.strip():
                st.error("Both the original document and a summary input are required.")
            else:
                score = verifier.compute_similarity(original_doc, final_summary)
                st.markdown(f"#### Cosine Similarity Score: **{score:.4f}**")
                
                if score >= 0.40:
                    st.success("✅ **VERIFICATION PASSED**: Conceptual alignment verified (Score ≥ 0.40). Assignment authenticated.")
                else:
                    st.session_state.auth_attempts += 1
                    remaining = 3 - st.session_state.auth_attempts
                    st.error(f"❌ **VERIFICATION FAILED**: Low semantic correlation (Score < 0.40). Remaining attempts: {remaining}")
                    
                    if st.session_state.auth_attempts >= 3:
                        st.critical("⚠️ **FORMAL WARNING**: Plagiarism/AI overreliance flagged. Assignment voided (Score: 0).")


# TEACHER MODE (PASSWORD PROTECTED)

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
            draft_question = st.text_area("Paste draft prompt or assignment question:", height=120)
            
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