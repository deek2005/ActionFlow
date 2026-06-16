import streamlit as st
import pandas as pd
import json
import re
import tempfile
import os
from datetime import datetime
import dateparser
import spacy
from groq import Groq
import base64

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(page_title="ActionFlow", layout="wide", initial_sidebar_state="expanded")

# -------------------------------
# Custom CSS for smooth cursor & glassmorphism
# -------------------------------
st.markdown("""
<style>
    /* Smooth cursor transition & custom cursor */
    * {
        cursor: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="24" height="27" viewBox="0 0 24 27"><polygon points="2,2 22,13 12,16 2,22" fill="%234285f4" stroke="%23ffffff" stroke-width="1.5"/><circle cx="12" cy="16" r="2" fill="white"/></svg>') 12 8, auto;
        transition: all 0.2s cubic-bezier(0.2, 0.9, 0.4, 1.1);
    }
    /* Glassmorphic background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        background-attachment: fixed;
    }
    /* Cards and input areas */
    .stTextArea textarea, .stButton button, .stFileUploader, .stDataFrame {
        background: rgba(255,255,255,0.1) !important;
        backdrop-filter: blur(12px);
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.2);
        color: white !important;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: scale(1.02);
        background: rgba(255,255,255,0.2) !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        cursor: pointer;
    }
    h1, h2, h3, .stMarkdown, .stMetric {
        color: #f0f0f0;
    }
    /* Editable table styling */
    .stDataFrame {
        background: rgba(0,0,0,0.6);
        border-radius: 16px;
        overflow: auto;
    }
    /* Custom scrollbar */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: #1e1e2f; border-radius: 10px; }
    ::-webkit-scrollbar-thumb { background: #888; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #aaa; }
    /* Smooth fade-in for results */
    .element-container {
        animation: fadeInUp 0.5s ease-out;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Title & description
# -------------------------------
st.title("✨ ActionFlow – Cognitive Meeting Intelligence")
st.markdown("*Turn conversations into commitments — hybrid AI, smooth as glass.*")

# -------------------------------
# Load spaCy model (cached)
# -------------------------------
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# -------------------------------
# Sidebar: API key & info
# -------------------------------
with st.sidebar:
    st.header("⚙️ AI Engine")
    groq_api_key = st.text_input("Groq API Key", type="password", 
                                 help="Get free key from console.groq.com")
    if groq_api_key:
        groq_client = Groq(api_key=groq_api_key)
    else:
        groq_client = None
    st.markdown("---")
    st.markdown("###  Hybrid Intelligence")
    st.markdown("""
    - **Groq Llama 3.1 Instant** (fast & free)
    - **spaCy NLP** for entity recognition
    - **Regex patterns** for instant matches
    - **dateparser** for natural deadlines
    """)
    st.markdown("###  How to use")
    st.markdown("1. Paste transcript or upload audio\n2. Click **Extract**\n3. Edit table inline\n4. Download CSV/JSON")

# -------------------------------
# Hybrid extraction function
# -------------------------------
def extract_actions_hybrid(transcript):
    """Combine regex, spaCy NER, and Groq LLM for robust action extraction"""
    actions = []
    
    # ---- Rule-based pass (regex) ----
    speaker_pattern = r"(?P<assignee>\w+):\s*(?:I'?ll|I will|We need to|needs to)?\s*(?P<action>.+?)\s*(?:by|before|on)\s*(?P<deadline>.+?)(?:\.|\n|$)"

    patterns = [
    (speaker_pattern, "speaker"),
    (r"(?P<assignee>\w+) (?:will|to) (?P<action>.+?) (?:by|on|before) (?P<deadline>.+?)(?:\n|$)", "by"),
    (r"Action item: (?P<action>.+?) for (?P<assignee>\w+)(?: due (?P<deadline>.+?))?", "action"),
    (r"(?P<assignee>\w+) needs to (?P<action>.+?)(?:$|\.)", "needs"),
    (r"(?P<action>[A-Za-z].+?) assigned to (?P<assignee>\w+)", "assigned")
    ]
    for pattern, _ in patterns:
        matches = re.finditer(pattern, transcript, re.IGNORECASE)
        for m in matches:
            action = m.groupdict().get("action", "TBD").strip()
            assignee = m.groupdict().get("assignee", "TBD").strip()
            deadline_raw = m.groupdict().get("deadline", "TBD").strip()
            if action != "TBD":
                actions.append({
                    "Action": action,
                    "Assignee": assignee,
                    "Deadline": deadline_raw
                })
    
    # ---- LLM pass (Groq) to catch complex items ----
    if groq_client and len(transcript) < 4000:
        prompt = f"""
Extract action items from the meeting transcript.

Return ONLY valid JSON in this format:

{{
  "actions": [
    {{
      "action": "task description",
      "assignee": "person name",
      "deadline": "deadline or TBD"
    }}
  ]
}}

Transcript:
{transcript[:3000]}
"""
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            content = content.replace("```json", "").replace("```", "").strip()

            data = json.loads(content)
            if isinstance(data, dict) and "actions" in data:
                llm_items = data["actions"]
            elif isinstance(data, list):
                llm_items = data
            else:
                llm_items = []
            for item in llm_items:
                actions.append({
                    "Action": item.get("action", "TBD"),
                    "Assignee": item.get("assignee", "TBD"),
                    "Deadline": item.get("deadline", "TBD")
                })
        except Exception as e:
            st.warning(f"LLM fallback skipped: {e}")
    
    # ---- Deduplicate by action description ----
    seen = set()
    unique = []
    for a in actions:
        key = a["Action"].lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(a)
    
    # ---- Parse natural language deadlines ----
    for a in unique:
        if a["Deadline"] and a["Deadline"] != "TBD":
            parsed = dateparser.parse(a["Deadline"], settings={'PREFER_DATES_FROM': 'future'})
            if parsed:
                a["Deadline"] = parsed.date().isoformat()
            else:
                a["Deadline"] = "TBD"
    
    # ---- Final fallback: if no actions, use spaCy to find verbs + persons ----
    if not unique:
        doc = nlp(transcript[:2000])
        persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]
        if persons and verbs:
            unique.append({
                "Action": verbs[0] + " ...",
                "Assignee": persons[0],
                "Deadline": "TBD"
            })
    
    return unique

# -------------------------------
# Audio transcription using Groq Whisper
# -------------------------------
def transcribe_audio(file_bytes, filename):
    if not groq_client:
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                file=(filename, f.read()),
                model="whisper-large-v3"
            )
        return transcription.text
    except Exception as e:
        st.error(f"Transcription failed: {e}")
        return None
    finally:
        os.unlink(tmp_path)

# -------------------------------
# Main UI: two columns
# -------------------------------
col_left, col_right = st.columns([1, 1], gap="medium")

with col_left:
    st.subheader("📄 Input")
    input_type = st.radio(
    "Choose input type",
    ["📝 Text Transcript", "🎤 Audio File (MP3/WAV)"],
    horizontal=True
    )
    
    transcript = ""
    
    if input_type == "📝 Text Transcript":
        sample_text = """Meeting: Product Launch Review – April 10, 2025
John: We need to finalize the landing page copy by Friday.
Sarah: I'll handle the social media graphics by April 15.
Mike: Contact legal tomorrow about the disclaimer.
Action item: User testing completed by next Wednesday – assigned to Priya."""
        transcript = st.text_area("Paste transcript or edit sample", value=sample_text, height=280)
    
    elif input_type == "🎤 Audio File (MP3/WAV)":
        audio_file = st.file_uploader("Upload audio", type=["mp3", "wav", "m4a"])
        if audio_file and groq_client:
            if st.button("🗣️ Transcribe Audio"):
                with st.spinner("Transcribing with Whisper (Groq)..."):
                    transcribed = transcribe_audio(audio_file.read(), audio_file.name)
                    if transcribed:
                        st.session_state["transcript"] = transcribed
                        transcript = transcribed
                        st.success("Transcription complete!")
                        st.text_area("Transcribed text", transcript, height=200)
        elif audio_file and not groq_client:
            st.warning("Enter Groq API key in sidebar for transcription.")
    # Extract button
    if st.button("⚡ Extract Action Items (Hybrid AI)", type="primary", use_container_width=True):
        if not transcript.strip():
            st.warning("Please provide a transcript or audio.")
        elif not groq_client:
            st.error("Enter your Groq API key in the sidebar.")
        else:
            with st.spinner("Hybrid AI at work (regex + LLM + NLP)..."):
                actions = extract_actions_hybrid(transcript)
                if actions:
                    df = pd.DataFrame(actions)
                    # Ensure correct column order
                    df = df[["Action", "Assignee", "Deadline"]] if all(col in df.columns for col in ["Action","Assignee","Deadline"]) else df
                    st.session_state["action_df"] = df
                    st.success(f" Extracted {len(df)} action items")
                else:
                    st.error("No action items found. Try a different transcript.")

with col_right:
    st.subheader("✅ Action Items (Editable)")
    if "action_df" in st.session_state and not st.session_state.action_df.empty:
        edited_df = st.data_editor(st.session_state.action_df, num_rows="dynamic", use_container_width=True, key="editor")
        st.session_state.action_df = edited_df
        
        # Statistics row
        st.markdown("---")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Actions", len(edited_df))
        col_b.metric("With Deadlines", edited_df[edited_df["Deadline"] != "TBD"].shape[0])
        col_c.metric("Assigned", edited_df[edited_df["Assignee"] != "TBD"].shape[0])
        
        # Export options
        st.markdown("### 📎 Export")
        csv_data = edited_df.to_csv(index=False).encode("utf-8")
        json_data = edited_df.to_json(orient="records", indent=2).encode("utf-8")
        col_csv, col_json = st.columns(2)
        with col_csv:
            st.download_button("📥 Download CSV", csv_data, "actionflow_tasks.csv", "text/csv", use_container_width=True)
        with col_json:
            st.download_button("📥 Download JSON", json_data, "actionflow_tasks.json", "application/json", use_container_width=True)
    else:
        st.info("👈 No actions yet. Upload a transcript or audio and click 'Extract'.")
        st.image("https://img.icons8.com/fluency/96/null/meeting.png", width=100)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("ActionFlow | Hybrid AI: Groq Llama 3 + spaCy NLP + dateparser | Smooth cursor & glassmorphic UI | Built for FlowZint 2026")