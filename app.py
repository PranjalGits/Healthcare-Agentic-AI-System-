# ============================================================
# ADVANCED AGENTIC AI HEALTHCARE SYSTEM
# ============================================================
# Main application file — Streamlit frontend
#
# ARCHITECTURE FLOW:
# User Input → Orchestrator → NLP Extractor + Specialist Agents
# → Synthesis Agent → Final Report + Voice Output
#
# PAGES:
# 1. Home Dashboard
# 2. AI Doctor (Main agentic diagnosis)
# 3. Disease Prediction (ML-based with NLP extraction)
# 4. Report Analyzer (Upload medical report PDF/text)
# 5. Voice + Vision (Audio + image diagnosis)
# 6. AI Chat (Conversational doctor with memory)
# ============================================================

import streamlit as st
import os
import tempfile
import time
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

# Auto-create results directory for audio files
os.makedirs("results", exist_ok=True)

# Backend imports
from backend.agents.orchestrator import MasterOrchestratorAgent
from backend.agents.specialists import AGENT_REGISTRY, get_agent
from backend.agents.synthesis import SynthesisAgent
from backend.agents.disease_prediction_agent import AgenticDiseasePredictionAgent
from backend.services.symptom_extractor import SymptomExtractorService
from backend.services.medical_data import medical_data
from backend.services.voice_service import transcribe_audio, text_to_speech
from backend.services.vision_service import analyze_medical_image

# ============================================================
# PAGE CONFIG & GLOBAL STYLING
# ============================================================
st.set_page_config(
    page_title="AI Healthcare Companion",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS — clean professional look
st.markdown("""
<style>
/* Main styling */
.stApp { font-family: 'Segoe UI', sans-serif; }

/* Agent cards */
.agent-card {
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #4CAF50;
    background: #f8f9fa;
    margin: 8px 0;
}

/* Urgency banners */
.urgency-emergency {
    background: #ffebee; border-left: 5px solid #f44336;
    padding: 12px; border-radius: 8px; margin: 10px 0;
}
.urgency-high {
    background: #fff3e0; border-left: 5px solid #ff9800;
    padding: 12px; border-radius: 8px; margin: 10px 0;
}
.urgency-medium {
    background: #fffde7; border-left: 5px solid #ffeb3b;
    padding: 12px; border-radius: 8px; margin: 10px 0;
}
.urgency-low {
    background: #e8f5e9; border-left: 5px solid #4caf50;
    padding: 12px; border-radius: 8px; margin: 10px 0;
}

/* Confidence bar */
.confidence-high { color: #2e7d32; font-weight: bold; }
.confidence-med  { color: #e65100; font-weight: bold; }
.confidence-low  { color: #c62828; font-weight: bold; }

/* Info box */
.info-box {
    background: #02c39a; border-radius: 8px;
    padding: 12px; margin: 8px 0;
    border-left: 3px solid #1976d2;
}

/* Disclaimer */
.disclaimer {
    background: #00a896; border-radius: 8px;
    padding: 12px; margin: 16px 0;
    border: 1px solid #f48fb1; font-size: 0.85em;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INITIALIZATION
# (Streamlit mein session state = memory across rerenders)
# ============================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "diagnosis_results" not in st.session_state:
    st.session_state.diagnosis_results = None
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = MasterOrchestratorAgent()
if "extractor" not in st.session_state:
    st.session_state.extractor = SymptomExtractorService()
if "synthesis_agent" not in st.session_state:
    st.session_state.synthesis_agent = SynthesisAgent()
if "disease_agent" not in st.session_state:
    st.session_state.disease_agent = AgenticDiseasePredictionAgent()

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.markdown("## 🏥 AI Healthcare Companion")
st.sidebar.markdown("*Advanced Agentic AI System*")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "🏠 Home",
        "🧠 AI Doctor (Agentic)",
        "🔬 Disease Prediction",
        "📄 Report Analyzer",
        "🎤 Voice + Vision",
        "💬 AI Chat Doctor",
        "ℹ️ About System"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**8 Specialist Agents:**
- 👨‍⚕️ General Physician
- ❤️ Cardiologist
- 🧠 Neurologist
- 🫁 Pulmonologist
- 🧘 Psychologist
- 🫀 Gastroenterologist
- 🩹 Dermatologist
- 🔬 Endocrinologist
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class='disclaimer'>
⚕️ <b>Disclaimer:</b> This AI system is for educational and informational purposes only. Always consult a qualified medical professional.
</div>
""", unsafe_allow_html=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def run_agents_parallel(selected_agents: list, patient_input: str,
                        image_desc: str = None) -> list:
    """
    Selected agents ko parallel mein chalao (faster results).
    ThreadPoolExecutor use karta hai — sab agents ek saath kaam karte hain.
    """
    results = []

    def run_single_agent(agent_key):
        agent = get_agent(agent_key)
        return agent.analyze(patient_input, image_desc)

    # Parallel execution
    with ThreadPoolExecutor(max_workers=len(selected_agents)) as executor:
        future_to_agent = {
            executor.submit(run_single_agent, ak): ak
            for ak in selected_agents
        }
        for future in as_completed(future_to_agent):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                agent_key = future_to_agent[future]
                results.append({
                    "agent": agent_key,
                    "specialty": agent_key,
                    "analysis": f"Error: {str(e)}",
                    "status": "error"
                })

    return results


def show_urgency_banner(urgency: str, message: str):
    """Urgency level ke hisaab se colored banner dikhao"""
    css_class = f"urgency-{urgency}"
    st.markdown(f'<div class="{css_class}">{message}</div>', unsafe_allow_html=True)


def show_disclaimer():
    """Medical disclaimer hamesha dikhao"""
    st.markdown("""
    <div class='disclaimer'>
    ⚕️ <b>Medical Disclaimer:</b> This analysis is generated by AI for informational purposes only.
    It does NOT replace professional medical advice, diagnosis, or treatment.
    Always consult a qualified healthcare provider before making any medical decisions.
    In case of emergency, call <b>108</b> immediately.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE 1: HOME DASHBOARD
# ============================================================
if page == "🏠 Home":
    st.title("🏥 AI Healthcare Companion")
    st.markdown("### Advanced Agentic AI Medical System")

    st.markdown("""
    <div class='info-box'>
    🤖 <b>Powered by 8 Specialist AI Agents</b> running in parallel — General Physician,
    Cardiologist, Neurologist, Pulmonologist, Psychologist, Gastroenterologist,
    Dermatologist, and Endocrinologist — all coordinated by a Master Orchestrator.
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🧠 AI Doctor (Agentic)")
        st.markdown("Describe your symptoms → Master AI decides which specialists to consult → Comprehensive diagnosis")
        if st.button("Try AI Doctor →", key="home_agent"):
            st.session_state["nav"] = "🧠 AI Doctor (Agentic)"
            st.rerun()

    with col2:
        st.markdown("#### 🔬 Disease Prediction")
        st.markdown("NLP extracts symptoms from your description → AI model predicts disease → Full treatment plan")
        if st.button("Predict Disease →", key="home_pred"):
            pass

    with col3:
        st.markdown("#### 💬 AI Chat Doctor")
        st.markdown("Full conversation memory → Ask follow-up questions → Doctor explains in detail")
        if st.button("Chat Now →", key="home_chat"):
            pass

    st.markdown("---")

    # System stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Specialist Agents", "8", "Active")
    col2.metric("Diseases in DB", "41+", "AI Model")
    col3.metric("Symptoms Recognized", "132+", "NLP + AI")
    col4.metric("AI Model", "LLaMA 3.3 70B", "Groq")

    st.markdown("---")
    show_disclaimer()


# ============================================================
# PAGE 2: AI DOCTOR (MAIN AGENTIC FEATURE)
# ============================================================
elif page == "🧠 AI Doctor (Agentic)":
    st.title("🧠 AI Doctor — Agentic Diagnosis System")
    st.markdown("""
    <div class='info-box'>
    <b>How it works:</b> You describe your problem → Master Orchestrator Agent decides which
    specialist doctors to consult → All selected agents analyze in PARALLEL → Synthesis Agent
    creates the final unified diagnosis. This mimics how real hospital MDT (Multi-Disciplinary Team) works.
    </div>
    """, unsafe_allow_html=True)

    # Input section
    st.markdown("### Describe Your Health Problem")
    patient_input = st.text_area(
        "Enter symptoms, complaints, or health concerns (any language — Hindi/English both work):",
        placeholder="e.g. 'I have been having severe headache for 3 days, feel dizzy, slight fever. Mere sir mein bahut dard hai aur chakkar aa rahe hain.'",
        height=120,
        key="agentic_input"
    )

    col1, col2 = st.columns(2)
    with col1:
        uploaded_image = st.file_uploader(
            "📸 Upload image (X-ray, skin, rash, injury...)",
            type=["jpg", "jpeg", "png", "webp"],
            key="agentic_image"
        )
    with col2:
        uploaded_report = st.file_uploader(
            "📄 Upload medical report (.txt)",
            type=["txt"],
            key="agentic_report"
        )

    run_diagnosis = st.button("🚀 Run Full AI Diagnosis", type="primary", use_container_width=True)

    if run_diagnosis and patient_input.strip():
        # ---- STEP 1: ORCHESTRATOR ----
        st.markdown("---")
        with st.status("🎯 Step 1: Orchestrator is analyzing your input...", expanded=True) as status:
            orchestrator = st.session_state.orchestrator

            plan = orchestrator.decide(
                patient_input,
                has_image=uploaded_image is not None,
                has_report=uploaded_report is not None
            )

            selected_agents = plan.get("selected_agents", ["general_physician"])
            urgency = plan.get("urgency", "medium")
            reasoning = plan.get("reasoning", "")

            st.write(f"✅ Selected {len(selected_agents)} specialists: {', '.join(selected_agents)}")
            st.write(f"📊 Urgency Level: **{urgency.upper()}**")
            st.write(f"💭 Reasoning: {reasoning}")
            status.update(label="✅ Orchestrator analysis complete!", state="complete")

        # Show urgency banner
        urgency_msg, _ = orchestrator.get_urgency_message(urgency)
        show_urgency_banner(urgency, urgency_msg)

        # ---- STEP 2: VISION ANALYSIS (if image) ----
        image_description = None
        if uploaded_image:
            with st.status("🔍 Step 2: Analyzing uploaded image...", expanded=True) as status:
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_image.name.split('.')[-1]}") as tmp:
                    tmp.write(uploaded_image.read())
                    tmp_path = tmp.name

                vision_result = analyze_medical_image(tmp_path, patient_input)
                os.unlink(tmp_path)

                if vision_result["success"]:
                    image_description = vision_result["visual_findings"]
                    st.write("✅ Image analyzed successfully")
                    with st.expander("View image analysis"):
                        st.write(image_description)
                else:
                    st.warning(f"⚠️ Image analysis unavailable: {vision_result.get('error', 'Unknown')}")
                status.update(label="✅ Image analysis complete!", state="complete")

        # ---- STEP 3: AGENTIC DISEASE PREDICTION (replaces ML model) ----
        with st.status("🧬 Step 3: Agentic Disease Prediction Agent analyzing...", expanded=True) as status:
            disease_agent = st.session_state.disease_agent
            agent_prediction = disease_agent.predict(patient_input, image_description)

            if agent_prediction.get("success") and agent_prediction.get("top_diseases"):
                top = agent_prediction["top_diseases"][0]
                st.write(f"✅ Top Prediction: **{top['disease']}** — {top['confidence']}% confidence")
                st.write(f"🔍 Symptoms identified: {', '.join(agent_prediction.get('extracted_symptoms', []))}")
                st.write(f"💡 {agent_prediction.get('immediate_advice', '')}")
            else:
                st.write("⚠️ Running specialist agents as primary analysis")
            status.update(label="✅ Agentic prediction complete!", state="complete")

        # ---- STEP 4: RUN SPECIALIST AGENTS IN PARALLEL ----
        with st.status(f"👥 Step 4: Running {len(selected_agents)} specialist agents in parallel...", expanded=True) as status:
            progress_bar = st.progress(0)

            # Report text agar uploaded hai
            report_text = None
            if uploaded_report:
                report_text = uploaded_report.read().decode("utf-8")
                patient_input_with_report = f"{patient_input}\n\nMedical Report Content:\n{report_text}"
            else:
                patient_input_with_report = patient_input

            specialist_results = run_agents_parallel(
                selected_agents,
                patient_input_with_report,
                image_description
            )

            progress_bar.progress(100)
            successful = sum(1 for r in specialist_results if r.get("status") == "success")
            st.write(f"✅ {successful}/{len(selected_agents)} specialists completed analysis")
            status.update(label="✅ All specialists analyzed!", state="complete")

        # ---- STEP 5: SYNTHESIS ----
        with st.status("🧩 Step 5: Synthesis Agent creating final diagnosis...", expanded=True) as status:
            synthesis_agent = st.session_state.synthesis_agent
            final_diagnosis = synthesis_agent.synthesize(
                patient_input,
                specialist_results,
                agent_prediction  # Agentic prediction pass karo (ML ki jagah)
            )
            status.update(label="✅ Final diagnosis ready!", state="complete")

        # ---- DISPLAY RESULTS ----
        st.markdown("---")
        st.markdown("## 📊 Complete Diagnosis Report")

        # Final diagnosis (main result)
        st.markdown(final_diagnosis)

        # ---- AGENTIC TOP 3 DISEASES WITH FULL TREATMENT ----
        if agent_prediction.get("success") and agent_prediction.get("top_diseases"):
            st.markdown("---")
            st.markdown("### 🧬 AI Agent — Top 3 Differential Diagnoses")
            st.caption(f"*Powered by: {agent_prediction.get('source', 'Agentic AI')}*")

            for disease_data in agent_prediction["top_diseases"]:
                rank = disease_data.get("rank", 1)
                disease_name = disease_data.get("disease", "Unknown")
                confidence = disease_data.get("confidence", 0)
                conf_color = disease_agent.get_confidence_color(confidence)

                rank_emoji = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "🔹")

                with st.expander(
                    f"{rank_emoji} #{rank} — {disease_name}   |   Confidence: {confidence}%",
                    expanded=(rank == 1)
                ):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**🔬 Clinical Reasoning:** {disease_data.get('reasoning', '')}")
                        st.markdown(f"**📖 Description:** {disease_data.get('description', '')}")
                    with col2:
                        st.markdown(f"**Confidence**")
                        st.progress(confidence / 100)
                        st.markdown(f"<span style='color:{conf_color};font-weight:bold;font-size:1.2em'>{confidence}%</span>", unsafe_allow_html=True)
                        st.caption(disease_data.get("confidence_label", ""))

                    # When to see doctor
                    wtsd = disease_data.get("when_to_see_doctor", {})
                    urgency_icon = disease_agent.get_urgency_color(wtsd.get("urgency", "soon"))
                    st.markdown(f"**{urgency_icon} When to See Doctor:** {wtsd.get('urgency_label', '')} *{wtsd.get('time_frame', '')}*")
                    if wtsd.get("red_flags"):
                        st.markdown("**🚨 Red Flags — Go immediately if:**")
                        for rf in wtsd["red_flags"]:
                            st.markdown(f"  - {rf}")

                    st.markdown("---")
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "💊 Medications", "🥗 Diet", "🏋️ Workout", "⚠️ Precautions", "🩺 About"
                    ])

                    with tab1:
                        meds = disease_data.get("medications", [])
                        for m in meds:
                            badge = "🟢 OTC" if m.get("type") == "OTC" else "🔴 Rx"
                            st.markdown(f"**{m.get('name', '')}** {badge} *{m.get('purpose', '')}*")
                        if not meds:
                            st.info("Consult doctor for specific medications")

                    with tab2:
                        for d in disease_data.get("diet", []):
                            st.markdown(f"• {d}")

                    with tab3:
                        for w in disease_data.get("workout", []):
                            st.markdown(f"• {w}")

                    with tab4:
                        for p in disease_data.get("precautions", []):
                            st.markdown(f"• {p}")

                    with tab5:
                        st.markdown(disease_data.get("description", ""))

        # Individual agent reports
        st.markdown("---")
        st.markdown("### 🏥 Individual Specialist Reports")
        for result in specialist_results:
            if result.get("status") == "success":
                with st.expander(f"📋 {result['agent']} ({result['specialty']})"):
                    st.markdown(result["analysis"])

        # Voice output
        st.markdown("---")
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔊 Read Diagnosis Aloud"):
                with st.spinner("Generating voice response..."):
                    audio_path = text_to_speech(
                        final_diagnosis,
                        "results/doctor_response.mp3"
                    )
                    if audio_path and os.path.exists(audio_path):
                        st.audio(audio_path, format="audio/mp3")
                    else:
                        st.warning("Voice generation failed. Install gTTS: pip install gTTS")

        show_disclaimer()

    elif run_diagnosis:
        st.warning("⚠️ Please enter your symptoms or health complaint first.")


# ============================================================
# PAGE 3: DISEASE PREDICTION — PURE AGENTIC AI (ML replaced)
# ============================================================
elif page == "🔬 Disease Prediction":
    st.title("🔬 Agentic Disease Prediction")
    st.markdown("""
    <div class='info-box'>
    <b>🧬 Pure Agentic AI Prediction:</b> ML model completely replaced with
    <b>LLaMA 3.3 70B Intelligent Agent</b> — understands natural language (Hindi/English),
    gives <b>Top 3 differential diagnoses</b> with confidence scores, and provides
    complete treatment plans from medical knowledge — no CSV dependency!
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Describe Your Symptoms")
    symptom_input = st.text_area(
        "Enter symptoms in any language — natural, conversational description works:",
        placeholder="e.g. 'I have severe headache, feel nauseous, and slight fever since yesterday evening' or 'Mujhe 3 din se sar dard, chakkar, aur ulti jaisi feeling hai'",
        height=120,
        key="pred_input"
    )

    col_a, col_b = st.columns(2)
    with col_a:
        age = st.text_input("Age", placeholder="e.g. 25")
    with col_b:
        gender = st.selectbox("Gender", ["Not specified", "Male", "Female", "Other"])

    if st.button("🧬 Predict with Agentic AI", type="primary", use_container_width=True):
        if symptom_input.strip():
            disease_agent = st.session_state.disease_agent

            # Add age/gender context if provided
            full_input = symptom_input
            if age:
                full_input += f" Patient age: {age}."
            if gender != "Not specified":
                full_input += f" Gender: {gender}."

            with st.status("🧠 Agentic AI analyzing your symptoms...", expanded=True) as status:
                st.write("🔍 Extracting and understanding symptoms...")
                st.write("🧬 Running differential diagnosis...")
                st.write("📋 Generating complete treatment plans...")
                result = disease_agent.predict(full_input)
                status.update(label="✅ Agentic prediction complete!", state="complete")

            if not result.get("success") or not result.get("top_diseases"):
                st.error("❌ Prediction failed. Please try again.")
            else:
                # ---- EXTRACTED SYMPTOMS ----
                st.markdown("### 🔍 Identified Symptoms")
                symptoms = result.get("extracted_symptoms", [])
                if symptoms:
                    symptom_html = " ".join([
                        f"<span style='background:#e8f5e9;padding:4px 12px;border-radius:15px;margin:3px;display:inline-block;font-size:0.9em'>✓ {s}</span>"
                        for s in symptoms
                    ])
                    st.markdown(symptom_html, unsafe_allow_html=True)

                st.markdown(f"*{result.get('symptom_summary', '')}*")

                # ---- OVERALL URGENCY ----
                overall_urg = result.get("overall_urgency", "soon")
                urg_icon = disease_agent.get_urgency_color(overall_urg)
                urg_messages = {
                    "emergency": "🚨 EMERGENCY — Call 108 or go to ER immediately!",
                    "urgent": "⚠️ URGENT — See a doctor today.",
                    "soon": "📅 See a doctor within this week.",
                    "routine": "✅ Routine — Monitor and consult when convenient."
                }
                st.info(f"{urg_icon} **Overall Urgency:** {urg_messages.get(overall_urg, '')}")
                st.markdown(f"**💡 Immediate Advice:** {result.get('immediate_advice', '')}")

                st.markdown("---")
                st.markdown("### 🏥 Top 3 Differential Diagnoses")

                for disease_data in result["top_diseases"]:
                    rank = disease_data.get("rank", 1)
                    disease_name = disease_data.get("disease", "Unknown")
                    confidence = disease_data.get("confidence", 0)
                    conf_color = disease_agent.get_confidence_color(confidence)
                    rank_emoji = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "🔹")

                    with st.expander(
                        f"{rank_emoji} #{rank}  {disease_name}   ·   {confidence}% Confidence",
                        expanded=(rank == 1)
                    ):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**📖 Description:** {disease_data.get('description', '')}")
                            st.markdown(f"**🔬 Why this diagnosis:** {disease_data.get('reasoning', '')}")
                        with col2:
                            st.markdown("**Confidence**")
                            st.progress(confidence / 100)
                            st.markdown(
                                f"<span style='color:{conf_color};font-weight:bold;font-size:1.4em'>{confidence}%</span>",
                                unsafe_allow_html=True
                            )
                            st.caption(disease_data.get("confidence_label", ""))

                        # When to see doctor block
                        wtsd = disease_data.get("when_to_see_doctor", {})
                        urg_icon2 = disease_agent.get_urgency_color(wtsd.get("urgency", "soon"))
                        st.markdown(f"**{urg_icon2} When to See Doctor:** `{wtsd.get('urgency_label', '')}` {wtsd.get('time_frame', '')}")

                        if wtsd.get("red_flags"):
                            st.markdown("**🚨 Go to doctor IMMEDIATELY if:**")
                            for rf in wtsd["red_flags"]:
                                st.markdown(f"  - ⚡ {rf}")

                        st.markdown("---")
                        t1, t2, t3, t4, t5 = st.tabs([
                            "💊 Medications", "🥗 Diet", "🏋️ Workout", "⚠️ Precautions", "📋 Full Info"
                        ])

                        with t1:
                            meds = disease_data.get("medications", [])
                            if meds:
                                for m in meds:
                                    badge_color = "#e8f5e9" if m.get("type") == "OTC" else "#fce4ec"
                                    badge_text = "OTC" if m.get("type") == "OTC" else "Prescription"
                                    med_name = m.get('name', '')
                                    med_purpose = m.get('purpose', '')
                                    st.markdown(f"**{med_name}** `{badge_text}` — *{med_purpose}*")
                                    st.markdown("")
                            else:
                                st.info("Consult a doctor for specific medications")

                        with t2:
                            diet_items = disease_data.get("diet", [])
                            for d in diet_items:
                                st.markdown(f"🥗 {d}")

                        with t3:
                            workout_items = disease_data.get("workout", [])
                            for w in workout_items:
                                st.markdown(f"🏃 {w}")

                        with t4:
                            prec_items = disease_data.get("precautions", [])
                            for p in prec_items:
                                st.markdown(f"⚠️ {p}")

                        with t5:
                            st.markdown(f"**Disease:** {disease_name}")
                            st.markdown(f"**Description:** {disease_data.get('description', '')}")
                            st.markdown(f"**Confidence:** {confidence}% ({disease_data.get('confidence_label', '')})")
                            st.markdown(f"**Clinical basis:** {disease_data.get('reasoning', '')}")

            show_disclaimer()
        else:
            st.warning("Please describe your symptoms first.")


# ============================================================
# PAGE 4: REPORT ANALYZER
# ============================================================
elif page == "📄 Report Analyzer":
    st.title("📄 Medical Report Analyzer")
    st.markdown("""
    <div class='info-box'>
    Upload any medical report (text format) and 8 specialist AI agents will analyze it
    from their respective perspectives. Then a Synthesis Agent creates a unified summary.
    </div>
    """, unsafe_allow_html=True)

    report_file = st.file_uploader("Upload medical report (.txt)", type=["txt"])
    additional_context = st.text_input(
        "Any additional context?",
        placeholder="e.g. 'Patient is 45 years old, has diabetes history'"
    )

    if report_file and st.button("📊 Analyze Report with All Specialists", type="primary"):
        report_text = report_file.read().decode("utf-8")

        st.markdown("---")
        with st.spinner("8 Specialist Agents analyzing your report in parallel..."):
            all_agents = list(AGENT_REGISTRY.keys())
            patient_context = f"{additional_context}\n\nReport:\n{report_text}" if additional_context else report_text

            specialist_results = run_agents_parallel(all_agents, patient_context)

            synthesis_agent = st.session_state.synthesis_agent
            final = synthesis_agent.synthesize(patient_context, specialist_results)

        st.markdown("## 📋 Multidisciplinary Analysis Report")
        st.markdown(final)

        st.markdown("---")
        st.markdown("### Individual Specialist Reports")
        for result in specialist_results:
            if result.get("status") == "success":
                with st.expander(f"📋 {result['agent']}"):
                    st.markdown(result["analysis"])

        show_disclaimer()


# ============================================================
# PAGE 5: VOICE + VISION  (Live Recording + Upload both supported)
# ============================================================
elif page == "🎤 Voice + Vision":
    st.title("🎤 Voice + Vision Diagnosis")
    st.markdown("""
    <div class='info-box'>
    <b>🎙️ Live Recording supported!</b> Click the mic to record your complaint directly
    in <b>any language (Hindi/English/mixed)</b> — Groq Whisper AI transcribes it instantly.
    Upload an image (skin/rash/injury) for visual analysis.
    </div>
    """, unsafe_allow_html=True)

    # ── Audio Input ──────────────────────────────────────────
    st.markdown("### 🎙️ Step 1 — Speak or Upload Your Complaint")

    audio_tab1, audio_tab2 = st.tabs(["🔴 Live Record (Recommended)", "📁 Upload Audio File"])

    audio_bytes = None  # will hold raw WAV bytes regardless of source

    with audio_tab1:
        st.markdown("""
        <div style='background:#1a1a2e;border-radius:10px;padding:16px;margin:8px 0;border:1px solid #333'>
        🎤 <b>Click the mic button below → Speak your symptoms → Click stop</b><br>
        <span style='color:#aaa;font-size:0.85em'>Works in Hindi, English, or mixed. No app install needed.</span>
        </div>
        """, unsafe_allow_html=True)

        live_audio = st.audio_input(
            "🔴 Click to start recording",
            key="live_mic"
        )
        if live_audio is not None:
            audio_bytes = live_audio.read()
            st.success("✅ Recording captured! Proceed to Step 2.")

    with audio_tab2:
        uploaded_audio = st.file_uploader(
            "Upload audio file (MP3 / WAV / M4A)",
            type=["mp3", "wav", "m4a"],
            key="audio_upload"
        )
        if uploaded_audio is not None:
            audio_bytes = uploaded_audio.read()
            st.success(f"✅ File uploaded: {uploaded_audio.name}")

    # ── Image Input ──────────────────────────────────────────
    st.markdown("### 📸 Step 2 — Upload Image ")
    image_file = st.file_uploader(
        "Upload image of skin condition, rash, wound, or injury (JPG/PNG)",
        type=["jpg", "jpeg", "png"],
        key="vision_image"
    )
    if image_file:
        col_prev, col_info = st.columns([1, 2])
        with col_prev:
            st.image(image_file, caption="Uploaded image", width=200)
        with col_info:
            st.info("✅ Image will be analyzed by Dermatologist AI agent")

    # ── Diagnose Button ───────────────────────────────────────
    st.markdown("### 🩺 Step 3 — Run Diagnosis")
    diagnose_btn = st.button(
        "🩺 Diagnose from Voice + Image",
        type="primary",
        use_container_width=True,
        disabled=(audio_bytes is None)
    )

    if audio_bytes is None:
        st.caption("⬆️ Record or upload audio first to enable diagnosis")

    if diagnose_btn and audio_bytes:
        # ---- Save audio bytes to temp file ----
        with st.status("🎤 Transcribing with Groq Whisper AI...", expanded=True) as status:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                audio_path = tmp.name

            transcription = transcribe_audio(audio_path)
            os.unlink(audio_path)

            if transcription["success"]:
                transcript_text = transcription["text"]
                lang = transcription.get("language", "unknown")
                st.write(f"✅ Transcription complete!")
                st.write(f"🌐 Detected language: **{lang}**")
                status.update(label="✅ Voice transcribed successfully!", state="complete")
            else:
                st.error(f"❌ Transcription failed: {transcription.get('error', 'Unknown error')}")
                st.stop()

        # Show what was heard
        st.markdown("### 🗣️ What AI Heard:")
        st.info(f'"{transcript_text}"')

        # Option to correct
        corrected = st.text_area(
            "✏️ Correct transcription if needed (optional):",
            value=transcript_text,
            height=80,
            key="correction"
        )
        final_text = corrected.strip() if corrected.strip() else transcript_text

        # ---- Analyze Image if provided ----
        image_desc = None
        if image_file:
            with st.status("🔍 Dermatologist AI analyzing image...", expanded=True) as status:
                image_file.seek(0)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(image_file.read())
                    img_path = tmp.name

                vision_result = analyze_medical_image(img_path, final_text)
                os.unlink(img_path)

                if vision_result["success"]:
                    image_desc = vision_result["visual_findings"]
                    st.write("✅ Visual findings identified")
                    with st.expander("👁️ View Dermatologist Image Analysis"):
                        st.markdown(image_desc)
                else:
                    st.warning("⚠️ Image analysis unavailable — proceeding with voice only")
                status.update(label="✅ Image analyzed!", state="complete")

        # ---- Orchestrator ----
        with st.status("🎯 Orchestrator deciding specialists...", expanded=True) as status:
            orchestrator = st.session_state.orchestrator
            plan = orchestrator.decide(final_text, has_image=image_file is not None)
            selected = plan.get("selected_agents", ["general_physician"])
            urgency = plan.get("urgency", "medium")
            st.write(f"✅ Selected: {', '.join(selected)}")
            st.write(f"📊 Urgency: **{urgency.upper()}**")
            status.update(label="✅ Specialists selected!", state="complete")

        urgency_msg, _ = orchestrator.get_urgency_message(urgency)
        show_urgency_banner(urgency, urgency_msg)

        # ---- Agentic Disease Prediction ----
        with st.status("🧬 Agentic Disease Prediction Agent analyzing...", expanded=True) as status:
            disease_agent = st.session_state.disease_agent
            agent_pred = disease_agent.predict(final_text, image_desc)
            if agent_pred.get("success") and agent_pred.get("top_diseases"):
                top = agent_pred["top_diseases"][0]
                st.write(f"✅ Top: **{top['disease']}** — {top['confidence']}% confidence")
            status.update(label="✅ Prediction complete!", state="complete")

        # ---- Specialist Agents ----
        with st.status(f"👥 Running {len(selected)} specialist agents...", expanded=True) as status:
            specialist_results = run_agents_parallel(selected, final_text, image_desc)
            st.write(f"✅ {len(specialist_results)} specialists analyzed")
            status.update(label="✅ Specialist analysis done!", state="complete")

        # ---- Synthesis ----
        with st.status("🧩 Synthesis Agent creating final report...", expanded=True) as status:
            final_report = st.session_state.synthesis_agent.synthesize(
                final_text, specialist_results, agent_pred
            )
            status.update(label="✅ Final report ready!", state="complete")

        # ---- Display Results ----
        st.markdown("---")
        st.markdown("## 🩺 Complete Voice + Vision Diagnosis")
        st.markdown(final_report)

        # Top 3 from agentic prediction
        if agent_pred.get("success") and agent_pred.get("top_diseases"):
            st.markdown("---")
            st.markdown("### 🧬 Top 3 Differential Diagnoses")
            for d in agent_pred["top_diseases"]:
                rank_emoji = {1: "🥇", 2: "🥈", 3: "🥉"}.get(d["rank"], "🔹")
                with st.expander(f"{rank_emoji} {d['disease']} — {d['confidence']}%", expanded=(d["rank"]==1)):
                    st.markdown(f"**Description:** {d.get('description','')}")
                    wtsd = d.get("when_to_see_doctor", {})
                    st.markdown(f"**When to see doctor:** {wtsd.get('urgency_label','')} — {wtsd.get('time_frame','')}")
                    t1,t2,t3,t4 = st.tabs(["💊 Medications","🥗 Diet","🏋️ Workout","⚠️ Precautions"])
                    with t1:
                        for m in d.get("medications",[]):
                            med_n = m.get("name",""); med_p = m.get("purpose",""); med_t = m.get("type","")
                            st.markdown(f"**{med_n}** `{med_t}` — *{med_p}*")
                    with t2:
                        for item in d.get("diet",[]): st.markdown(f"🥗 {item}")
                    with t3:
                        for item in d.get("workout",[]): st.markdown(f"🏃 {item}")
                    with t4:
                        for item in d.get("precautions",[]): st.markdown(f"⚠️ {item}")

        # Individual specialist reports
        st.markdown("---")
        st.markdown("### 🏥 Individual Specialist Reports")
        for res in specialist_results:
            if res.get("status") == "success":
                with st.expander(f"📋 {res['agent']}"):
                    st.markdown(res["analysis"])

        # Voice output of diagnosis
        st.markdown("---")
        if st.button("🔊 Read Diagnosis Aloud"):
            with st.spinner("Generating audio response..."):
                audio_out = text_to_speech(final_report, "results/voice_diagnosis.mp3")
                if audio_out and os.path.exists(audio_out):
                    st.audio(audio_out, format="audio/mp3")
                else:
                    st.warning("Install gTTS: pip install gTTS")

        show_disclaimer()


# ============================================================
# PAGE 6: AI CHAT DOCTOR (with memory)
# ============================================================
elif page == "💬 AI Chat Doctor":
    st.title("💬 AI Doctor Chat")
    st.markdown("""
    <div class='info-box'>
    <b>Memory-enabled conversation:</b> The AI doctor remembers everything you've said
    in this session. Ask follow-up questions, get explanations, discuss your health freely.
    </div>
    """, unsafe_allow_html=True)

    # Chat history display
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # System prompt for chat doctor
    CHAT_DOCTOR_SYSTEM = """You are Dr. AI, a knowledgeable and empathetic AI doctor assistant.

Your personality:
- Warm, caring, and professional
- Explain medical terms in simple language
- Always encourage professional medical consultation
- Never make definitive diagnoses — provide information and guidance

You remember the full conversation history and can reference earlier points.
Always end significant responses with a reminder to consult a real doctor.
If the patient seems in serious distress or mentions emergency symptoms, provide emergency hotline (108 in India).

Respond in the same language the patient uses (Hindi or English)."""

    if prompt := st.chat_input("Ask Dr. AI anything about your health..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Dr. AI is thinking..."):
                from groq import Groq
                client = Groq(api_key=os.getenv("GROQ_API_KEY"))

                # Build messages with full history (this is the MEMORY)
                messages = [{"role": "system", "content": CHAT_DOCTOR_SYSTEM}]
                messages.extend(st.session_state.chat_history)

                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    temperature=0.5,
                    max_tokens=700
                )

                reply = response.choices[0].message.content
                st.markdown(reply)

        # Save assistant reply to memory
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    # Clear chat button
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


# ============================================================
# PAGE 7: ABOUT SYSTEM
# ============================================================
elif page == "ℹ️ About System":
    st.title("ℹ️ About This System")

    st.markdown("""
    ## Advanced Agentic AI Healthcare System

    ### Architecture Overview
    This system uses a **Multi-Agent AI Architecture** where multiple specialized AI doctors
    work together to provide comprehensive medical analysis.

    ### How the Agentic System Works:

    **1. Master Orchestrator Agent**
    - Receives patient input (text/voice/image/report)
    - Analyzes the nature of the complaint
    - Decides which specialist agents to consult
    - Determines urgency level (Emergency/High/Medium/Low)

    **2. 8 Specialist AI Agents (run in PARALLEL)**
    - Each agent has a specialized medical system prompt
    - They analyze the same input from their domain's perspective
    - Running in parallel means faster results

    **3. NLP Symptom Extractor**
    - Converts natural language to structured symptoms
    - Works with Hindi, English, and mixed language
    - Feeds symptoms into the AI model

    **4. AI Disease Predictor Model**
    - Trained on 41+ diseases and 132+ symptoms
    - Now enhanced with NLP extraction (no more exact keyword matching!)
    - Returns confidence percentage with each prediction

    **5. Synthesis Agent**
    - Collects all specialist reports
    - Creates a unified, comprehensive final diagnosis
    - Adds actionable recommendations and safety warnings

    **6. Memory System**
    - Chat history maintained throughout session
    - AI Doctor remembers earlier parts of conversation

    ### Technology Stack
    - **Frontend:** Streamlit
    - **LLM:** LLaMA 3.3 70B via Groq API (ultra-fast inference)
    - **Speech-to-Text:** Groq Whisper Large V3
    - **Vision:** LLaMA 4 Scout (multimodal)
    - **AI Model:** Hugging Face
    - **Agent Framework:** Custom multi-agent with ThreadPoolExecutor

    ### Key Improvements Over Previous Version
    - ✅ AI model now uses NLP for symptom extraction (not exact keywords)
    - ✅ 8 specialists instead of 3
    - ✅ Real image analysis (not fake)
    - ✅ Confidence scoring on predictions
    - ✅ Memory in chat
    - ✅ API keys in .env (not hardcoded)
    - ✅ Clean code (no commented junk)
    - ✅ Medical disclaimer always present
    - ✅ Parallel agent execution for speed
    """)

    show_disclaimer()