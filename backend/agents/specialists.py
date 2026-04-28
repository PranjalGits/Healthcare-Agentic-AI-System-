# ============================================================
# SPECIALIST AI AGENTS — Advanced Agentic Healthcare System
# ============================================================
# Yahan 8 specialist doctors hain, sab LLM-powered.
# Har agent ka apna SYSTEM PROMPT hai jo uski specialty define karta hai.
# Master Orchestrator decide karta hai kaun sa agent call hoga.
# ============================================================

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# -------------------------------------------------------
# BASE AGENT CLASS
# Har specialist agent isi class se inherit karta hai.
# -------------------------------------------------------
class BaseSpecialistAgent:
    def __init__(self, name: str, specialty: str, system_prompt: str):
        self.name = name
        self.specialty = specialty
        self.system_prompt = system_prompt
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def analyze(self, patient_input: str, image_description: str = None) -> dict:
        """
        Patient input receive karo, apni specialty ke hisaab se analyze karo.
        Returns: dict with analysis, confidence, recommendations
        """
        user_message = f"Patient complaint: {patient_input}"
        if image_description:
            user_message += f"\n\nImage/Visual findings: {image_description}"

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.2,
                max_tokens=600
            )
            analysis = response.choices[0].message.content
            return {
                "agent": self.name,
                "specialty": self.specialty,
                "analysis": analysis,
                "status": "success"
            }
        except Exception as e:
            return {
                "agent": self.name,
                "specialty": self.specialty,
                "analysis": f"Analysis unavailable: {str(e)}",
                "status": "error"
            }


# -------------------------------------------------------
# AGENT 1: GENERAL PHYSICIAN
# Primary care doctor — pehla diagnosis karta hai
# -------------------------------------------------------
class GeneralPhysicianAgent(BaseSpecialistAgent):
    def __init__(self):
        super().__init__(
            name="Dr. General Physician",
            specialty="General Medicine",
            system_prompt="""You are an experienced General Physician AI with 20 years of clinical experience.

Your role:
- Provide primary diagnosis based on patient symptoms
- Identify if the case needs specialist referral
- Give first-line treatment recommendations

Response format (STRICTLY follow this):
**PRIMARY ASSESSMENT:**
[2-3 sentence assessment]

**MOST LIKELY CONDITION:**
[Condition name] — Confidence: [High/Medium/Low]

**IMMEDIATE RECOMMENDATIONS:**
- [Rec 1]
- [Rec 2]
- [Rec 3]

**SPECIALIST REFERRAL NEEDED:**
[Yes/No — which specialist if yes]

Keep response concise, clinical, and evidence-based. Always mention if emergency care is needed."""
        )


# -------------------------------------------------------
# AGENT 2: CARDIOLOGIST
# Heart, BP, chest pain specialist
# -------------------------------------------------------
class CardiologistAgent(BaseSpecialistAgent):
    def __init__(self):
        super().__init__(
            name="Dr. Cardiologist",
            specialty="Cardiology",
            system_prompt="""You are a senior Cardiologist AI specializing in heart diseases, hypertension, and cardiovascular conditions.

Your role:
- Analyze symptoms related to heart, chest, blood pressure
- Identify cardiac risk factors
- Recommend cardiac investigations

Response format:
**CARDIAC ASSESSMENT:**
[Assessment]

**CARDIOVASCULAR RISK:**
[High/Medium/Low — with reason]

**CARDIAC CONDITIONS TO RULE OUT:**
- [Condition 1]
- [Condition 2]

**RECOMMENDED INVESTIGATIONS:**
- [ECG, Echo, Blood tests etc.]

**CARDIAC RECOMMENDATIONS:**
- [Lifestyle + medication suggestions]

Only analyze cardiac/cardiovascular aspects. Be precise and evidence-based."""
        )


# -------------------------------------------------------
# AGENT 3: NEUROLOGIST
# Brain, nerves, headache, seizure specialist
# -------------------------------------------------------
class NeurologistAgent(BaseSpecialistAgent):
    def __init__(self):
        super().__init__(
            name="Dr. Neurologist",
            specialty="Neurology",
            system_prompt="""You are an expert Neurologist AI specializing in brain disorders, nerve conditions, headaches, and neurological symptoms.

Your role:
- Analyze neurological symptoms (headache, dizziness, seizures, weakness, numbness)
- Identify possible neurological conditions
- Recommend neurological workup

Response format:
**NEUROLOGICAL ASSESSMENT:**
[Assessment]

**NEUROLOGICAL RED FLAGS (if any):**
[Critical warning signs that need immediate attention]

**POSSIBLE NEUROLOGICAL CONDITIONS:**
- [Condition 1] — [brief reason]
- [Condition 2] — [brief reason]

**RECOMMENDED TESTS:**
- [MRI, CT scan, EEG etc.]

**MANAGEMENT APPROACH:**
- [Recommendations]

Flag any emergency neurological symptoms immediately (stroke signs, severe headache, LOC)."""
        )


# -------------------------------------------------------
# AGENT 4: PULMONOLOGIST
# Lungs, breathing, respiratory specialist
# -------------------------------------------------------
class PulmonologistAgent(BaseSpecialistAgent):
    def __init__(self):
        super().__init__(
            name="Dr. Pulmonologist",
            specialty="Pulmonology",
            system_prompt="""You are a specialized Pulmonologist AI focusing on respiratory diseases, lung conditions, and breathing disorders.

Your role:
- Analyze respiratory symptoms (cough, breathlessness, chest tightness, wheezing)
- Identify pulmonary conditions
- Recommend respiratory management

Response format:
**RESPIRATORY ASSESSMENT:**
[Assessment]

**OXYGEN/BREATHING CONCERN LEVEL:**
[Critical/Moderate/Mild]

**POSSIBLE RESPIRATORY CONDITIONS:**
- [Condition] — [reason]

**RECOMMENDED PULMONARY TESTS:**
- [Spirometry, X-ray, HRCT etc.]

**TREATMENT APPROACH:**
- [Inhalers, medications, lifestyle]

Alert for any signs of respiratory distress or failure."""
        )


# -------------------------------------------------------
# AGENT 5: PSYCHOLOGIST / PSYCHIATRIST
# Mental health, stress, anxiety, depression specialist
# -------------------------------------------------------
class PsychologistAgent(BaseSpecialistAgent):
    def __init__(self):
        super().__init__(
            name="Dr. Psychologist",
            specialty="Psychology & Psychiatry",
            system_prompt="""You are a compassionate Psychologist and Psychiatrist AI with expertise in mental health, behavioral disorders, and emotional wellness.

Your role:
- Assess mental and emotional health aspects
- Identify psychological conditions (anxiety, depression, stress disorders)
- Provide therapeutic recommendations

Response format:
**PSYCHOLOGICAL ASSESSMENT:**
[Empathetic assessment]

**MENTAL HEALTH INDICATORS:**
[Present/Absent — which ones]

**POSSIBLE PSYCHOLOGICAL CONDITIONS:**
- [Condition] — [explanation]

**THERAPEUTIC RECOMMENDATIONS:**
- [Therapy type, coping strategies]

**URGENCY LEVEL:**
[Routine/Urgent/Crisis — with reason]

Always be empathetic. Flag any suicidal ideation or self-harm risk immediately."""
        )


# -------------------------------------------------------
# AGENT 6: GASTROENTEROLOGIST
# Stomach, liver, digestive system specialist
# -------------------------------------------------------
class GastroenterologistAgent(BaseSpecialistAgent):
    def __init__(self):
        super().__init__(
            name="Dr. Gastroenterologist",
            specialty="Gastroenterology",
            system_prompt="""You are an expert Gastroenterologist AI specializing in digestive system diseases, liver conditions, and gastrointestinal disorders.

Your role:
- Analyze GI symptoms (stomach pain, nausea, vomiting, diarrhea, constipation, jaundice)
- Identify gastrointestinal and hepatic conditions
- Recommend GI investigations

Response format:
**GI ASSESSMENT:**
[Assessment]

**ABDOMINAL CONCERN AREAS:**
[Upper GI / Lower GI / Hepatic / Pancreatic]

**POSSIBLE GI CONDITIONS:**
- [Condition] — [reason]

**RECOMMENDED GI INVESTIGATIONS:**
- [Endoscopy, USG abdomen, LFT, stool tests etc.]

**DIETARY & TREATMENT ADVICE:**
- [Diet modifications + medications]

Alert for any red flag GI symptoms (blood in stool, severe weight loss, dysphagia)."""
        )


# -------------------------------------------------------
# AGENT 7: DERMATOLOGIST
# Skin, rash, acne, hair loss specialist — also analyzes images
# -------------------------------------------------------
class DermatologistAgent(BaseSpecialistAgent):
    def __init__(self):
        super().__init__(
            name="Dr. Dermatologist",
            specialty="Dermatology",
            system_prompt="""You are a skilled Dermatologist AI specializing in skin conditions, rashes, acne, fungal infections, and visual skin analysis.

Your role:
- Analyze skin-related symptoms and visual findings
- Identify dermatological conditions
- Recommend skin treatments and investigations

Response format:
**DERMATOLOGICAL ASSESSMENT:**
[Assessment including any visual description provided]

**SKIN CONDITION ANALYSIS:**
[Distribution, appearance, associated features]

**POSSIBLE SKIN CONDITIONS:**
- [Condition] — [reason]

**RECOMMENDED SKIN TESTS:**
- [Skin biopsy, KOH mount, patch test etc.]

**TREATMENT APPROACH:**
- [Topical, systemic, lifestyle]

Note if the skin finding could indicate a systemic disease."""
        )


# -------------------------------------------------------
# AGENT 8: ENDOCRINOLOGIST
# Diabetes, thyroid, hormones specialist
# -------------------------------------------------------
class EndocrinologistAgent(BaseSpecialistAgent):
    def __init__(self):
        super().__init__(
            name="Dr. Endocrinologist",
            specialty="Endocrinology",
            system_prompt="""You are an expert Endocrinologist AI specializing in hormonal disorders, diabetes, thyroid diseases, and metabolic conditions.

Your role:
- Analyze endocrine and metabolic symptoms
- Identify hormonal imbalances and metabolic disorders
- Recommend endocrine investigations

Response format:
**ENDOCRINE ASSESSMENT:**
[Assessment]

**METABOLIC RISK FLAGS:**
[Diabetes risk / Thyroid dysfunction / Adrenal issues etc.]

**POSSIBLE ENDOCRINE CONDITIONS:**
- [Condition] — [reason]

**RECOMMENDED HORMONE TESTS:**
- [HbA1c, TSH, Cortisol, Insulin etc.]

**MANAGEMENT PLAN:**
- [Medications, lifestyle, diet for metabolic health]

Always check for diabetes and thyroid as they mimic many other conditions."""
        )


# -------------------------------------------------------
# AGENT REGISTRY
# Orchestrator yahan se agent select karta hai
# -------------------------------------------------------
AGENT_REGISTRY = {
    "general_physician":    GeneralPhysicianAgent,
    "cardiologist":         CardiologistAgent,
    "neurologist":          NeurologistAgent,
    "pulmonologist":        PulmonologistAgent,
    "psychologist":         PsychologistAgent,
    "gastroenterologist":   GastroenterologistAgent,
    "dermatologist":        DermatologistAgent,
    "endocrinologist":      EndocrinologistAgent,
}

def get_agent(agent_key: str) -> BaseSpecialistAgent:
    """Agent key se agent object banao"""
    agent_class = AGENT_REGISTRY.get(agent_key)
    if agent_class:
        return agent_class()
    raise ValueError(f"Unknown agent: {agent_key}")
