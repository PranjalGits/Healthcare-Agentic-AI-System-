# ============================================================
# AGENTIC DISEASE PREDICTION AGENT
# ============================================================
# ML model COMPLETELY replace — Pure LLM-based intelligent agent
#
# ML model ki limitations thi:
#   - Sirf 41 diseases jaanta tha
#   - Sirf CSV ka data use karta tha
#   - Koi thinking nahi thi — sirf pattern match
#   - Hindi/natural language samajh nahi aata tha
#
# Naya Agentic approach:
#   - LLaMA 3.3 70B ki full medical knowledge use karta hai
#   - Top 3 diseases with realistic confidence %
#   - Har disease ke liye complete: description, medications,
#     diet, workout, precautions, when-to-see-doctor
#   - Hindi/English/mixed language — sab samajhta hai
#   - Internet-level medical knowledge — CSV ka mohtaj nahi
#   - THINKING ability hai — symptoms ke combinations samjhta hai
# ============================================================

import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# DISEASE PREDICTION AGENT SYSTEM PROMPT
# ============================================================
DISEASE_AGENT_SYSTEM_PROMPT = """You are an expert AI Medical Diagnostician with deep knowledge of clinical medicine, pathology, and symptomatology. You have the medical knowledge equivalent to a senior physician with 25+ years of experience.

Your task: Analyze patient symptoms and provide a comprehensive differential diagnosis with full treatment information.

You MUST respond ONLY with valid JSON. No explanation before or after. No markdown backticks. Pure JSON only.

Required JSON structure:
{
  "extracted_symptoms": ["symptom1", "symptom2", ...],
  "symptom_summary": "Brief clinical summary of what the patient has",
  "top_diseases": [
    {
      "rank": 1,
      "disease": "Disease Name",
      "confidence": 87,
      "confidence_label": "High",
      "reasoning": "Why this disease fits the symptoms (2-3 sentences)",
      "description": "What this disease is — clear explanation for patient (3-4 sentences)",
      "medications": [
        {"name": "Medicine name", "type": "OTC/Prescription", "purpose": "what it does"},
        {"name": "Medicine name", "type": "OTC/Prescription", "purpose": "what it does"},
        {"name": "Medicine name", "type": "OTC/Prescription", "purpose": "what it does"}
      ],
      "diet": [
        "Specific dietary recommendation 1",
        "Specific dietary recommendation 2",
        "Specific dietary recommendation 3",
        "Specific dietary recommendation 4"
      ],
      "workout": [
        "Specific exercise/activity recommendation 1",
        "Specific exercise/activity recommendation 2",
        "Specific exercise/activity recommendation 3"
      ],
      "precautions": [
        "Important precaution 1",
        "Important precaution 2",
        "Important precaution 3",
        "Important precaution 4"
      ],
      "when_to_see_doctor": {
        "urgency": "emergency/urgent/soon/routine",
        "urgency_label": "Go to ER Now / See Doctor Today / See Doctor This Week / Routine Checkup",
        "red_flags": [
          "Red flag symptom that needs immediate attention 1",
          "Red flag symptom that needs immediate attention 2"
        ],
        "time_frame": "How soon to see a doctor and why"
      }
    },
    {
      "rank": 2,
      "disease": "Disease Name",
      "confidence": 65,
      ...same structure...
    },
    {
      "rank": 3,
      "disease": "Disease Name",
      "confidence": 45,
      ...same structure...
    }
  ],
  "overall_urgency": "emergency/urgent/soon/routine",
  "immediate_advice": "One most important thing the patient should do right now"
}

Confidence scoring rules:
- 80-95%: Symptoms strongly and specifically match this disease
- 60-79%: Symptoms moderately match, some atypical features
- 40-59%: Possible but other conditions more likely
- Never give 100% — always some diagnostic uncertainty
- Ranks must be in descending confidence order

Always base recommendations on current medical guidelines. Be specific, not generic."""


class AgenticDiseasePredictionAgent:
    """
    Pure LLM-based Disease Prediction Agent.
    
    ML model ki jagah yeh agent use karo.
    CSV files ki koi zaroorat nahi — LLM khud sab jaanta hai.
    
    Features:
    - Top 3 differential diagnoses with confidence %
    - Complete treatment plan per disease (meds, diet, workout, precautions)
    - "When to See Doctor" urgency per disease
    - Natural language input (Hindi/English/mixed)
    - Clinical reasoning explanation
    """

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def predict(self, patient_input: str, image_description: str = None) -> dict:
        """
        Patient input se Top 3 diseases predict karo with full details.
        
        Args:
            patient_input: Patient ki complaint (any language)
            image_description: Optional — vision agent se image findings
            
        Returns:
            Complete prediction dict with top 3 diseases + full treatment info
        """
        # Context build karo
        user_message = f"Patient complaint: {patient_input}"
        if image_description:
            user_message += f"\n\nVisual findings from image analysis: {image_description}"

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": DISEASE_AGENT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.15,  # Low temperature = consistent medical reasoning
                max_tokens=3000    # Enough for 3 complete disease profiles
            )

            raw = response.choices[0].message.content.strip()
            # Clean any accidental markdown
            raw = raw.replace("```json", "").replace("```", "").strip()

            result = json.loads(raw)
            result["success"] = True
            result["source"] = "Agentic AI (LLaMA 3.3 70B)"
            return result

        except json.JSONDecodeError as e:
            # JSON parse fail — retry with stricter prompt
            return self._retry_prediction(patient_input, str(e))

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "extracted_symptoms": [],
                "top_diseases": [],
                "overall_urgency": "soon",
                "immediate_advice": "Please consult a doctor for proper evaluation."
            }

    def _retry_prediction(self, patient_input: str, error: str) -> dict:
        """
        JSON parse fail hone pe simplified retry.
        """
        try:
            retry_prompt = """Analyze these symptoms and respond with ONLY valid JSON. 
No text before or after. Start directly with { and end with }.
Keep it simple — just top 3 diseases with basic info."""

            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": retry_prompt},
                    {"role": "user", "content": f"""Patient: {patient_input}
                    
Respond in this exact JSON format:
{{
  "extracted_symptoms": ["symptom1", "symptom2"],
  "symptom_summary": "brief summary",
  "top_diseases": [
    {{
      "rank": 1,
      "disease": "Disease Name",
      "confidence": 80,
      "confidence_label": "High",
      "reasoning": "reason",
      "description": "description",
      "medications": [{{"name": "med", "type": "OTC", "purpose": "use"}}],
      "diet": ["diet1", "diet2"],
      "workout": ["exercise1"],
      "precautions": ["precaution1"],
      "when_to_see_doctor": {{
        "urgency": "soon",
        "urgency_label": "See Doctor This Week",
        "red_flags": ["flag1"],
        "time_frame": "within a week"
      }}
    }}
  ],
  "overall_urgency": "soon",
  "immediate_advice": "advice"
}}"""}
                ],
                temperature=0.1,
                max_tokens=2000
            )

            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(raw)
            result["success"] = True
            result["source"] = "Agentic AI (retry)"
            return result

        except Exception as e2:
            return {
                "success": False,
                "error": f"Both attempts failed: {str(e2)}",
                "extracted_symptoms": [],
                "top_diseases": [],
                "overall_urgency": "soon",
                "immediate_advice": "Please consult a qualified doctor for proper diagnosis."
            }

    def get_urgency_color(self, urgency: str) -> str:
        colors = {
            "emergency": "🔴",
            "urgent": "🟠",
            "soon": "🟡",
            "routine": "🟢"
        }
        return colors.get(urgency, "🟡")

    def get_confidence_color(self, confidence: int) -> str:
        if confidence >= 75:
            return "green"
        elif confidence >= 55:
            return "orange"
        else:
            return "red"
