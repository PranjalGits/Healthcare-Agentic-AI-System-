# ============================================================
# MASTER ORCHESTRATOR AGENT
# ============================================================
# Yeh agent sabse pehle chalata hai.
# Patient ka input padh ke decide karta hai:
#   - Kaun se specialist agents call karne hain
#   - Input kis type ka hai (text/voice/image/report)
#   - Kitna urgent hai
#
# CONCEPT: Ek real hospital mein pehle receptionist/triage nurse
# decide karta hai ki patient ko kaun se doctor ke paas bhejna hai.
# Yehi kaam Orchestrator karta hai.
# ============================================================

import os
import json
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

ORCHESTRATOR_SYSTEM_PROMPT = """You are a Medical Triage Orchestrator AI. Your job is to analyze patient input and decide which specialist doctors should examine this case.

Available specialists:
- general_physician: For general symptoms, fever, body pain, weakness, common cold
- cardiologist: For chest pain, palpitations, high BP, heart issues, shortness of breath
- neurologist: For headache, dizziness, numbness, tingling, seizures, memory issues, stroke symptoms
- pulmonologist: For cough, breathing difficulty, wheezing, asthma, lung issues
- psychologist: For anxiety, depression, stress, mood swings, sleep issues, mental health
- gastroenterologist: For stomach pain, nausea, vomiting, diarrhea, constipation, jaundice, liver issues
- dermatologist: For skin rash, itching, acne, hair loss, skin color changes, any skin conditions
- endocrinologist: For diabetes symptoms (excessive thirst/urination), thyroid issues, weight changes, fatigue

Rules:
1. ALWAYS include general_physician
2. Add specialists based on symptom keywords
3. If symptoms are vague or multiple, include 3-4 specialists
4. If clearly specific (e.g. only skin rash), include 2-3 specialists

You MUST respond ONLY with valid JSON. No explanation. No markdown. Just JSON.

Example response:
{"selected_agents": ["general_physician", "cardiologist"], "urgency": "high", "reasoning": "Chest pain with breathlessness needs cardiac evaluation", "input_type": "text"}

Urgency levels: "emergency" (call 911), "high" (see doctor today), "medium" (see doctor this week), "low" (routine checkup)
"""


class MasterOrchestratorAgent:
    """
    Orchestrator ka kaam:
    1. Patient input analyze karo
    2. Relevant specialists select karo
    3. Urgency level determine karo
    4. Structured plan return karo
    """

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def decide(self, patient_input: str, has_image: bool = False, has_report: bool = False) -> dict:
        """
        Patient input se triage plan banao.
        Returns dict with: selected_agents, urgency, reasoning, input_type
        """

        # Input mein context add karo
        context = patient_input
        if has_image:
            context += "\n[Patient has also provided an image for visual analysis]"
        if has_report:
            context += "\n[Patient has also provided a medical report document]"

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": ORCHESTRATOR_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Patient says: {context}"}
                ],
                temperature=0.1,
                max_tokens=300
            )

            raw = response.choices[0].message.content.strip()

            # JSON parse karo
            # Sometimes LLM adds backticks, unhe remove karo
            raw = raw.replace("```json", "").replace("```", "").strip()
            plan = json.loads(raw)

            # Validate karo — general_physician always hona chahiye
            if "general_physician" not in plan.get("selected_agents", []):
                plan["selected_agents"].insert(0, "general_physician")

            # Dermatologist add karo agar image hai
            if has_image and "dermatologist" not in plan["selected_agents"]:
                plan["selected_agents"].append("dermatologist")

            return plan

        except Exception as e:
            # Fallback: safe default plan
            return {
                "selected_agents": ["general_physician", "cardiologist", "neurologist"],
                "urgency": "medium",
                "reasoning": "Default analysis plan due to parsing error",
                "input_type": "text",
                "error": str(e)
            }

    def get_urgency_message(self, urgency: str) -> tuple[str, str]:
        """
        Urgency level ke hisaab se message aur color return karo
        """
        messages = {
            "emergency": (
                "🚨 EMERGENCY — Please call 108 (Ambulance) or go to Emergency Room immediately!",
                "red"
            ),
            "high": (
                "⚠️ HIGH PRIORITY — Please consult a doctor today or visit urgent care.",
                "orange"
            ),
            "medium": (
                "📅 MODERATE — Please schedule a doctor appointment this week.",
                "yellow"
            ),
            "low": (
                "✅ ROUTINE — Monitor symptoms. Consult a doctor for routine checkup.",
                "green"
            )
        }
        return messages.get(urgency, messages["medium"])
