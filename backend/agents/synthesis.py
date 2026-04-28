# ============================================================
# SYNTHESIS AGENT
# ============================================================
# Yeh agent sabse akhir mein chalta hai.
# Sab specialists ke reports collect karke:
#   1. Ek unified final diagnosis banata hai
#   2. Confidence score calculate karta hai
#   3. Action plan create karta hai
#   4. Safety disclaimer add karta hai
#
# CONCEPT: Jaise hospital mein final MDT (Multi-Disciplinary Team)
# meeting hoti hai jahan sab doctors milke final decision lete hain.
# ============================================================

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

SYNTHESIS_SYSTEM_PROMPT = """You are a Senior Medical Synthesis AI. You receive analysis reports from multiple specialist doctors and create a comprehensive, unified final medical assessment.

Your job:
1. Synthesize all specialist findings
2. Identify the most likely diagnosis
3. Create a prioritized action plan
4. Provide a confidence assessment

IMPORTANT RULES:
- Always be clear this is AI assistance, not a replacement for real doctors
- Highlight emergency signs if any specialist flagged them
- Provide practical, actionable recommendations
- Structure your response clearly with sections

Response Format (STRICTLY follow):

## 🩺 FINAL INTEGRATED DIAGNOSIS

**Most Likely Condition(s):**
[Top 1-2 conditions based on all specialist inputs]

**Diagnostic Confidence:** [High (>80%) / Medium (50-80%) / Low (<50%)]
*Reasoning: [Why this confidence level]*

---

## 📋 SPECIALIST CONSENSUS

[2-3 sentences summarizing what the specialists agreed on]

---

## 🎯 PRIORITY ACTION PLAN

**Immediate Steps:**
1. [Action 1]
2. [Action 2]

**Investigations Recommended:**
- [Test 1 — reason]
- [Test 2 — reason]

**Medications to Consider:**
- [Medicine — for what symptom]

---

## 🥗 LIFESTYLE RECOMMENDATIONS

- **Diet:** [Specific dietary advice]
- **Exercise:** [Activity recommendations]  
- **Sleep:** [Sleep hygiene tips if relevant]
- **Stress:** [Stress management if relevant]

---

## ⚠️ WARNING SIGNS — See Doctor Immediately If:
- [Red flag 1]
- [Red flag 2]
- [Red flag 3]

---

*⚕️ MEDICAL DISCLAIMER: This AI analysis is for informational purposes only and does NOT replace professional medical advice. Please consult a qualified doctor for proper diagnosis and treatment.*
"""


class SynthesisAgent:
    """
    Final synthesis karta hai:
    - Sab specialist reports input mein leta hai
    - Ek comprehensive final answer banata hai
    - Confidence score calculate karta hai
    """

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def synthesize(self, patient_input: str, specialist_reports: list[dict],
                   ml_prediction: dict = None) -> str:
        """
        Sab inputs combine karke final diagnosis banao.

        Args:
            patient_input: Original patient complaint
            specialist_reports: List of dicts from each specialist agent
            ml_prediction: Optional ML model output (disease name + data)

        Returns:
            Final synthesized medical report as string
        """

        # Specialist reports ko format karo
        reports_text = ""
        for report in specialist_reports:
            if report.get("status") == "success":
                reports_text += f"\n\n### {report['agent']} ({report['specialty']}):\n"
                reports_text += report["analysis"]

        # ML prediction ko include karo agar available hai
        ml_text = ""
        if ml_prediction and ml_prediction.get("disease"):
            ml_text = f"""
### ML Model Prediction (Pattern Matching):
- Predicted Disease: {ml_prediction['disease']}
- Matching Symptoms Found: {', '.join(ml_prediction.get('matched_symptoms', []))}
- Note: ML prediction is based on symptom pattern matching from training data
"""

        # Full context banao
        full_context = f"""
PATIENT'S COMPLAINT:
{patient_input}

{ml_text}

SPECIALIST REPORTS:
{reports_text}

Please synthesize all the above information into a final comprehensive medical assessment.
"""

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                    {"role": "user", "content": full_context}
                ],
                temperature=0.2,
                max_tokens=1200
            )
            return response.choices[0].message.content

        except Exception as e:
            return f"""## ⚠️ Synthesis Error

Unable to generate complete synthesis: {str(e)}

**Individual specialist reports are available below.**

*Please consult a qualified doctor for proper medical advice.*"""
