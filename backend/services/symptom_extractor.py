# ============================================================
# NLP SYMPTOM EXTRACTOR SERVICE
# ============================================================
# Purane project ki sabse badi problem:
#   User "mujhe sar dard hai" likhta tha toh predict nahi hota tha
#   kyunki model sirf "headache" exact word dhundhta tha.
#
# Naya solution:
#   LLM se natural language samjhao aur proper symptom names nikalo.
#   Phir ML model pe pass karo.
#
# Flow:
#   "I have been having severe headache and feel nauseous since morning"
#   -> LLM extracts -> ["headache", "nausea"]
#   -> ML model pe match karo symptoms_dict ke saath
# ============================================================

import os
import json
import numpy as np
import pickle
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# -------------------------------------------------------
# Symptoms dictionary — yeh ML model ke input features hain
# 132 symptoms — index number ke saath
# -------------------------------------------------------
SYMPTOMS_DICT = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3,
    'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8,
    'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12,
    'spotting_urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16,
    'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20,
    'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24,
    'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28,
    'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
    'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36,
    'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40,
    'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44,
    'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48,
    'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51,
    'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55,
    'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58,
    'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61,
    'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65,
    'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69,
    'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72,
    'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75,
    'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78,
    'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82,
    'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85,
    'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88,
    'bladder_discomfort': 89, 'foul_smell_of_urine': 90, 'continuous_feel_of_urine': 91,
    'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look': 94, 'depression': 95,
    'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99,
    'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic_patches': 102,
    'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105,
    'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108,
    'lack_of_concentration': 109, 'visual_disturbances': 110,
    'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112,
    'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115,
    'history_of_alcohol_consumption': 116, 'fluid_overload_1': 117,
    'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120,
    'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123,
    'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126,
    'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129,
    'red_sore_around_nose': 130, 'yellow_crust_ooze': 131
}

# Symptoms list for LLM reference
SYMPTOMS_LIST_TEXT = ", ".join(SYMPTOMS_DICT.keys())

EXTRACTION_SYSTEM_PROMPT = f"""You are a medical symptom extraction AI. Your ONLY job is to extract symptoms from patient descriptions and map them to standardized medical symptom names.

Available standardized symptoms (use ONLY these exact names):
{SYMPTOMS_LIST_TEXT}

Rules:
1. Map patient's words to the closest matching symptom from the list
2. If patient says "sar dard" or "head pain" → headache
3. If patient says "ulti" or "vomiting feeling" → nausea or vomiting
4. If patient says "pet dard" or "stomach hurts" → stomach_pain or abdominal_pain
5. Only include symptoms that are CLEARLY mentioned or implied
6. Return ONLY a JSON array of symptom names. No explanation. No markdown.

Example:
Patient: "I have fever, headache and feel like vomiting"
Response: ["high_fever", "headache", "nausea", "vomiting"]

Patient: "mere pet me dard hai aur ulti ho rahi hai"
Response: ["stomach_pain", "nausea", "vomiting"]
"""


class SymptomExtractorService:
    """
    NLP-based symptom extraction.
    Natural language (any language) → standardized symptom list → ML model input
    """

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        # ML Model load karo
        model_path = os.path.join(os.path.dirname(__file__), "../models/svc.pkl")
        try:
            with open(model_path, "rb") as f:
                self.svc_model = pickle.load(f)
            self.model_loaded = True
        except Exception as e:
            self.svc_model = None
            self.model_loaded = False
            print(f"ML Model load failed: {e}")

    def extract_symptoms_nlp(self, patient_text: str) -> list[str]:
        """
        Natural language se symptoms extract karo using LLM.
        """
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Patient says: {patient_text}"}
                ],
                temperature=0.1,
                max_tokens=200
            )

            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            symptoms = json.loads(raw)

            # Validate — sirf valid symptoms rakhna
            valid_symptoms = [s for s in symptoms if s in SYMPTOMS_DICT]
            return valid_symptoms

        except Exception as e:
            # Fallback: direct text matching
            return self._fallback_extraction(patient_text)

    def _fallback_extraction(self, text: str) -> list[str]:
        """
        LLM fail hone pe simple keyword matching fallback.
        """
        text = text.lower().replace("_", " ")
        found = []
        for symptom_key in SYMPTOMS_DICT.keys():
            readable = symptom_key.replace("_", " ")
            if readable in text:
                found.append(symptom_key)
        return found

    def predict_disease(self, symptoms: list[str]) -> dict:
        """
        Symptoms list se ML model se disease predict karo.
        Returns disease name, matched symptoms, confidence info.
        """
        if not self.model_loaded or not symptoms:
            return {
                "disease": None,
                "matched_symptoms": symptoms,
                "confidence_note": "ML prediction unavailable",
                "error": "Model not loaded or no symptoms"
            }

        # Input vector banao — 132 features, 0 ya 1
        input_vector = np.zeros(len(SYMPTOMS_DICT))
        matched = []

        for symptom in symptoms:
            if symptom in SYMPTOMS_DICT:
                input_vector[SYMPTOMS_DICT[symptom]] = 1
                matched.append(symptom)

        if not matched:
            return {
                "disease": None,
                "matched_symptoms": [],
                "confidence_note": "No symptoms matched the training data",
                "error": "No matching symptoms"
            }

        try:
            # SVC prediction — str() se numpy type ko plain string banao
            prediction = str(self.svc_model.predict([input_vector])[0])

            # Confidence: SVC decision function se approximate karo
            decision = self.svc_model.decision_function([input_vector])
            confidence_pct = self._calculate_confidence(decision, matched)

            return {
                "disease": prediction,
                "matched_symptoms": matched,
                "total_symptoms_entered": len(symptoms),
                "matched_count": len(matched),
                "confidence_percentage": confidence_pct,
                "confidence_note": self._confidence_label(confidence_pct),
                "error": None
            }

        except Exception as e:
            return {
                "disease": None,
                "matched_symptoms": matched,
                "confidence_note": "Prediction failed",
                "error": str(e)
            }

    def _calculate_confidence(self, decision_scores, matched_symptoms: list) -> int:
        """
        Confidence score calculate karo.
        More symptoms matched = higher confidence.
        """
        base = 60
        symptom_bonus = min(len(matched_symptoms) * 5, 30)
        # Decision function magnitude bhi consider karo
        try:
            max_score = float(decision_scores.max())
            score_bonus = min(int(max_score * 3), 10)
        except:
            score_bonus = 0

        return min(base + symptom_bonus + score_bonus, 95)

    def _confidence_label(self, pct: int) -> str:
        if pct >= 80:
            return f"High Confidence ({pct}%)"
        elif pct >= 60:
            return f"Moderate Confidence ({pct}%)"
        else:
            return f"Low Confidence ({pct}%) — Please consult a doctor"

    def full_analysis(self, patient_text: str) -> dict:
        """
        Complete pipeline:
        1. NLP symptom extraction
        2. ML prediction
        3. Return full result
        """
        extracted_symptoms = self.extract_symptoms_nlp(patient_text)
        prediction = self.predict_disease(extracted_symptoms)
        prediction["extracted_symptoms_raw"] = extracted_symptoms
        return prediction
