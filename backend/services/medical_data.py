# ============================================================
# MEDICAL DATA SERVICE
# ============================================================
# CSV files se medicines, diet, workout, precautions load karta hai.
# Disease predict hone ke baad yeh data dikhaya jaata hai.
# ============================================================

import os
import pandas as pd


class MedicalDataService:
    """
    Yeh service CSV files se medical data load karti hai.
    Disease name dene pe related medicines, diet, workout return karti hai.
    """

    def __init__(self):
        data_dir = os.path.join(os.path.dirname(__file__), "../../data")

        # Sab CSV files load karo
        try:
            self.description_df = pd.read_csv(os.path.join(data_dir, "description.csv"))
            self.medications_df = pd.read_csv(os.path.join(data_dir, "medications.csv"))
            self.diets_df = pd.read_csv(os.path.join(data_dir, "diets.csv"))
            self.precautions_df = pd.read_csv(os.path.join(data_dir, "precautions_df.csv"))
            self.workout_df = pd.read_csv(os.path.join(data_dir, "workout_df.csv"))
            self.severity_df = pd.read_csv(os.path.join(data_dir, "Symptom-severity.csv"))
            self.loaded = True
        except Exception as e:
            print(f"Medical data load error: {e}")
            self.loaded = False

    def get_disease_info(self, disease_name: str) -> dict:
        """
        Disease name se complete medical information lo.
        Returns dict with description, medications, diet, workout, precautions
        """
        if not self.loaded:
            return self._empty_result(disease_name)

        # numpy.int32 ya koi bhi type ho — pehle string banao
        disease_clean = str(disease_name).strip()

        result = {
            "disease": disease_clean,
            "description": self._get_description(disease_clean),
            "medications": self._get_medications(disease_clean),
            "diet": self._get_diet(disease_clean),
            "workout": self._get_workout(disease_clean),
            "precautions": self._get_precautions(disease_clean),
        }

        return result

    def _get_description(self, disease: str) -> str:
        try:
            mask = self.description_df['Disease'].str.strip() == disease
            rows = self.description_df[mask]
            if not rows.empty:
                return rows.iloc[0]['Description']
            return "Description not available."
        except:
            return "Description not available."

    def _get_medications(self, disease: str) -> list[str]:
        try:
            mask = self.medications_df['Disease'].str.strip() == disease
            meds = self.medications_df[mask]['Medication'].tolist()
            return [str(m) for m in meds if str(m) != 'nan'][:6]
        except:
            return []

    def _get_diet(self, disease: str) -> list[str]:
        try:
            mask = self.diets_df['Disease'].str.strip() == disease
            diets = self.diets_df[mask]['Diet'].tolist()
            return [str(d) for d in diets if str(d) != 'nan'][:6]
        except:
            return []

    def _get_workout(self, disease: str) -> list[str]:
        try:
            mask = self.workout_df['disease'].str.strip() == disease
            workouts = self.workout_df[mask]['workout'].tolist()
            return [str(w) for w in workouts if str(w) != 'nan'][:5]
        except:
            return []

    def _get_precautions(self, disease: str) -> list[str]:
        try:
            mask = self.precautions_df['Disease'].str.strip() == disease
            rows = self.precautions_df[mask]
            if not rows.empty:
                row = rows.iloc[0]
                precs = []
                for col in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']:
                    if col in row and str(row[col]) not in ['nan', '']:
                        precs.append(str(row[col]))
                return precs
            return []
        except:
            return []

    def _empty_result(self, disease: str) -> dict:
        return {
            "disease": disease,
            "description": "Data not available",
            "medications": [],
            "diet": [],
            "workout": [],
            "precautions": []
        }


# Singleton instance — ek baar load karo
medical_data = MedicalDataService()
