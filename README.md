# 🏥 Advanced Agentic AI Healthcare System

## Project Overview
A multi-agent AI healthcare system powered by 8 specialist AI doctors, built using LLaMA 3.3 70B via Groq API. This system mimics a real hospital's Multi-Disciplinary Team (MDT) approach where multiple specialists collaborate on a diagnosis.

## Architecture

```
User Input (Text/Voice/Image/Report)
         ↓
Master Orchestrator Agent
(Decides which specialists to call)
         ↓
NLP Symptom Extractor + 8 Parallel Specialist Agents
         ↓
Synthesis Agent (Final unified diagnosis)
         ↓
Structured Output + Voice Response
```

## 8 Specialist AI Agents
1. **General Physician** — Primary diagnosis, first-line treatment
2. **Cardiologist** — Heart, BP, chest pain, ECG analysis
3. **Neurologist** — Brain, nerves, headache, dizziness, seizures
4. **Pulmonologist** — Lungs, breathing, asthma, COPD
5. **Psychologist** — Mental health, anxiety, depression, stress
6. **Gastroenterologist** — Stomach, liver, digestion, jaundice
7. **Dermatologist** — Skin conditions, rash, acne, image analysis
8. **Endocrinologist** — Diabetes, thyroid, hormones, metabolism

## Key Features
- **Parallel Agent Execution** — All specialists analyze simultaneously
- **NLP Symptom Extraction** — Natural language input (Hindi/English)
- **Real Vision Analysis** — Actual image analysis (not fake)
- **Confidence Scoring** — ML predictions with percentage confidence
- **Chat Memory** — Full conversation history maintained
- **Voice I/O** — Speak symptoms, hear diagnosis
- **Medical Safety** — Disclaimer always shown, emergency alerts

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Key
Create `.env` file in project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Run the application
```bash
streamlit run app.py
```

## Project Structure
```
healthcare_ai/
├── app.py                          # Main Streamlit application
├── .env                            # API keys (never commit this!)
├── requirements.txt
├── data/                           # CSV medical databases
│   ├── description.csv
│   ├── medications.csv
│   ├── diets.csv
│   ├── precautions_df.csv
│   ├── workout_df.csv
│   └── Symptom-severity.csv
├── backend/
│   ├── agents/
│   │   ├── specialists.py          # 8 Specialist Agent classes
│   │   ├── orchestrator.py         # Master Orchestrator Agent
│   │   └── synthesis.py            # Synthesis Agent
│   ├── services/
│   │   ├── symptom_extractor.py    # NLP symptom extraction + ML prediction
│   │   ├── medical_data.py         # CSV data loader service
│   │   ├── voice_service.py        # STT + TTS
│   │   └── vision_service.py       # Real image analysis
│   └── models/
│       └── svc.pkl                 # Trained ML model (41 diseases, 132 symptoms)
└── results/                        # Generated audio files
```

## Technology Stack
| Component | Technology |
|-----------|-----------|
| Frontend | Streamlit |
| LLM | LLaMA 3.3 70B (Groq) |
| Speech-to-Text | Groq Whisper Large V3 |
| Vision | LLaMA 4 Scout (multimodal) |
| ML Model | Support Vector Classifier (SVC) |
| NLP | LLM-based symptom extraction |
| TTS | Google Text-to-Speech (gTTS) |

## Disclaimer
This system is for educational and informational purposes only.
It does NOT replace professional medical advice.
Always consult a qualified healthcare provider.
