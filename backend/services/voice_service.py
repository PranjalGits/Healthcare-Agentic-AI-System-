# ============================================================
# VOICE SERVICE
# ============================================================
# Groq Whisper se voice → text (Speech to Text)
# gTTS se text → voice (Text to Speech)
# ============================================================

import os
import tempfile
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def transcribe_audio(audio_file_path: str) -> dict:
    """
    Audio file ko text mein convert karo using Groq Whisper.
    Whisper automatically language detect karta hai (Hindi/English/mixed sab chalega).
    
    Returns: dict with transcription text and detected language
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    try:
        with open(audio_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_file_path), audio_file.read()),
                model="whisper-large-v3",
                response_format="verbose_json",  # Language info bhi milegi
                temperature=0.0
            )

        return {
            "text": transcription.text,
            "language": getattr(transcription, "language", "unknown"),
            "success": True
        }

    except Exception as e:
        return {
            "text": "",
            "language": "unknown",
            "success": False,
            "error": str(e)
        }


def text_to_speech(text: str, output_path: str = None) -> str:
    """
    Text ko audio mein convert karo using gTTS.
    Doctor ki voice mein response sunao.
    
    Returns: path to generated audio file
    """
    try:
        from gtts import gTTS

        # Output path decide karo
        if not output_path:
            output_path = os.path.join(
                os.path.dirname(__file__), "../../results/doctor_response.mp3"
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Text clean karo — markdown symbols remove karo
        clean_text = text
        for symbol in ["##", "**", "*", "---", "🩺", "📋", "🎯", "🥗", "⚠️", "⚕️", "#"]:
            clean_text = clean_text.replace(symbol, "")
        clean_text = clean_text.strip()

        # 600 words tak limit karo (avoid very long audio)
        words = clean_text.split()
        if len(words) > 500:
            clean_text = " ".join(words[:500]) + "... Please read the full report on screen."

        tts = gTTS(text=clean_text, lang="en", slow=False)
        tts.save(output_path)

        return output_path

    except Exception as e:
        print(f"TTS Error: {e}")
        return None
