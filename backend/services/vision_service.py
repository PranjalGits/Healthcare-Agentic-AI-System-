# ============================================================
# VISION SERVICE — REAL Image Analysis (Fixed!)
# ============================================================
# Purane project mein image upload hoti thi lekin actually
# analyze nahi hoti thi — sirf text prompt bhejte the.
#
# Iss naye version mein:
#   1. Image actually base64 mein encode hoti hai
#   2. LLM vision model ko REAL image + query bheja jaata hai
#   3. Dermatologist agent bhi image description use karta hai
# ============================================================

import os
import base64
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def encode_image_to_base64(image_path: str) -> str:
    """Image file ko base64 string mein convert karo"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_medical_image(image_path: str, patient_complaint: str = "") -> dict:
    """
    Medical image ko ACTUALLY analyze karo.
    
    Yeh Groq ka vision-capable model use karta hai.
    Image base64 encoded ho ke directly LLM ko jaati hai.
    
    Args:
        image_path: Local path to image file
        patient_complaint: Patient's text description (optional, adds context)
    
    Returns:
        dict with visual_findings, skin_observations, recommendations
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    try:
        # Image encode karo
        encoded_image = encode_image_to_base64(image_path)

        # Image type detect karo
        ext = os.path.splitext(image_path)[1].lower()
        media_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp"
        }
        media_type = media_type_map.get(ext, "image/jpeg")

        # System prompt for medical image analysis
        system_prompt = """You are a medical image analysis AI. Analyze the provided medical image carefully.

Focus on:
1. Visible skin conditions (rash, discoloration, lesions, swelling)
2. Any visible abnormalities
3. Color changes, texture, distribution patterns
4. Severity assessment

Provide a structured clinical description that a doctor can use.
Be specific about what you observe. Do not make definitive diagnoses — describe findings."""

        # User message with context
        user_text = "Please analyze this medical image."
        if patient_complaint:
            user_text = f"Patient complaint: {patient_complaint}\n\nPlease analyze this medical image in context of the complaint."

        # REAL vision API call — image actually bheja ja raha hai
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",  # Vision capable model
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{encoded_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": user_text
                        }
                    ]
                }
            ],
            temperature=0.2,
            max_tokens=600
        )

        visual_findings = response.choices[0].message.content

        return {
            "success": True,
            "visual_findings": visual_findings,
            "image_analyzed": True,
            "model_used": "llama-4-scout (vision)"
        }

    except Exception as e:
        # Fallback: Vision model not available pe text-only analysis
        return {
            "success": False,
            "visual_findings": f"Image analysis unavailable: {str(e)}. Please describe your visual symptoms in text.",
            "image_analyzed": False,
            "error": str(e)
        }
