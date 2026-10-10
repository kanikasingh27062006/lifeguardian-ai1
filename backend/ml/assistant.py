"""
LifeGuardian AI — AI Response Assistant
=========================================
A transparent rule-based response engine.  It takes the ML prediction and
the raw activity inputs, and returns a structured human-readable explanation
with a suggested next step.

NO generative AI or paid LLM is involved.  The engine is fully rule-based
and is described honestly in the UI.  Responses distinguish a model
*prediction* from a *confirmed event*.
"""

from typing import Literal

RiskLevel = Literal["low", "medium", "high", "unknown"]


def build_response(
    activity_state: str,
    inactivity_duration: float,
    movement_intensity: float,
    ml_prediction: RiskLevel,
    hour_of_day: int = 12,
    lang: str = "en",
) -> dict:
    """
    Build a bilingual structured response for the given inputs.

    Returns:
        {
            headline       : str,
            explanation    : str,
            suggestion     : str,
            influencing_factors: list[str],
            risk_level     : str,
            disclaimer     : str,
        }
    """
    r = ml_prediction

    # ── factor detection ─────────────────────────────────────────────────
    factors_en = []
    factors_hi = []

    if activity_state == "inactive":
        factors_en.append("Activity state: Inactive (no movement detected)")
        factors_hi.append("गतिविधि अवस्था: निष्क्रिय (कोई हलचल नहीं)")
    elif activity_state == "lying_down":
        factors_en.append("Activity state: Lying down")
        factors_hi.append("गतिविधि अवस्था: लेटे हुए")
    elif activity_state == "walking":
        factors_en.append("Activity state: Walking (positive movement)")
        factors_hi.append("गतिविधि अवस्था: चल रहे हैं (सकारात्मक हलचल)")
    else:
        factors_en.append("Activity state: Sitting")
        factors_hi.append("गतिविधि अवस्था: बैठे हुए")

    if inactivity_duration >= 60:
        factors_en.append(f"Long inactivity: {inactivity_duration:.0f} min (≥ 60 min threshold)")
        factors_hi.append(f"लंबी निष्क्रियता: {inactivity_duration:.0f} मिनट (60 मिनट से अधिक)")
    elif inactivity_duration >= 20:
        factors_en.append(f"Extended inactivity: {inactivity_duration:.0f} min")
        factors_hi.append(f"लंबी निष्क्रियता: {inactivity_duration:.0f} मिनट")
    else:
        factors_en.append(f"Inactivity duration: {inactivity_duration:.0f} min (normal)")
        factors_hi.append(f"निष्क्रियता अवधि: {inactivity_duration:.0f} मिनट (सामान्य)")

    if movement_intensity < 10:
        factors_en.append(f"Very low movement intensity: {movement_intensity:.0f}/100")
        factors_hi.append(f"बहुत कम हलचल तीव्रता: {movement_intensity:.0f}/100")
    elif movement_intensity < 30:
        factors_en.append(f"Low movement intensity: {movement_intensity:.0f}/100")
        factors_hi.append(f"कम हलचल तीव्रता: {movement_intensity:.0f}/100")
    else:
        factors_en.append(f"Movement intensity: {movement_intensity:.0f}/100 (adequate)")
        factors_hi.append(f"हलचल तीव्रता: {movement_intensity:.0f}/100 (पर्याप्त)")

    night_hours = hour_of_day < 6 or hour_of_day >= 22
    if night_hours:
        factors_en.append(f"Time of day: {hour_of_day:02d}:00 (night hours — inactivity is expected)")
        factors_hi.append(f"दिन का समय: {hour_of_day:02d}:00 बजे (रात — निष्क्रियता अपेक्षित)")

    # ── branch responses ─────────────────────────────────────────────────
    RESPONSES = {
        "low": {
            "headline_en":    "Normal Activity Pattern Detected",
            "headline_hi":    "सामान्य गतिविधि पैटर्न पहचाना गया",
            "explanation_en": (
                "The simulated sensor data shows normal activity levels. "
                "Movement intensity and inactivity duration are both within "
                "expected ranges for regular daily activity. "
                "The model's illustrative prediction is: Low Risk."
            ),
            "explanation_hi": (
                "अनुकरणीय सेंसर डेटा सामान्य गतिविधि स्तर दर्शाता है। "
                "हलचल तीव्रता और निष्क्रियता अवधि दोनों सामान्य दैनिक "
                "गतिविधि की अपेक्षित सीमाओं के भीतर हैं। "
                "मॉडल का प्रदर्शनात्मक अनुमान है: कम जोखिम।"
            ),
            "suggestion_en":  (
                "No action required in this demonstration scenario. "
                "Continue normal monitoring."
            ),
            "suggestion_hi":  (
                "इस प्रदर्शन परिदृश्य में कोई कार्रवाई आवश्यक नहीं है। "
                "सामान्य निगरानी जारी रखें।"
            ),
        },
        "medium": {
            "headline_en": "Unusual Inactivity Pattern — Attention Suggested",
            "headline_hi": "असामान्य निष्क्रियता पैटर्न — ध्यान देने की सलाह",
            "explanation_en": (
                "The simulated sensor data shows an unusual pattern: "
                "the combination of activity state, inactivity duration, "
                "and movement intensity suggests the person may have been "
                "inactive for longer than usual. "
                "The model's illustrative prediction is: Medium Risk. "
                "This is a simulated demonstration, not a confirmed event."
            ),
            "explanation_hi": (
                "अनुकरणीय सेंसर डेटा एक असामान्य पैटर्न दर्शाता है: "
                "गतिविधि अवस्था, निष्क्रियता अवधि और हलचल तीव्रता का संयोजन "
                "सुझाता है कि व्यक्ति सामान्य से अधिक समय तक निष्क्रिय रहा होगा। "
                "मॉडल का प्रदर्शनात्मक अनुमान है: मध्यम जोखिम। "
                "यह एक अनुकरणीय प्रदर्शन है, कोई पुष्ट घटना नहीं।"
            ),
            "suggestion_en": (
                "In a real system, a caregiver check-in would be suggested. "
                "For this demonstration: verify the simulated inputs and "
                "consider adjusting them to explore how the model responds."
            ),
            "suggestion_hi": (
                "वास्तविक प्रणाली में, देखभालकर्ता से जाँच की सलाह दी जाती। "
                "इस प्रदर्शन में: अनुकरणीय इनपुट की जाँच करें और "
                "देखें कि मॉडल कैसे प्रतिक्रिया करता है।"
            ),
        },
        "high": {
            "headline_en": "Possible Fall Scenario Indicators Detected",
            "headline_hi": "संभावित गिरावट परिदृश्य के संकेतक पहचाने गए",
            "explanation_en": (
                "The simulated sensor data shows a high-risk pattern: "
                "very low movement intensity combined with prolonged inactivity "
                "or lying-down state matches patterns associated with "
                "a possible fall in the demonstration labelling rules. "
                "The model's illustrative prediction is: High Risk. "
                "IMPORTANT: This is a SIMULATED scenario. No real emergency "
                "services are notified. This model is NOT clinically validated."
            ),
            "explanation_hi": (
                "अनुकरणीय सेंसर डेटा एक उच्च-जोखिम पैटर्न दर्शाता है: "
                "बहुत कम हलचल तीव्रता के साथ लंबे समय तक निष्क्रियता या "
                "लेटे हुए अवस्था प्रदर्शन नियमों में गिरावट के पैटर्न से मेल खाती है। "
                "मॉडल का प्रदर्शनात्मक अनुमान है: अधिक जोखिम। "
                "महत्वपूर्ण: यह एक अनुकरणीय परिदृश्य है। कोई वास्तविक "
                "आपातकालीन सेवा सूचित नहीं की गई। यह मॉडल चिकित्सकीय रूप से मान्य नहीं है।"
            ),
            "suggestion_en": (
                "In a real system, an immediate caregiver alert would be triggered. "
                "For this demonstration: this shows the system's high-risk "
                "detection branch. Adjust inputs to see different model responses."
            ),
            "suggestion_hi": (
                "वास्तविक प्रणाली में, तत्काल देखभालकर्ता अलर्ट भेजा जाता। "
                "इस प्रदर्शन में: यह प्रणाली की उच्च-जोखिम पहचान शाखा दिखाता है। "
                "विभिन्न मॉडल प्रतिक्रियाएँ देखने के लिए इनपुट बदलें।"
            ),
        },
        "unknown": {
            "headline_en":    "Assessment Unavailable",
            "headline_hi":    "मूल्यांकन उपलब्ध नहीं",
            "explanation_en": "The ML model is not ready. Run `python ml/train.py` first.",
            "explanation_hi": "ML मॉडल तैयार नहीं है। पहले `python ml/train.py` चलाएँ।",
            "suggestion_en":  "Train the model and restart the backend server.",
            "suggestion_hi":  "मॉडल को प्रशिक्षित करें और बैकएंड सर्वर पुनः आरंभ करें।",
        },
    }

    branch = RESPONSES.get(r, RESPONSES["unknown"])

    disclaimer_en = (
        "PROTOTYPE DISCLAIMER: This AI Response Assistant is rule-based. "
        "No generative AI or paid LLM is used. Predictions are illustrative "
        "and trained on synthetic data only. Not clinically validated."
    )
    disclaimer_hi = (
        "प्रोटोटाइप अस्वीकरण: यह AI प्रतिक्रिया सहायक नियम-आधारित है। "
        "कोई जनरेटिव AI या सशुल्क LLM उपयोग नहीं किया गया। अनुमान प्रदर्शनात्मक हैं "
        "और केवल अनुकरणीय डेटा पर प्रशिक्षित हैं। चिकित्सकीय रूप से मान्य नहीं।"
    )

    if lang == "hi":
        return {
            "headline":             branch["headline_hi"],
            "explanation":          branch["explanation_hi"],
            "suggestion":           branch["suggestion_hi"],
            "influencing_factors":  factors_hi,
            "risk_level":           r,
            "disclaimer":           disclaimer_hi,
        }
    return {
        "headline":             branch["headline_en"],
        "explanation":          branch["explanation_en"],
        "suggestion":           branch["suggestion_en"],
        "influencing_factors":  factors_en,
        "risk_level":           r,
        "disclaimer":           disclaimer_en,
    }
