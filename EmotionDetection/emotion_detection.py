import requests
import json

def emotion_detector(text_to_analyse):
    """This function calls the Watson EmotionPredict endpoint and extracts the
    emotion scores for anger, disgust, fear, joy, and sadness. It also computes
    the dominant emotion (the emotion with the highest score).
    - If the API response status code is 400, return a dictionary with the same
      keys but all values set to None.
    - If the dominant emotion cannot be determined (e.g., missing/None scores or
      malformed response), return a dictionary with all values set to None.

    Returns:
        dict: Dictionary with keys:
            - 'anger', 'disgust', 'fear', 'joy', 'sadness', 'dominant_emotion'
          Values are floats when successful; otherwise None for all keys.
    """
    url = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    myobj = {"raw_document": {"text": text_to_analyse}}
    header = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}

    def _all_none():
        """Return the required result dictionary with all values set to None."""
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None
        }

    response = requests.post(url, json=myobj, headers=header)

    # Requirement: status_code == 400 => return all None values
    if response.status_code == 400:
        return _all_none()

    # Parse JSON safely; if parsing/structure fails, treat as invalid
    try:
        formatted_response = json.loads(response.text)
        emotions = formatted_response["emotionPredictions"][0]["emotion"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return _all_none()

    # Extract scores using .get in case keys are missing
    anger_score = emotions.get("anger")
    disgust_score = emotions.get("disgust")
    fear_score = emotions.get("fear")
    joy_score = emotions.get("joy")
    sadness_score = emotions.get("sadness")

    # If any score is None, dominant emotion cannot be determined
    scores = [anger_score, disgust_score, fear_score, joy_score, sadness_score]
    if any(s is None for s in scores):
        return _all_none()

    # Compute dominant emotion; if emotions dict is empty or invalid, return all None
    dominant_emotion = max(emotions, key=emotions.get) if emotions else None
    if dominant_emotion is None:
        return _all_none()

    return {
        "anger": anger_score,
        "disgust": disgust_score,
        "fear": fear_score,
        "joy": joy_score,
        "sadness": sadness_score,
        "dominant_emotion": dominant_emotion
    }