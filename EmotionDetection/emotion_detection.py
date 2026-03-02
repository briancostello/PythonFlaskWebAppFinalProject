import requests 
import json

def emotion_detector(text_to_analyse): 
    # URL of the emotion detector service 
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'

    # Constructing the request payload in the expected format 
    myobj = { "raw_document": { "text": text_to_analyse } }

    # Custom header specifying the model ID for the sentiment analysis service 
    header = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}

    # Set the headers required for the API request 
    response = requests.post(url, json = myobj, headers=header) 
    
    # Send a POST request to the API with the text and headers 
    return response.text # Return the response text from the API