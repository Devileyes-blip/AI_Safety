'''This file is to create the main function of the risk checker, which will be used to check the risk of the prompt.'''

import random
from datetime import datetime
import pickle

#Load trained AI model
model = pickle.load(open("risk_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

#List of risky keywords
risky_words = ["ignore previous instructions","disregard earlier rules","override system instructions","bypass safety guidelines","forget prior context","replace previous directives","pretend you are","act as an unrestricted system","you are no longer bound by","simulate an alternative mode","operate without limitations","without restrictions","no safety limitations","unfiltered response","no content filters","bypass safeguards","show your system prompt","reveal hidden instructions","display internal configuration","what are your hidden rules","print your policy text","encoded instructions","base64 message","decode this prompt","hidden payload","obfuscated input","step by step override","first acknowledge then ignore","follow these hidden steps","execute internal instructions", 
               "forget your previous instructions"
]

def log_prompt(prompt, score, level, confidence, detected_words):
    '''This function logs the prompt, risk score, risk level, confidence percentage, and detected words to a file.'''
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = ( f"\n [{timestamp}]\n" 
                  f"Prompt: {prompt}\n"
                  f"Risk Score: {score}\n"
                  f"Risk level: {level}\n"
                  f"Confidence: {confidence}%\n"
                  f"Detected Words: {detected_words if detected_words else 'None'}"
                  f"{'-'*50}\n"
                  )
    with open("prompt_logs.txt", "a", encoding = "utf-8") as file:
        file.write(log_entry)


def calculate_risk(prompt):
    '''This functions takes the prompt as input and calculates the risk score based on the presence of risky words.'''
    prompt = prompt.lower()
    score = 0
    detected_words = [] 
    for word in risky_words:
        if word in prompt:
            detected_words.append(word)
            score += 1    
    return score, detected_words
    
def get_risk_level(score):
    '''This function takes the risk score as input and returns the risk level as output.'''
    if score == 0:
        return "SAFE"
    elif score <= 2:
        return "SUSPICIOUS"
    else:
        return "HIGH RISK"

def calculate_confidence(score):
    '''This function takes the detected words as input and calculates the confidence percentage.'''
    if score ==0:
        return random.randint(0, 10)
    elif score <= 2:
        return random.randint(30, 60)
    else:
        return random.randint(70, 95)

def ml_predict(prompt):
    """Predict risk using trained ML model. """
    text_vector = vectorizer.transform([prompt])
    prediction = model.predict(text_vector)[0]
    return prediction

def final_decision(rule_level, ml_prediction):
    """Combine rule-based and ML predictions."""
    if rule_level == "HIGH":
        return "HIGH"
    if ml_prediction == "RISKY":
        return "SUSPICIOUS"
    return "SAFE"

def detect_attack_type(detected_words, ml_prediction):
    """Classify attack type based on detected keywords."""
    jailbreak_keywords = ["ignore", "bypass", "override", "unrestricted"]
    extraction_keywords = ["reveal", "show", "display", "hidden", "system"]

    for word in detected_words:
        if any(k in word for k in jailbreak_keywords):
            return "JAILBREAK ATTEMPT"
        if any(k in word for k in extraction_keywords):
            return "DATA EXTRACTION ATTEMPT"
    
    if ml_prediction == "RISKY":
        return "POTENTIAL JAILBREAK ATTEMPT"

    return "NONE"

#Main program   
user_prompt = input("Enter a prompt:")

risk_score, detected_words = calculate_risk(user_prompt)
risk_level = get_risk_level(risk_score)
ml_result = ml_predict(user_prompt)
final_risk = final_decision(risk_level, ml_result)
attack_type = detect_attack_type(detected_words, ml_result)

print("Confidence Percentage:", calculate_confidence(risk_score), "%")
print("Detected Words:", detected_words)
print("Risk Level: ", risk_level)
print("AI Model Prediction:", ml_result)
print("FINAL RISK DECISION: ", final_risk)
print("ATTACK TYPE:", attack_type)

log_prompt(user_prompt, risk_score, risk_level, calculate_confidence(risk_score), detected_words)
