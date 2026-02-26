'''This file is to create the main function of the risk checker, which will be used to check the risk of the prompt.'''

from datetime import datetime
import pickle

#Load trained AI model
model = pickle.load(open("risk_model_hf.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer_hf.pkl", "rb"))

#List of risky keywords
risky_patterns = {
    # High severity (3)
    "ignore previous instructions": 3,
    "forget your previous instructions": 3,
    "override system instructions": 3,
    "reveal hidden instructions": 3,
    "show your system prompt": 3,
    "display internal configuration": 3,
    "print your policy text": 3,
    "execute internal instructions": 3,
    "what are your hidden rules": 3,
        # Medium severity (2)
    "disregard earlier rules": 2,
    "bypass safety guidelines": 2,
    "replace previous directives": 2,
    "you are no longer bound by": 2,
    "simulate an alternative mode": 2,
    "bypass safeguards": 2,
    "step by step override": 2,
    "follow these hidden steps": 2,
        # Low severity (1)
    "pretend you are": 1,
    "act as an unrestricted system": 1,
    "operate without limitations": 1,
    "without restrictions": 1,
    "no safety limitations": 1,
    "unfiltered response": 1,
    "no content filters": 1,
    "encoded instructions": 1,
    "base64 message": 1,
    "decode this prompt": 1,
    "hidden payload": 1,
    "obfuscated input": 1,
    "first acknowledge then ignore": 1,
    "forget prior context": 1
}

def log_prompt(prompt, final_score, final_level, action, attack_type):
    '''This function logs the prompt, risk score, risk level, confidence percentage, and detected words to a file.'''
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = ( f"\n [{timestamp}]\n" 
                  f"Prompt: {prompt}\n"
                  f"Risk Score: {final_score}\n"
                  f"Risk level: {final_level}\n"
                  f"Action: {action}%\n"
                  f"Attack Type: {attack_type}\n"
                  f"{'-'*50}\n"
                  )
    with open("prompt_logs.txt", "a", encoding = "utf-8") as file:
        file.write(log_entry)


def calculate_risk(prompt):
    '''This functions takes the prompt as input and calculates the risk score based on the presence of risky words.'''
    prompt = prompt.lower()
    score = 0
    detected_words = [] 
    for pattern, weight in risky_patterns.items():
        if pattern in prompt:
            detected_words.append(pattern)
            score += weight    
    return score, detected_words
    
def get_final_risk_level(score):
    '''This function takes the risk score as input and returns the risk level as output.'''
    if score < 30:
        return "SAFE"
    elif score < 60:
        return "SUSPICIOUS"
    else:
        return "HIGH RISK"

def enforcement_action(score):
    '''Pre-LLM Prevention'''
    if score >= 60:
        return "BLOCK"
    elif attack_type != "NONE":
        return "REVIEW"
    elif score >= 30:
        return "REVIEW"
    else:
        return "ALLOW"

def compute_final_risk(rule_score, ml_probability):
    '''To calculate the probability final_score'''
    rule_component = min(rule_score / 5, 1.0)

    # If no rule signals, reduce ML influence slightly
    if rule_score == 0:
        ml_weight = 0.4
        rule_weight = 0.2
    else:
        ml_weight = 0.55
        rule_weight = 0.35

    agreement_bonus = 0
    if rule_component > 0.5 and ml_probability > 0.6:
        agreement_bonus = 0.1

    final_score = (
        0.35 * rule_component +
        0.55 * ml_probability +
        agreement_bonus
    )

    return round(min(final_score, 1.0) * 100, 2)

def normalize_rule_score(score):
    max_possible = 10  # adjustable calibration
    return min(score / max_possible, 1.0)

def ml_predict_proba(prompt):
    """Predict risk using trained ML model. """
    text_vector = vectorizer.transform([prompt])
    probability = model.predict_proba(text_vector)[0][1]
    return probability


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
user_prompt = input("Enter a prompt: ")

rule_score, detected_words = calculate_risk(user_prompt)

ml_probability = ml_predict_proba(user_prompt)

final_score = compute_final_risk(rule_score, ml_probability)
final_level = get_final_risk_level(final_score)
action = enforcement_action(final_score)

attack_type = detect_attack_type(detected_words, ml_probability)

print("\nDetected Patterns:", detected_words)
print("Rule Score:", rule_score)
print("ML Injection Probability:", round(ml_probability * 100, 2), "%")
print("FINAL RISK SCORE:", final_score)
print("FINAL RISK LEVEL:", final_level)
print("ENFORCEMENT ACTION:", action)
print("ATTACK TYPE:", attack_type)

log_prompt(user_prompt, final_score, final_level, action, attack_type)
