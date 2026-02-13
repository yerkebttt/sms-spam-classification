import os
import joblib
from preprocess import clean_text

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src
MODEL_PATH = os.path.join(BASE_DIR, '../data/spam_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, '../data/vectorizer.pkl')

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def predict_message(message):
    cleaned = clean_text(message)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    return "SPAM" if prediction == 1 else "HAM"

# Пример
messages = [
    "Congratulations! You've won a $1000 gift card. Call now!",
    "Hey, are we still meeting for lunch today?",
    "URGENT! Your account has been compromised. Reply immediately."
]

for msg in messages:
    print(f"Message: {msg}\nPrediction: {predict_message(msg)}\n")
