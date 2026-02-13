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

# Примеры для определения 
# Примеры для определения (более сложные)
messages = [
    "Congratulations! You've won a $1000 gift card. Call now!",
    "Hey, are we still meeting for lunch today?",
    "URGENT! Your account has been compromised. Reply immediately.",
    "Win a free vacation to Bahamas! Text WIN to 12345.",
    "Can you send me the report by 5pm?",
    "Get cheap meds online without prescription, limited offer!",
    "LOL 😂 that meme you sent was hilarious!",
    "You have an unpaid invoice. Pay immediately to avoid penalties.",
    "Reminder: meeting with team tomorrow at 10am.",
    "Exclusive deal for you: 50% off on all items today!",
    "Are we still on for the movie night tonight?",
    "Your phone number won a lottery! Claim prize now.",
    "Hey, did you finish the homework?",
    "Congratulations! Your credit card has been approved instantly.",
    "Don't miss out! Earn $500 per day working from home."
]

for msg in messages:
    print(f"Message: {msg}\nPrediction: {predict_message(msg)}\n")
