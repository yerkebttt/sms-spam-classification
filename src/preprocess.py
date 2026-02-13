import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import os

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Preprocess a text message:
    - lowercase
    - remove non-letter characters
    - remove stopwords
    - lemmatize
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)  # keep letters only
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

def preprocess_data(filepath):
    """
    Load SMS dataset, encode labels, and clean messages.
    If file has no headers, add them automatically.
    """
    # Detect if CSV has headers
    with open(filepath, 'r', encoding='latin-1') as f:
        first_line = f.readline()
    
    # If first line starts with 'ham' or 'spam', assume no headers
    if first_line.startswith('ham') or first_line.startswith('spam'):
        df = pd.read_csv(filepath, sep='\t', header=None, encoding='latin-1')
        df.columns = ['label', 'message']
        # Save back as CSV with headers for future use
        csv_path = os.path.splitext(filepath)[0] + '_with_headers.csv'
        df.to_csv(csv_path, index=False)
        print(f"Headers added. CSV saved as: {csv_path}")
    else:
        df = pd.read_csv(filepath, encoding='latin-1')
    
    # Encode labels: ham=0, spam=1
    df['label'] = df['label'].map({'ham':0, 'spam':1})
    
    # Clean messages
    df['clean_message'] = df['message'].apply(clean_text)
    
    return df
