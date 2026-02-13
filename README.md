# SMS Spam Detection NLP Project

This project classifies SMS messages as **spam** or **ham** using NLP and machine learning.

## Folder Structure

- `data/` - CSV dataset (`spam.csv`)
- `src/` - Python scripts
    - `preprocess.py` - text preprocessing
    - `train_model.py` - train and evaluate model
    - `predict.py` - classify new messages
- `requirements.txt` - Python dependencies
- `README.md` - project description

## How to Run

1. Install dependencies:
pip install -r requirements.txt


2. Train the model:


python src/train_model.py


3. Predict new messages:


python src/predict.py

✅ Usage

Open the project folder in Visual Studio or VS Code.

Ensure data/spam.csv exists.

Run train_model.py to train and save the model.

Run predict.py to test new messages.