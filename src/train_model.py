import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from preprocess import preprocess_data
import joblib

# Загрузка и предобработка данных осторжная
filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/smsspamcollection.txt'))
df = preprocess_data(filepath)

X = df['clean_message']
y = df['label']

# Разделяем на train/test для проверки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# TF-IDF векторизация с биграммами и ограничением по частоте слов для более высокой точности
vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# === Naive Bayes === первое улучшение
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
y_pred_nb = nb_model.predict(X_test_vec)

# === Logistic Regression === второе улучшение
lr_model = LogisticRegression(max_iter=500, class_weight='balanced')
lr_model.fit(X_train_vec, y_train)
y_pred_lr = lr_model.predict(X_test_vec)

# Функция для оценки насколько модель точна
def evaluate_model(y_true, y_pred, model_name):
    print(f"=== {model_name} ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

# Оценка моделей
evaluate_model(y_test, y_pred_nb, "Naive Bayes")
evaluate_model(y_test, y_pred_lr, "Logistic Regression")

# Сохраняем лучшую модель которая выигрывает 
best_model = lr_model  # допустим LR лучше по метрикам
joblib.dump(best_model, os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/spam_model.pkl')))
joblib.dump(vectorizer, os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/vectorizer.pkl')))
print("Модель и векторизатор сохранены.")
