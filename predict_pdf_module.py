# predict_pdf_module.py

import PyPDF2
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Make sure stopwords are available
nltk.download("stopwords", quiet=True)

# --- Simple Training (Demo) ---
# Replace this with your trained model if you already have one
train_texts = [
    "Congratulations you have won a prize money claim now",
    "Free lottery offer click here",
    "Meeting with client tomorrow regarding project",
    "Reminder for IT/GST filing form submission",
]
train_labels = ["spam", "spam", "ham", "ham"]

vectorizer = TfidfVectorizer(stop_words=stopwords.words("english"))
X_train = vectorizer.fit_transform(train_texts)

model = LogisticRegression()
model.fit(X_train, train_labels)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def predict_pdf(pdf_path: str):
    """Predict if a PDF is SPAM or HAM."""
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        return {"error": "No text could be extracted from the PDF"}

    X_input = vectorizer.transform([text])
    label = model.predict(X_input)[0]

    preview = text[:500].replace("\n", " ") + ("..." if len(text) > 500 else "")
    return {
        "label": label,
        "text_length": len(text),
        "preview": preview,
    }