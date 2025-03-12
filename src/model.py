from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from src.utils import load_intents, create_training_data, text_to_bow


class IntentClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.classifier = LogisticRegression(max_iter=1000)  # Increased iterations for convergence
        self.words = []
        self.classes = []

    def train(self, intents_file):
        intents = load_intents(intents_file)
        self.words, self.classes, documents = create_training_data(intents)
        X = []
        y = []
        for doc, tag in documents:
            bow = text_to_bow(" ".join(doc), self.words)
            X.append(bow)
            y.append(self.classes.index(tag))

        X = np.array(X)
        y = np.array(y)
        self.classifier.fit(X, y)

    def predict(self, text):
        bow = text_to_bow(text, self.words)
        probabilities = self.classifier.predict_proba([bow])[0]
        max_prob = max(probabilities)
        if max_prob > 0.7:  # Confidence threshold
            prediction = self.classes[probabilities.argmax()]
            return prediction
        else:
            return "unknown"