import os
from src.model import IntentClassifier
import pickle

def train_model():
    base_dir = os.path.dirname(__file__)  # Directory of train.py
    intents_path = os.path.join(base_dir, 'data', 'intents.json')
    classifier = IntentClassifier()
    classifier.train(intents_path)
    return classifier

if __name__ == "__main__":
    classifier = train_model()
    # Save the model
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(classifier, f)
    print(f"Model saved to {model_path}")