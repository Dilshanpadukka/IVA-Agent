import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import json
import numpy as np

lemmatizer = WordNetLemmatizer()


def load_intents(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return [lemmatizer.lemmatize(token) for token in tokens]


def create_training_data(intents):
    words = []
    classes = []
    documents = []
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            tokenized_pattern = preprocess_text(pattern)
            words.extend(tokenized_pattern)
            documents.append((tokenized_pattern, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = sorted(list(set(words)))
    classes = sorted(classes)
    return words, classes, documents


def text_to_bow(text, words):
    tokenized_text = preprocess_text(text)
    bag = [0] * len(words)
    for w in tokenized_text:
        if w in words:
            bag[words.index(w)] = 1
    return np.array(bag)