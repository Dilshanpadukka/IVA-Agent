import random
from src.utils import load_intents

class ResponseGenerator:
    def __init__(self, intents_file):
        self.intents = load_intents(intents_file)

    def get_response(self, intent_tag):
        for intent in self.intents['intents']:
            if intent['tag'] == intent_tag:
                return random.choice(intent['responses'])
        return "I’m not sure how to respond to that!"