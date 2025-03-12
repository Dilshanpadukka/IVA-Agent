import pickle
import os
from groq import Groq
from src.model import IntentClassifier
from src.responses import ResponseGenerator
from src.utils import load_intents, preprocess_text
import json
import subprocess
from dotenv import load_dotenv

load_dotenv()
# Initialize Grok client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def load_model(model_file):
    with open(model_file, 'rb') as f:
        return pickle.load(f)

def save_model(classifier, model_file):
    with open(model_file, 'wb') as f:
        pickle.dump(classifier, f)

def update_intents(intents_file, query, response):
    """Add new query and response to intents.json under general_healthcare."""
    intents = load_intents(intents_file)
    general_intent = next((intent for intent in intents['intents'] if intent['tag'] == 'general_healthcare'), None)
    if general_intent is None:
        general_intent = {'tag': 'general_healthcare', 'patterns': [], 'responses': []}
        intents['intents'].append(general_intent)
    general_intent['patterns'].append(query)
    general_intent['responses'].append(response)
    with open(intents_file, 'w') as f:
        json.dump(intents, f, indent=4)

def retrain_model(intents_file, model_file):
    """Retrain the classifier and save the updated model."""
    classifier = IntentClassifier()
    classifier.train(intents_file)
    save_model(classifier, model_file)
    return classifier

def get_llm_response(query):
    """Get response from LLM, restricted to healthcare domain."""
    prompt = (
        "You are an assistant specialized in healthcare. If the following query is related to healthcare, "
        "provide a helpful response using maximum 50 words. If not, respond with 'I can't help with this. Please check with your sector agent.'\n\n"
        f"Query: {query}"
    )
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content

def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(base_dir, 'model.pkl')
    intents_path = os.path.join(base_dir, 'data', 'intents.json')

    # Load or train the initial model
    try:
        classifier = load_model(model_path)
    except FileNotFoundError:
        classifier = IntentClassifier()
        classifier.train(intents_path)
        save_model(classifier, model_path)

    response_gen = ResponseGenerator(intents_path)

    print("IVA Agent: Hello! How can I assist you today? (Type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("IVA Agent: Goodbye!")
            break

        # Try IVA classifier first
        intent = classifier.predict(user_input)
        if intent != "unknown":
            response = response_gen.get_response(intent)
        else:
            # Fallback to LLM
            llm_response = get_llm_response(user_input)
            if llm_response != "I can't help with this. Please check with your sector agent.":
                response = llm_response
                # Update intents and retrain
                update_intents(intents_path, user_input, llm_response)
                classifier = retrain_model(intents_path, model_path)
            else:
                response = llm_response

        print(f"IVA Agent: {response}")

if __name__ == "__main__":
    main()