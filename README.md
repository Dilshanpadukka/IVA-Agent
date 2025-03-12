# IVA-Agent: Healthcare-Focused Intelligent Virtual Assistant

IVA-Agent is an Intelligent Virtual Assistant (IVA) designed for the healthcare sector. It combines a lightweight intent classification model with the power of a large language model (LLM) from Groq ("llama-3.3-70b-versatile") to provide accurate, domain-specific responses. The assistant can handle predefined intents, dynamically learn from new healthcare-related queries, and restrict responses to the healthcare domain.

## Features
- **Intent Classification**: Uses a `LogisticRegression` model trained on predefined intents from `intents.json`.
- **LLM Integration**: Falls back to Groq's "llama-3.3-70b-versatile" for unrecognized queries, ensuring robust healthcare responses.
- **Domain Restriction**: Limits responses to healthcare topics; non-healthcare queries receive a fallback message.
- **Dynamic Learning**: Updates `intents.json` with new healthcare queries and retrains the model automatically.
- **Command-Line Interface**: Simple text-based interaction for testing and development.

## Project Structure
```
IVA-Agent/
├── data/
│   └── intents.json      # Intent definitions and responses
├── src/
│   ├── model.py          # Intent classification logic
│   ├── responses.py      # Response generation logic
│   ├── utils.py          # Helper functions
│   └── main.py           # Main application with LLM integration
├── train.py              # Script to train the initial model
├── model.pkl             # Pre-trained model (generated after training)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation (this file)
```

## Prerequisites
- Python 3.8+
- A Groq API key (set as an environment variable: `GROQ_API_KEY`)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd IVA-Agent
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK Data**:
   Run the following Python commands:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   ```

5. **Set the Groq API Key**:
   ```bash
   export GROQ_API_KEY=your_groq_api_key_here  # On Windows: set GROQ_API_KEY=your_groq_api_key_here
   ```

## Usage
1. **Train the Model** (if `model.pkl` doesn’t exist):
   ```bash
   python train.py
   ```
   This generates `model.pkl` in the project root.

2. **Run the IVA**:
   ```bash
   python src/main.py
   ```
   The assistant starts and prompts for input:
   ```
   IVA Agent: Hello! How can I assist you today? (Type 'exit' to quit)
   ```

3. **Interact**:
   - Try healthcare queries: "What are symptoms of diabetes?"
   - Try non-healthcare queries: "How to fix my car?"
   - Type "exit" to quit.

### Example Interaction
```
IVA Agent: Hello! How can I assist you today? (Type 'exit' to quit)
You: Hi
IVA Agent: Hello! How can I assist you today?
You: What are symptoms of diabetes?
IVA Agent: Common symptoms of diabetes include increased thirst, frequent urination, fatigue, and blurred vision. Please consult a healthcare professional for a proper diagnosis.
You: How to fix my car?
IVA Agent: I can't help with this. Please check with your sector agent.
You: exit
IVA Agent: Goodbye!
```

## Developer Guide
This section is for developers looking to extend or modify the IVA-Agent.

### Key Components
- **`src/model.py`**: Defines `IntentClassifier` using `LogisticRegression`. Predicts intents with a confidence threshold (0.7); falls back to "unknown" if below.
- **`src/responses.py`**: Manages predefined responses from `intents.json`.
- **`src/utils.py`**: Provides text preprocessing (tokenization, lemmatization) and utility functions.
- **`src/main.py`**: Orchestrates the IVA loop, integrating the classifier, Groq LLM, and dynamic learning.
- **`train.py`**: Trains and saves the initial model.
- **`data/intents.json`**: Stores intents, patterns, and responses; dynamically updated with new healthcare queries.

### Adding New Features
1. **Extend Intents**:
   - Edit `data/intents.json` to add new intent categories (e.g., "appointment_booking").
   - Example:
     ```json
     {
       "tag": "appointment_booking",
       "patterns": ["book an appointment", "schedule a visit"],
       "responses": ["I can help you book an appointment. Please provide a date and time."]
     }
     ```
   - Retrain with `python train.py`.

2. **Integrate APIs**:
   - Modify `get_llm_response` in `main.py` to fetch real-time healthcare data (e.g., from a medical API).
   - Example: Add `requests` to query a healthcare API and blend its response with LLM output.

3. **Improve Intent Recognition**:
   - Replace `LogisticRegression` with a more advanced model (e.g., BERT via `transformers`).
   - Update `model.py` to load and use a pre-trained NLP model.

4. **Add Context Awareness**:
   - Store conversation history in `main.py` (e.g., as a list) and pass it to the LLM prompt for multi-turn dialogues.

5. **Optimize Retraining**:
   - Current retraining happens after every new healthcare query, which may slow performance.
   - Batch updates by collecting new intents in a temporary list and retraining periodically (e.g., every 10 updates).

### Code Example: Adding a New Intent via Code
To programmatically add an intent without editing `intents.json` manually:
```python
from src.utils import load_intents
import json

intents_path = "data/intents.json"
new_intent = {
    "tag": "medication_info",
    "patterns": ["what is aspirin", "tell me about ibuprofen"],
    "responses": ["Aspirin is a pain reliever and anti-inflammatory drug. Consult a doctor for usage."]
}
intents = load_intents(intents_path)
intents["intents"].append(new_intent)
with open(intents_path, 'w') as f:
    json.dump(intents, f, indent=4)
```

Then retrain:
```bash
python train.py
```

### Debugging Tips
- **FileNotFoundError**: Ensure `model.pkl` and `intents.json` exist in the expected paths.
- **AttributeError**: Check input types (e.g., strings vs. lists) passed to functions.
- **LLM Failures**: Verify `GROQ_API_KEY` is set and the Groq API is accessible.

## License
This project is for educational and development purposes. Ensure compliance with Groq's API usage terms and any healthcare regulations if deployed in production.


