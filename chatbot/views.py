import json
import ollama
from langchain_core.prompts import ChatPromptTemplate


with open("data/intents.json", "r", encoding="utf-8") as f:
    intent_definitions = json.load(f)

intents_block = json.dumps(intent_definitions, indent=2)

def get_chatbot_response(query: str):
    intent_prompt = f"""
    Answer questions by classifying them into intents.
    INTENTS = {intents_block}

    User Query: {query}

    ANSWER:
    """

    response = ollama.chat(model='gemma3',
                           messages=[{'role': 'user', 'content': intent_prompt}]
                           )

    print(response['message']['content'])

get_chatbot_response("How can i add offer percentage?")