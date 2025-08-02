from enum import Enum

class Message(Enum):
    AI_BOT = "=== AI Chatbot ==="
    QUIT_INSTRUCTION = "Type 'quit' to exit.\n"
    USER_INPUT = "You: "
    BOT_RESPONSE = "Bot: "
    EXIT_MESSAGE = "Goodbye!"
    REPHRASE_MESSAGE = "I'm not sure how to categorize your question. Could you please rephrase it?"

class Error(Enum):
    GEMINI_CONNECTION = "Failed to connect to Gemini API. Please check your API key."

class Success(Enum):
    GEMINI_CONNECTED = "Gemini connected successfully!"