#!/usr/bin/env python3
"""
Standalone Chatbot Script
Uses Gemini AI to classify user queries based on intent.json
Determines if query is FAQ-based or SQL-related
"""

import logging
from chatbot.data.constants import Message, Error, Success
from chatbot.helpers.gemini_model import GeminiBot
from chatbot.helpers.config_connector import get_db_connection
logger = logging.getLogger(__name__)


class AIAssistant(GeminiBot):

    def __init__(self):
        super().__init__()
        self.db_connector, self.database_schema = get_db_connection()
        self.chatbot = GeminiBot()

    def run(self):
        """Function to run the chatbot interactively"""
        print(Message.AI_BOT)
        print(Message.QUIT_INSTRUCTION)
        try:
            if not self.chatbot.test_connection():
                print(Error.GEMINI_CONNECTION)
                return

            print(Success.GEMINI_CONNECTED)
            intents_text = self.chatbot.get_intent_descriptions()

            while True:
                user_input = input(Message.USER_INPUT).strip()

                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print(Message.EXIT_MESSAGE)
                    break

                if not user_input:
                    continue

                variables = {
                    "INTENTS_TEXT": intents_text,
                    "DATABASE_SCHEMA": self.database_schema,
                    "USER_QUERY": user_input
                }

                result = self.chatbot.process_query(self.chatbot, user_input, self.database_schema, self.db_connector, result=None,
                                               intents_text=intents_text, variables=variables)
                print(f"Bot: {result.get('response')}")
        except Exception as e:
            logger.error(f"Error in main: {str(e)}")
            print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    chatbot = AIAssistant()
    chatbot.run()
