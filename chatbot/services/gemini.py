#!/usr/bin/env python3
"""
Standalone Chatbot Script
Uses Gemini AI to classify user queries based on intent.json
Determines if query is FAQ-based or SQL-related
"""

import logging
from chatbot.data.constants import Message, Error, Success
from chatbot.helpers.gemini_model import GeminiBot
logger = logging.getLogger(__name__)


class AIAssistant(GeminiBot):

    def __init__(self):
        super().__init__()
        # self.chatbot = GeminiBot()

    def run(self):
        """Function to run the chatbot interactively"""
        print(Message.AI_BOT.value)
        print(Message.QUIT_INSTRUCTION.value)
        try:
            if not self.test_connection():
                print(Error.GEMINI_CONNECTION.value)
                return

            print(Success.GEMINI_CONNECTED.value)


            while True:
                user_input = input(Message.USER_INPUT.value).strip()

                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print(Message.EXIT_MESSAGE.value)
                    break

                if not user_input:
                    continue

                result = self.process_query(user_input)
                print(f"Bot: {result.get('response')}")
        except Exception as e:
            logger.error(f"Error in main: {str(e)}")
            print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    chatbot = AIAssistant()
    chatbot.run()
