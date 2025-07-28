import json
import logging
from typing import Dict, Optional
import google.generativeai as genai
from chatbot.data.prompts import initial_prompt, prompt_with_sql_data, prompt_with_sql_error
from chatbot.helpers.config_connector import load_config, load_intents
logger = logging.getLogger(__name__)

class GeminiBot:
    """Simple chatbot that classifies user queries using Gemini AI"""

    def __init__(self):
        self.prompt = initial_prompt
        self.prompt_with_sql_data = prompt_with_sql_data
        self.prompt_with_sql_error = prompt_with_sql_error
        self.config = load_config()
        self.intents_data = load_intents()

        genai.configure(api_key=self.config.get('GEMINI_API_KEY', ''))
        self.model = genai.GenerativeModel(self.config.get('MODEL_NAME', 'gemini-2.5-flash-lite'))

        self.generation_config = genai.types.GenerationConfig(
            temperature=self.config.get('TEMPERATURE', 0.1),
            top_p=self.config.get('TOP_P', 0.8),
            top_k=self.config.get('TOP_K', 40),
            max_output_tokens=self.config.get('MAX_OUTPUT_TOKENS', 1024),
        )

        print("Gemini connected successfully!")

    def get_intent_descriptions(self) -> str:
        """Get intent descriptions for prompt"""
        intent_descriptions = []
        for intent in self.intents_data.get('intents', []):
            description = intent.get('description', '')
            keywords = ', '.join(intent.get('keywords', []))
            intent_descriptions.append(
                f"- {intent['page_id']}: {description} (Keywords: {keywords})"
            )

        intents_text = '\n'.join(intent_descriptions)
        return intents_text

    def classify_query(self, user_query: str, database_schema: Dict) -> Dict:
        try:
            intents_text = self.get_intent_descriptions()
            variables = {
                "INTENTS_TEXT": intents_text,
                "DATABASE_SCHEMA": database_schema,
                "USER_QUERY": user_query
            }
            prompt = initial_prompt.format(**variables)

            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )

            result = json.loads(response.text.strip().replace('```json', '').replace('```', ''))

            total_tokens = response.usage_metadata.total_token_count
            print(f"Total tokens: {total_tokens}")
            print(result)

            if result.get('classification') == 'faq' and result.get('matched_intent_id'):
                matched_intent = self._find_intent_by_id(result['matched_intent_id'])
                if matched_intent:
                    result['intent_data'] = matched_intent

            return result

        except Exception as e:
            logger.error(f"Error in query classification: {str(e)}")
            return {
                'classification': 'sql_query',
                'confidence': 0.5,
                'matched_intent_id': None,
                'reasoning': 'Error in classification, defaulting to SQL query',
                'error': str(e)
            }

    def final_response(self, prompt_with_dta):
        """Generate final response from query results"""

        response = self.model.generate_content(
            prompt_with_dta,
            generation_config=self.generation_config
        )
        return response.text.strip()

    def _find_intent_by_id(self, page_id: str) -> Optional[Dict]:
        """Find intent data by page_id"""
        for intent in self.intents_data.get('intents', []):
            if intent.get('page_id') == page_id:
                return intent
        return None

    def process_query(self, chatbot, user_query, database_schema, db_connector, result, intents_text, variables) -> Dict:
        logger.info(f"Processing query: {user_query}")

        classification_result = self.classify_query(user_query, database_schema)

        response_data = {
            'user_query': user_query,
            'classification': classification_result,
            'response': '',
            'response_type': classification_result.get('classification', 'unknown')
        }

        if classification_result.get('classification') == 'faq':
            response_data['response'] = self._handle_faq_query(classification_result)
        elif classification_result.get('classification') == 'sql_query':
            response_data['response'] = self._handle_sql_query(chatbot, user_query, database_schema, db_connector, variables, classification_result)
        else:
            response_data['response'] = "I'm not sure how to categorize your question. Could you please rephrase it?"

        logger.info(f"Query processed: {classification_result.get('classification')}")
        return response_data

    def _handle_faq_query(self, classification_result: Dict) -> str:
        """Handle FAQ-type queries"""
        intent_data = classification_result.get('intent_data')
        if intent_data:
            response = intent_data.get('response', 'I can help with that topic.')

            # Add quick actions if available
            quick_actions = intent_data.get('quick_actions', [])
            if quick_actions:
                response += "\n\nQuick actions:"
                for action in quick_actions[:2]:  # Limit to 2 actions
                    response += f"\n• {action.get('text', '')}"

            return response
        else:
            return "I found a matching FAQ topic, but I don't have the detailed response available."

    def _handle_sql_query(self, chatbot, user_query, database_schema, db_connector, variables, result) -> str:
        """Handle SQL-related queries"""
        print("Executing query inside _handle_sql_query: ", result['sql_query'])
        if result:
            if result.get('error'):
                variables['ERROR'] = result['error']
                prompt_with_sql_error_ = self.prompt_with_sql_error.format(**variables)
                result = chatbot.process_query(chatbot, user_query, prompt_with_sql_error_, database_schema, db_connector)
                self._handle_sql_query(chatbot, user_query, database_schema, db_connector, variables, result)

        data_ = db_connector.execute_query(query=str(result['sql_query']))
        variables['DATA_'] = data_
        new_prompt = self.prompt_with_sql_data.format(**variables)
        response = chatbot.final_response(new_prompt)
        return response

    def test_connection(self) -> bool:
        """Test Gemini API connection"""
        try:
            response = self.model.generate_content(
                "Respond with 'OK' if you can read this message.",
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    max_output_tokens=10
                )
            )
            return response.text.strip().upper() == 'OK'
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            return False