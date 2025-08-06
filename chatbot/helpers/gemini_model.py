import json
import logging
from typing import Dict, Optional
import google.generativeai as genai
from chatbot.data.constants import Message
from chatbot.data.prompts import initial_prompt, prompt_with_sql_data, prompt_with_sql_error, prompt_for_navigation
from chatbot.helpers.config_connector import load_config, load_intents, get_db_connection
logger = logging.getLogger(__name__)

class GeminiBot:
    """Simple chatbot that classifies user queries using Gemini AI"""

    def __init__(self):
        self.prompt = initial_prompt
        self.prompt_with_sql_data = prompt_with_sql_data
        self.prompt_with_sql_error = prompt_with_sql_error
        self.config = load_config()
        self.intents_data = load_intents()
        self.intents_text = self.get_intent_descriptions()
        self.db_connector, self.database_schema = get_db_connection()

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

    def classify_query(self, user_query) -> Dict:
        try:
            variables = {
                "INTENTS_TEXT": self.intents_text,
                "DATABASE_SCHEMA": self.database_schema,
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

    def process_query(self, user_query) -> Dict:
        logger.info(f"Processing query: {user_query}")

        classification_result = self.classify_query(user_query)
        classified_as = classification_result.get('classification', 'unknown')

        response_data = {
            'user_query': user_query,
            'classification': classification_result,
            'response': '',
            'response_type': classified_as
        }

        if classified_as == 'faq':
            response_data['response'] = self._handle_faq_query(user_query, classification_result)
        elif classified_as == 'sql_query':
            response_data['response'] = self._handle_sql_query(user_query, classification_result)
        else:
            response_data['response'] = Message.REPHRASE_MESSAGE.value

        logger.info(f"Query processed: {classified_as}")
        return response_data

    def _handle_faq_query(self, user_query, classification_result):
        """Handle FAQ-type queries"""
        # Generate final response
        # Prepare variables for response generation
        response_variables = {"USER_QUERY": user_query, "CLASSIFICATION_RESULT": classification_result}
        response_prompt = prompt_for_navigation.format(**response_variables)
        final_response = self.final_response(response_prompt)
        response = final_response.replace('\n', '').replace(r'\"', '')

        intent_data = classification_result.get('intent_data')
        if intent_data:
            # response = intent_data.get('response', 'I can help with that topic.')

            # Add quick actions if available
            # quick_actions = intent_data.get('quick_actions', [])
            # if quick_actions:
            #     response += "\n\nQuick actions:"
            #     for action in quick_actions[:2]:  # Limit to 2 actions
            #         response += f"\n• {action.get('text', '')}"

            return response
        else:
            return "I found a matching FAQ topic, but I don't have the detailed response available."

    def _handle_sql_query(self, user_query, result) -> str:
        """Handle SQL-related queries with error correction and retry logic"""
        variables = {
            "INTENTS_TEXT": self.intents_text,
            "DATABASE_SCHEMA": self.database_schema,
            "USER_QUERY": user_query
        }
        return self._execute_sql_with_retry(
            user_query=user_query,
            variables=variables,
            classification_result=result,
            max_retries=3
        )

    def _execute_sql_with_retry(self, user_query, variables, classification_result, max_retries=3) -> str:
        """
        Generic method to execute SQL queries with error correction and retry logic.
        Handles both original queries and error correction scenarios.
        """
        current_result = classification_result
        retry_count = 0

        while retry_count <= max_retries:
            try:
                sql_query = current_result.get('sql_query')
                if not sql_query:
                    return "I couldn't generate a valid SQL query for your request. Please try rephrasing your question."

                logger.info(f"Executing SQL query (attempt {retry_count + 1}): {sql_query}")
                print(f"Executing query (attempt {retry_count + 1}): {sql_query}")

                # Execute the query
                query_result = self.db_connector.execute_query(query=str(sql_query))

                # Check if query execution was successful
                if query_result.get('error'):
                    error_message = query_result['error']
                    logger.warning(f"SQL query failed (attempt {retry_count + 1}): {error_message}")

                    if retry_count >= max_retries:
                        return f"I encountered an error while processing your request: {error_message}. Please try rephrasing your question or contact support."

                    # Prepare error correction
                    retry_count += 1
                    current_result = self._correct_sql_error(
                        user_query=user_query,
                        failed_sql_query=sql_query,
                        error_message=error_message,
                        variables=variables
                    )

                    if not current_result:
                        return "I couldn't correct the SQL query error. Please try rephrasing your question."

                    continue  # Retry with corrected query

                # Query executed successfully, generate response
                logger.info(f"SQL query executed successfully, {query_result.get('row_count', 0)} rows returned")

                # Prepare variables for response generation
                response_variables = variables.copy()
                response_variables['DATA_'] = query_result
                response_variables['SQL_QUERY'] = sql_query

                # Generate final response
                response_prompt = self.prompt_with_sql_data.format(**response_variables)
                final_response = self.final_response(response_prompt)

                return final_response

            except Exception as e:
                logger.error(f"Unexpected error in SQL execution (attempt {retry_count + 1}): {str(e)}")

                if retry_count >= max_retries:
                    return f"I encountered an unexpected error while processing your request: {str(e)}. Please try again later."

                retry_count += 1
                # For unexpected errors, we'll try to regenerate the query
                current_result = self._correct_sql_error(
                    user_query=user_query,
                    failed_sql_query=current_result.get('sql_query', ''),
                    error_message=str(e),
                    variables=variables
                )

                if not current_result:
                    return "I encountered an error and couldn't recover. Please try rephrasing your question."

        return "I couldn't process your request after multiple attempts. Please try rephrasing your question."

    def _correct_sql_error(self, user_query, failed_sql_query, error_message, variables) -> Optional[Dict]:
        """
        Generate a corrected SQL query based on the error message.
        Returns the corrected classification result or None if correction fails.
        """
        try:
            # Prepare variables for error correction prompt
            error_variables = variables.copy()
            error_variables.update({
                'USER_QUERY': user_query,
                'DATABASE_SCHEMA': self.database_schema,
                'SQL_QUERY': failed_sql_query,
                'ERROR': error_message
            })

            # Generate error correction prompt
            error_correction_prompt = self.prompt_with_sql_error.format(**error_variables)

            logger.info(f"Attempting to correct SQL error: {error_message}")

            # Get corrected query from Gemini
            response = self.model.generate_content(
                error_correction_prompt,
                generation_config=self.generation_config
            )

            # Parse the corrected result
            corrected_result = json.loads(response.text.strip().replace('```json', '').replace('```', ''))

            logger.info(f"Generated corrected SQL query: {corrected_result.get('sql_query')}")

            return corrected_result

        except Exception as e:
            logger.error(f"Error in SQL correction: {str(e)}")
            return None

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