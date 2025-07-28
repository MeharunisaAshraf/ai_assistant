#!/usr/bin/env python3
"""
Standalone Chatbot Script
Uses Gemini AI to classify user queries based on intent.json
Determines if query is FAQ-based or SQL-related
"""

import json
import os
from typing import Dict, Optional, List
import google.generativeai as genai
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from chatbot.data.prompts import initial_prompt, prompt_with_sql_data, prompt_with_sql_error
logger = logging.getLogger(__name__)


def _load_config() -> Dict:
    """Load configuration from config.json"""
    try:
        config_file_path = "config.json"
        if not os.path.exists(config_file_path):
            logger.error(f"Config file not found: {config_file_path}")
            return {}

        with open(config_file_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        return config_data

    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return {}


def _load_intents() -> Dict:
    """Load intents from JSON file"""
    try:
        intents_file_path = "chatbot/data/intents.json"
        if not os.path.exists(intents_file_path):
            logger.error(f"Intents file not found: {intents_file_path}")
            return {"intents": []}

        with open(intents_file_path, 'r', encoding='utf-8') as f:
            intents_data = json.load(f)

        logger.info(f"Loaded {len(intents_data.get('intents', []))} intents")
        return intents_data

    except Exception as e:
        logger.error(f"Error loading intents: {str(e)}")
        return {"intents": []}


def get_db_connection():
    config = _load_config()
    db_config = {
        'host': config.get('host'),
        'port': config.get('port'),
        'database': config.get('database'),
        'user': config.get('user'),
        'password': config.get('password')
    }
    database_schema = {}
    db_connector = PostgreSQLConnector(db_config)
    try:
        if db_connector.connect():
            print("Database connected successfully!")
            tables = db_connector.get_all_tables()
            print(f"Found {len(tables)} tables in database")
            database_schema = db_connector.get_database_schema()
            print(f"Retrieved schema for {len(database_schema)} tables")

    except Exception as e:
        print(f"Error connecting to database: {str(e)}")

    return db_connector, database_schema



class ChatbotClassifier:
    """Simple chatbot that classifies user queries using Gemini AI"""

    def __init__(self):
        self.prompt = initial_prompt
        self.prompt_with_sql_data = prompt_with_sql_data
        self.prompt_with_sql_error = prompt_with_sql_error
        self.config = _load_config()
        self.intents_data = _load_intents()

        genai.configure(api_key=self.config.get('GEMINI_API_KEY', ''))
        print("Connecting to Gemini...:", self.config.get('GEMINI_API_KEY', ''))
        self.model = genai.GenerativeModel(self.config.get('MODEL_NAME', 'gemini-2.5-flash-lite'))

        self.generation_config = genai.types.GenerationConfig(
            temperature=self.config.get('TEMPERATURE', 0.1),
            top_p=self.config.get('TOP_P', 0.8),
            top_k=self.config.get('TOP_K', 40),
            max_output_tokens=self.config.get('MAX_OUTPUT_TOKENS', 1024),
        )

        logger.info("Chatbot classifier initialized successfully")

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


class PostgreSQLConnector:
    """PostgreSQL database connector for the chatbot"""

    def __init__(self, db_config: Dict):
        """
        Initialize PostgreSQL connection

        Args:
            db_config: Dictionary with database connection parameters
                      {host, port, database, user, password}
        """
        self.db_config = db_config
        self.connection = None
        self.cursor = None

        logger.info("PostgreSQL connector initialized")

    def connect(self) -> bool:
        """
        Establish connection to PostgreSQL database

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.connection = psycopg2.connect(
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', 5432),
                database=self.db_config.get('database'),
                user=self.db_config.get('user'),
                password=self.db_config.get('password'),
                cursor_factory=RealDictCursor  # Returns dict-like rows
            )

            self.cursor = self.connection.cursor()
            logger.info("Successfully connected to PostgreSQL database")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
            return False

    def disconnect(self):
        """Close database connection"""
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            logger.info("Disconnected from PostgreSQL database")
        except Exception as e:
            logger.error(f"Error disconnecting from database: {str(e)}")

    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            if not self.connection:
                return self.connect()

            self.cursor.execute("SELECT 1")
            result = self.cursor.fetchone()
            return result is not None

        except Exception as e:
            logger.error(f"Database connection test failed: {str(e)}")
            return False

    def execute_query(self, query: str, params: Optional[tuple] = None) -> Dict:
        """
        Execute a SQL query and return results

        Args:
            query: SQL query string
            params: Optional query parameters for prepared statements

        Returns:
            Dict with results, row_count, and error information
        """
        try:
            if not self.connection or not self.cursor:
                if not self.connect():
                    return {
                        'results': [],
                        'row_count': 0,
                        'error': 'Database connection failed'
                    }

            if params:
                self.cursor.execute(query, params)
            else:
                logger.info(f"Executing query inside execute_query: {query}")
                self.cursor.execute(query)

            if query.strip().upper().startswith('SELECT'):
                results = self.cursor.fetchall()
                results = [dict(row) for row in results]
                row_count = len(results)
            else:
                # For INSERT, UPDATE, DELETE queries
                results = []
                row_count = 0

            logger.info(f"Query executed successfully, {row_count} rows affected")
            return {
                'results': results,
                'row_count': row_count,
                'error': None
            }

        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            if self.connection:
                self.connection.rollback()

            return {
                'results': [],
                'row_count': 0,
                'error': str(e)
            }

    def get_table_schema(self, table_name: str) -> Dict:
        """
        Get schema information for a specific table

        Args:
            table_name: Name of the table

        Returns:
            Dict with table schema information
        """
        try:
            query = """
            SELECT
                column_name,
                data_type,
                is_nullable,
                column_default,
                character_maximum_length
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position
            """

            result = self.execute_query(query, (table_name,))

            if result['error']:
                return {'error': result['error']}

            columns = []
            for row in result['results']:
                columns.append({
                    'name': row['column_name'],
                    'type': row['data_type'],
                    'nullable': row['is_nullable'] == 'YES',
                    'default': row['column_default'],
                    'max_length': row['character_maximum_length']
                })

            return {
                'table_name': table_name,
                'columns': columns,
                'error': None
            }

        except Exception as e:
            logger.error(f"Error getting table schema: {str(e)}")
            return {'error': str(e)}

    def get_all_tables(self) -> List[str]:
        """
        Get list of all tables in the database

        Returns:
            List of table names
        """
        try:
            query = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
            """

            result = self.execute_query(query)

            if result['error']:
                logger.error(f"Error getting table list: {result['error']}")
                return []

            return [row['table_name'] for row in result['results'] if (row['table_name'].startswith('properties') or row['table_name'].startswith('user'))]

        except Exception as e:
            logger.error(f"Error getting table list: {str(e)}")
            return []

    def get_database_schema(self) -> Dict:
        """
        Get complete database schema information

        Returns:
            Dict with schema information for all tables
        """
        try:
            tables_ = self.get_all_tables()
            tables = [t for t in tables_ if t.startswith('properties') or t.startswith('user')]
            schema = {}

            for table_name in tables:
                table_schema = self.get_table_schema(table_name)
                if not table_schema.get('error'):
                    schema[table_name] = table_schema

            logger.info(f"Retrieved schema for {len(schema)} tables")
            return schema

        except Exception as e:
            logger.error(f"Error getting database schema: {str(e)}")
            return {}


def main():
    """Main function to run the chatbot interactively"""
    print("=== AI Chatbot ===")
    print("Type 'quit' to exit.\n")

    print("Setting up database connection...")
    db_connector, database_schema = get_db_connection()
    try:
        chatbot = ChatbotClassifier()

        if not chatbot.test_connection():
            print("Failed to connect to Gemini API. Please check your API key.")
            return

        print("Chatbot initialized successfully!")
        print(f"Loaded {len(chatbot.intents_data.get('intents', []))} FAQ intents.\n")
        intents_text = chatbot.get_intent_descriptions()

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break

            if not user_input:
                continue

            variables = {
                "INTENTS_TEXT": intents_text,
                "DATABASE_SCHEMA": database_schema,
                "USER_QUERY": user_input
            }

            result = chatbot.process_query(chatbot, user_input, database_schema, db_connector, result=None, intents_text=intents_text, variables=variables)
            print(f"Bot: {result.get('response')}")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
