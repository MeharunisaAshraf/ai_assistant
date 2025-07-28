import os
import json
import logging
from typing import Dict
from chatbot.helpers.database_inspector import PostgreSQLConnector
logger = logging.getLogger(__name__)

def load_config() -> Dict:
    """Load configuration from config.json"""
    try:
        config_file_path = r"D:\Zapta\AI\ai_assistant\config.json"
        if not os.path.exists(config_file_path):
            logger.error(f"Config file not found: {config_file_path}")
            return {}

        with open(config_file_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        return config_data

    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return {}


def load_intents() -> Dict:
    """Load intents from JSON file"""
    try:
        intents_file_path = r"D:\Zapta\AI\ai_assistant\chatbot\data\intents.json"
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
    config = load_config()
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
            database_schema = db_connector.get_database_schema()
            print(f"Retrieved schema successfully!")

    except Exception as e:
        print(f"Error connecting to database: {str(e)}")

    return db_connector, database_schema

