"""
Database Inspector Service
Connects to PostgreSQL database and extracts schema information
"""

import logging
import psycopg2
from typing import Dict, List, Optional
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


class PostgreSQLConnector:
    """PostgreSQL database connector for the chatbot"""

    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.connection = None
        self.cursor = None

        logger.info("PostgreSQL connector initialized")

    def connect(self) -> bool:
        try:
            self.connection = psycopg2.connect(
                host=self.db_config.get('host', 'localhost'),
                port=self.db_config.get('port', 5432),
                database=self.db_config.get('database'),
                user=self.db_config.get('user'),
                password=self.db_config.get('password'),
                cursor_factory=RealDictCursor
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
