import re
import os
import time
from google import genai
from typing import List, Dict, Tuple, Optional
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool

class SQLAgent:
    """SQL Agent for database queries using Gemini"""

    def __init__(self,
                 db_uri: str,
                 gemini_api_key: str,
                 tables_to_use: List[str] = None,
                 model_name: str = "gemini-2.5-flash",
                 top_k: int = 5):

        self.db_uri = db_uri
        self.model_name = model_name
        self.top_k = top_k

        # Initialize Gemini client
        self.client = genai.Client(api_key=gemini_api_key)

        # Initialize database connection
        self._initialize_database(tables_to_use)

        # Build system prompt template
        self._build_system_prompt()

        print(f"SQL Agent initialized with {len(tables_to_use or [])} tables")
        print(f"Database dialect: {self.db.dialect}")

    def _initialize_database(self, tables_to_use: List[str] = None):
        """Initialize database connection and query tool"""

        try:
            if tables_to_use:
                self.db = SQLDatabase.from_uri(
                    self.db_uri,
                    include_tables=tables_to_use,
                    sample_rows_in_table_info=3,
                    indexes_in_table_info=False
                )
            else:
                self.db = SQLDatabase.from_uri(self.db_uri)

            # Initialize query execution tool
            self.execute_query_tool = QuerySQLDatabaseTool(db=self.db)

            # Get table info for prompts
            self.table_info = self.db.get_table_info()
            self.dialect = self.db.dialect

        except Exception as e:
            raise RuntimeError(f"Failed to initialize database connection: {str(e)}")

    def _build_system_prompt(self):
        """Build the system prompt template for SQL generation"""

        self.system_message_template = """
Given an input question, create a syntactically correct {dialect} query to
run to help find the answer. Unless the user specifies in his question a
specific number of examples they wish to obtain, always limit your query to
at most {top_k} results. You can order the results by a relevant column to
return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the
few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.

Only use the following tables:
{table_info}

Return only the SQL query without any additional text or formatting.
"""

        self.user_prompt_template = "Question: {input}"

    def _build_prompt(self, user_input: str, top_k: int = None) -> str:
        """Build the complete prompt for SQL generation"""

        if top_k is None:
            top_k = self.top_k

        system_prompt = self.system_message_template.format(
            dialect=self.dialect,
            top_k=top_k,
            table_info=self.table_info
        )

        user_prompt = self.user_prompt_template.format(input=user_input)

        return system_prompt + "\n\n" + user_prompt

    def _clean_sql_query(self, raw_sql: str) -> str:
        """Clean LLM-generated SQL query"""

        # Remove opening triple backticks and optional language specifier
        cleaned = re.sub(r"```[\w\-]*\s*", "", raw_sql, flags=re.IGNORECASE)
        # Remove closing triple backticks
        cleaned = re.sub(r"\s*```", "", cleaned)
        return cleaned.strip()

    def _generate_sql_query(self, question: str, top_k: int = None) -> str:
        """Generate SQL query using Gemini"""

        try:
            prompt = self._build_prompt(question, top_k)

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )

            raw_sql = response.text
            cleaned_sql = self._clean_sql_query(raw_sql)

            return cleaned_sql

        except Exception as e:
            raise RuntimeError(f"Failed to generate SQL query: {str(e)}")

    def _execute_sql_query(self, sql_query: str) -> str:
        """Execute SQL query against database"""

        try:
            result = self.execute_query_tool.invoke(sql_query)
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to execute SQL query: {str(e)}")

    def _generate_natural_language_answer(self, question: str, query: str, result: str) -> str:
        """Generate natural language answer from SQL results"""

        try:
            prompt = (
                "Given the following user question, corresponding SQL query, "
                "and SQL result, answer the user question in natural language.\n\n"
                f"Question: {question}\n"
                f"SQL Query: {query}\n"
                f"SQL Result: {result}"
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )

            return response.text.strip()

        except Exception as e:
            raise RuntimeError(f"Failed to generate natural language answer: {str(e)}")

    def query(self, question: str, top_k: int = None, return_sql: bool = False) -> Dict:
        """ Process a natural language question and return answer """

        start_time = time.time()

        try:
            # Step 1: Generate SQL query
            sql_generation_start = time.time()
            sql_query = self._generate_sql_query(question, top_k)
            sql_generation_time = time.time() - sql_generation_start

            # Step 2: Execute SQL query
            execution_start = time.time()
            sql_result = self._execute_sql_query(sql_query)
            execution_time = time.time() - execution_start

            # Step 3: Generate natural language answer
            answer_generation_start = time.time()
            answer = self._generate_natural_language_answer(question, sql_query, sql_result)
            answer_generation_time = time.time() - answer_generation_start

            total_time = time.time() - start_time

            result = {
                'question': question,
                'answer': answer,
                'sql_result': sql_result,
                'success': True,
                'total_time_s': round(total_time, 3),
                'sql_generation_time_s': round(sql_generation_time, 3),
                'execution_time_s': round(execution_time, 3),
                'answer_generation_time_s': round(answer_generation_time, 3),
                'model_name': self.model_name,
                'dialect': self.dialect
            }

            if return_sql:
                result['sql_query'] = sql_query

            return result

        except Exception as e:
            return {
                'question': question,
                'answer': f"I encountered an error while processing your question: {str(e)}",
                'success': False,
                'error': str(e),
                'total_time_s': round(time.time() - start_time, 3),
                'model_name': self.model_name,
                'dialect': self.dialect
            }

    def batch_query(self, questions: List[str], top_k: int = None, return_sql: bool = False) -> List[Dict]:
        """Process multiple questions"""

        results = []
        for question in questions:
            result = self.query(question, top_k, return_sql)
            results.append(result)

        return results

    def test_connection(self) -> bool:
        """Test database connection"""

        try:
            # Try a simple query
            test_result = self.db.run("SELECT 1 as test")
            return True
        except:
            return False

    def get_table_names(self) -> List[str]:
        """Get list of available table names"""

        return self.db.get_usable_table_names()

    def get_table_schema(self, table_name: str) -> str:
        """Get schema for a specific table"""

        try:
            return self.db.get_table_info_no_throw([table_name])
        except:
            return f"Table '{table_name}' not found or inaccessible"




if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    # Get credentials from environment
    db_uri = os.getenv("DB_URI")
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    tables_to_use = ['properties_amenities',
                     'properties_calendarslot',
                     'properties_costfee',
                     'properties_costfeescategory',
                     'properties_invitation',
                     'properties_listinginfo',
                     'properties_ownerinfo',
                     'properties_property',
                     'properties_propertyassignedamenities',
                     'properties_propertydocument',
                     'properties_propertyphoto',
                     'properties_propertytypeandamenity',
                     'properties_rentdetails',
                     'properties_unit',
                     'user_authentication_tenant',
                     'user',
                     'user_authentication_role',
                     'user_authentication_propertyowner',
                     'user_authentication_vendor',
                     'user_authentication_servicecategory',
                     'user_authentication_servicesubcategory',
                     'user_authentication_vendorservices',
                     'user_authentication_vendorinvitation',
                     'user_authentication_tenantinvitation']

    # Initialize SQLAgent
    agent = SQLAgent(
        db_uri=db_uri,
        gemini_api_key=gemini_api_key,
        tables_to_use=tables_to_use,
        model_name="gemini-2.5-flash",
        top_k=5
    )

    # Test connection
    if agent.test_connection():
        print("✅ Database connection successful.")
    else:
        print("❌ Failed to connect to the database.")

    # Example question
    # question = "Find units that allow pets, along with the allowed pet types."
    question = "On which properties tenants are already assigned?"

    # Execute query
    result = agent.query(question, return_sql=True)

    # Print results
    print("\n--- Query Result ---")
    print("Question:", result.get("question"))
    print("\n\nSQL Query:", result.get("sql_query"))
    print("\n\nSQL Result:", result.get("sql_result"))
    print("\n\nAnswer:", result.get("answer"))
    print("\nTotal Time (s):", result.get("total_time_s"))