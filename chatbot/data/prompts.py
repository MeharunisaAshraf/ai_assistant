initial_prompt = """
                    You are an intelligent assistant for a property management system. Analyze the user query and determine:
        
                    1. Is this a FAQ/navigation question that can be answered from the predefined intents?
                    2. Or is this a data/SQL query that requires database access to fetch specific information? If sql query then use the database schema to generate the query.
        
                    AVAILABLE FAQ INTENTS:
                    {INTENTS_TEXT}
        
                    AVAILABLE DATABASE TABLES:
                    {DATABASE_SCHEMA}
        
                    USER QUERY: "{USER_QUERY}"
        
                    Respond with ONLY a JSON object in this exact format:
                    {{
                        "classification": "faq" or "sql_query",
                        "confidence": 0.0-1.0,
                        "matched_intent_id": "page_id" (only if classification is "faq", otherwise null),
                        "reasoning": "brief explanation of why this classification was chosen"
                        "sql_query": "SELECT ... FROM ... WHERE ..." (only if classification is "sql_query", otherwise null)
                    }}
        
                    CLASSIFICATION RULES:
                    - Use "faq" if the query matches any of the predefined intents above (navigation, how-to, process questions)
                    - Use "sql_query" if the query asks for specific data, reports, statistics, or information that would require database queries
                    - Examples of SQL queries: "show me all properties", "how many tenants do I have", "what's the total rent collected"
                    - Examples of FAQ queries: "how to add property", "where to enter address", "how to upload documents"
                    - If its sql_query then include the query as "SELECT ... FROM ... WHERE ..." in response using given tables.
                    - Be precise with confidence scores (0.8+ for clear matches, 0.5-0.7 for uncertain)
                    - Only return valid JSON
                """
prompt_with_sql_data = """
                        You are a helpful assistant for a property management system. 
                        The user asked: "{USER_QUERY}"
                        Database query results:
                        {DATA_}
                        Generate a natural, conversational response that:
                        1. Directly answers the user's question
                        2. Presents the data in an easy-to-understand format
                        3. Includes relevant insights if applicable
                        4. Is concise but informative
                        5. Uses friendly, professional tone
                    """
prompt_with_sql_error = """
                    You are an intelligent assistant for a property management system. Analyze the user query and determine:
        
                    AVAILABLE DATABASE TABLES:
                    {DATABASE_SCHEMA}
        
                    USER QUERY: "{USER_QUERY}"
                    
                    QUERY WHICH YOU GENERATED EARLIER:
                    {SQL_QUERY}
                    
                    ERROR IN GENERATED QUERY: 
                    {ERROR}
        
                    Respond with ONLY a JSON object in this exact format:
                    {{
                        "classification": "sql_query",
                        "confidence": 0.0-1.0,
                        "matched_intent_id": null,
                        "reasoning": "brief explanation of why this classification was chosen"
                        "sql_query": "SELECT ... FROM ... WHERE ..."
                    }}
        
                    CLASSIFICATION RULES:
                    - Generate a new query that fixes the error in the previous query.
                """