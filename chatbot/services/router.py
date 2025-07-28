import os
import json
import time
from pathlib import Path
from google import genai
from google.genai import types
from typing import Dict, List, Optional
from enum import Enum
from dataclasses import dataclass

class QueryType(Enum):
    SQL_DATA = "sql_data"
    NAVIGATION = "navigation"
    GENERAL = "general"

@dataclass
class RouterResult:
    query_type: QueryType
    confidence: float
    reasoning: str
    intent_data: Optional[Dict] = None
    classification_time: float = 0.0


current_dir = Path(__file__).resolve().parent
intents_path = current_dir.parent / 'data' / 'intents.json'

with open(intents_path, 'r') as f:
    intents_data = json.load(f)


class QueryRouter:
    """ Gemini-powered query router for classification """
    
    def __init__(self, sql_agent, intent_classifier, model_name: str = "gemini-2.5-pro"):
        self.sql_agent = sql_agent
        self.intent_classifier = intent_classifier
        self.model_name = model_name
        
        # Initialize Gemini
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Build system prompt
        self._build_system_prompt()
        
        print(f"Query Router initialized with {model_name}")

    def _build_system_prompt(self) -> str:
        """Build system prompt for query classification"""
        
        # Get available database tables for context
        try:
            available_tables = self.sql_agent.get_table_names()
            table_context = f"Available database tables: {', '.join(available_tables)}"
        except:
            table_context = "Database tables available for queries"
        
        # Get available intents for context
        try:
            intent_pages = [intent['page_id'] for intent in intents_data['intents']]
            intent_context = f"Available navigation pages: {', '.join(intent_pages)}"
        except:
            intent_context = "Navigation intents available for website guidance"

        self.system_prompt = f"""You are a query classification expert for a real estate property management system.

Your task is to classify user queries into exactly ONE of these categories:

1. **SQL_DATA**: Queries asking for data, reports, statistics, or information retrieval
   - Examples: "Show me all tenants", "How many properties do I have?", "List overdue payments"
   - These require database queries to answer

2. **NAVIGATION**: Queries asking how to use the website, where to find features, or step-by-step guidance
   - Examples: "How to add a property?", "Where is the tenant page?", "How do I upload documents?"
   - These require website navigation guidance

3. **GENERAL**: Greetings, general questions, or unclear queries that don't fit above categories
   - Examples: "Hello", "What can you do?", "Help me"

Context:
- {table_context}
- {intent_context}

CRITICAL RULES:
1. Output ONLY valid JSON in this exact format: {{"query_type": "SQL_DATA|NAVIGATION|GENERAL", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}
2. Be decisive - choose the MOST likely category
3. Use confidence 0.8+ for clear matches, 0.5-0.7 for uncertain ones
4. Keep reasoning under 50 words
5. No additional text outside the JSON response

Examples:
- "Show me all my properties" → {{"query_type": "SQL_DATA", "confidence": 0.95, "reasoning": "Clear data retrieval request for properties"}}
- "How to add a new tenant?" → {{"query_type": "NAVIGATION", "confidence": 0.9, "reasoning": "Asking for step-by-step guidance on website feature"}}
- "Hello there" → {{"query_type": "GENERAL", "confidence": 0.8, "reasoning": "Greeting, no specific data or navigation request"}}
"""

    def classify_query(self, user_query: str) -> RouterResult:
        """
        Classify user query using Gemini 2.5 Pro
        """
        start_time = time.time()
        
        try:
            # Prepare the prompt
            full_prompt = f"{self.system_prompt}\n\nClassify this query: \"{user_query}\""
            
            # Generate response from Gemini
            response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=full_prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.1, # Low temperature for consistent classification
                            candidate_count=1
                        )
                    )
            
            # Parse the JSON response
            response_text = response.text.strip()
            
            # Clean response (remove markdown formatting if present)
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            classification = json.loads(response_text)
            
            # Validate response
            query_type_str = classification.get('query_type', '').upper()
            if query_type_str not in ['SQL_DATA', 'NAVIGATION', 'GENERAL']:
                raise ValueError(f"Invalid query_type: {query_type_str}")
            
            query_type = QueryType(query_type_str.lower())
            confidence = float(classification.get('confidence', 0.0))
            reasoning = classification.get('reasoning', 'No reasoning provided')
            
            classification_time = time.time() - start_time
            
            return RouterResult(
                query_type=query_type,
                confidence=confidence,
                reasoning=reasoning,
                classification_time=classification_time
            )
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}, Response: {response_text}")
            return RouterResult(
                query_type=QueryType.GENERAL,
                confidence=0.1,
                reasoning=f"Failed to parse classification response: {str(e)}",
                classification_time=time.time() - start_time
            )
            
        except Exception as e:
            print(f"Classification error: {e}")
            return RouterResult(
                query_type=QueryType.GENERAL,
                confidence=0.1,
                reasoning=f"Classification failed: {str(e)}",
                classification_time=time.time() - start_time
            )

    def process_query(self, user_query: str) -> Dict:
        """ Complete query processing pipeline """
        
        print(f"Processing query: {user_query}")

        # Step 1: Classify the query type
        routing_result = self.classify_query(user_query)
        
        # Step 2: Process based on classification
        if routing_result.query_type == QueryType.SQL_DATA:
            return self._process_sql_query(user_query, routing_result)
            
        elif routing_result.query_type == QueryType.NAVIGATION:
            return self._process_navigation_query(user_query, routing_result)
            
        else:
            return self._process_general_query(user_query, routing_result)

    def _process_sql_query(self, query: str, routing_result: RouterResult) -> Dict:
        """Process data retrieval queries using SQL Agent"""
        try:
            sql_result = self.sql_agent.query(query)
            
            return {
                'type': 'sql_response',
                'success': sql_result.get('success', False),
                'routing': {
                    'query_type': routing_result.query_type.value,
                    'confidence': routing_result.confidence,
                    'reasoning': routing_result.reasoning,
                    'classification_time': routing_result.classification_time
                },
                'response_text': sql_result.get('answer', 'No answer generated'),
                'sql_query': sql_result.get('sql_query') if sql_result.get('success') else None,
                'sql_result': sql_result.get('sql_result') if sql_result.get('success') else None,
                'execution_times': {
                    'total': sql_result.get('total_time_s', 0),
                    'sql_generation': sql_result.get('sql_generation_time_s', 0),
                    'execution': sql_result.get('execution_time_s', 0),
                    'answer_generation': sql_result.get('answer_generation_time_s', 0),
                    'classification': routing_result.classification_time
                },
                'error': sql_result.get('error') if not sql_result.get('success') else None
            }
            
        except Exception as e:
            return {
                'type': 'sql_response',
                'success': False,
                'routing': {
                    'query_type': routing_result.query_type.value,
                    'confidence': routing_result.confidence,
                    'reasoning': routing_result.reasoning,
                    'classification_time': routing_result.classification_time
                },
                'error': str(e),
                'response_text': f"I encountered an error retrieving that data: {str(e)}",
                'execution_times': {
                    'classification': routing_result.classification_time
                }
            }

    def _process_navigation_query(self, query: str, routing_result: RouterResult) -> Dict:
        """Process navigation/how-to queries using Intent Classifier"""
        try:
            intent_result = self.intent_classifier.classify_intent(query)
            
            if not intent_result.get('results'):
                return {
                    'type': 'navigation_response',
                    'success': False,
                    'routing': {
                        'query_type': routing_result.query_type.value,
                        'confidence': routing_result.confidence,
                        'reasoning': routing_result.reasoning,
                        'classification_time': routing_result.classification_time
                    },
                    'response_text': "I couldn't find specific guidance for that task. Could you be more specific about what you're trying to do?",
                    'suggestions': [
                        "Try asking: 'How to add a new property?'",
                        "Or: 'Where do I find tenant information?'",
                        "Or: 'How to upload documents?'"
                    ],
                    'execution_times': {
                        'classification': routing_result.classification_time,
                        'intent_classification': intent_result.get('inference_time_s', 0)
                    }
                }
            
            best_match = intent_result['results'][0]
            
            return {
                'type': 'navigation_response',
                'success': True,
                'routing': {
                    'query_type': routing_result.query_type.value,
                    'confidence': routing_result.confidence,
                    'reasoning': routing_result.reasoning,
                    'classification_time': routing_result.classification_time
                },
                'intent': {
                    'page_id': best_match['page_id'],
                    'confidence': best_match['confidence'],
                    'category': best_match['category'],
                    'intent_type': best_match.get('intent_type', 'navigation')
                },
                'response_text': best_match['response'],
                'quick_actions': best_match.get('quick_actions', []),
                'next_steps': best_match.get('next_steps', []),
                'execution_times': {
                    'classification': routing_result.classification_time,
                    'intent_classification': intent_result.get('inference_time_s', 0),
                    'total': routing_result.classification_time + intent_result.get('inference_time_s', 0)
                },
                'classifier_info': {
                    'type': intent_result.get('classifier_type', 'unknown'),
                    'primary_method': intent_result.get('primary_method'),
                    'fallback_used': intent_result.get('fallback_used', False)
                }
            }
            
        except Exception as e:
            return {
                'type': 'navigation_response',
                'success': False,
                'routing': {
                    'query_type': routing_result.query_type.value,
                    'confidence': routing_result.confidence,
                    'reasoning': routing_result.reasoning,
                    'classification_time': routing_result.classification_time
                },
                'error': str(e),
                'response_text': f"I encountered an error processing your navigation request: {str(e)}",
                'execution_times': {
                    'classification': routing_result.classification_time
                }
            }

    def _process_general_query(self, query: str, routing_result: RouterResult) -> Dict:
        """Process general queries or greetings"""
        
        # Basic greeting/help responses
        if any(greeting in query.lower() for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            response_text = "Hello! I'm here to help you with your property management tasks. I can help you find data about your properties and tenants, or guide you through website features."
        elif any(help_word in query.lower() for help_word in ['help', 'what can you do', 'capabilities']):
            response_text = "I can help you in two main ways:\n\n1. **Find Data**: Ask me about your properties, tenants, payments, or any other information in your database.\n2. **Website Guidance**: Ask me how to use features like adding properties, managing tenants, or uploading documents."
        else:
            response_text = "I'm here to help! You can ask me to find information about your properties and tenants, or I can guide you through using website features. What would you like to do?"
        
        return {
            'type': 'general_response',
            'success': True,
            'routing': {
                'query_type': routing_result.query_type.value,
                'confidence': routing_result.confidence,
                'reasoning': routing_result.reasoning,
                'classification_time': routing_result.classification_time
            },
            'response_text': response_text,
            'suggestions': [
                {
                    'text': 'Show me my properties',
                    'type': 'sql_example'
                },
                {
                    'text': 'How to add a new tenant?',
                    'type': 'navigation_example'
                },
                {
                    'text': 'List overdue payments',
                    'type': 'sql_example'
                },
                {
                    'text': 'How to upload documents?',
                    'type': 'navigation_example'
                }
            ],
            'execution_times': {
                'classification': routing_result.classification_time
            }
        }

    def batch_process(self, queries: List[str]) -> List[Dict]:
        """Process multiple queries in batch"""
        results = []
        for query in queries:
            result = self.process_query(query)
            results.append(result)
        return results

    def get_stats(self) -> Dict:
        """Get router statistics and health info"""
        return {
            'router_type': 'gemini_powered',
            'model_name': self.model_name,
            'available_tables': self.sql_agent.get_table_names() if hasattr(self.sql_agent, 'get_table_names') else [],
            'intent_count': len(self.intent_classifier.intents) if hasattr(self.intent_classifier, 'intents') else 0,
            'sql_agent_dialect': getattr(self.sql_agent, 'dialect', 'unknown'),
            'intent_classifier_type': type(self.intent_classifier).__name__
        }