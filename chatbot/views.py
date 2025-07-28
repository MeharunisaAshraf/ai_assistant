from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view
from django.conf import settings
import json
import os

from .services.router import QueryRouter
from .services.sql_model import SQLAgent
from .services.model_intent_classifier import HybridIntentClassifier
from .serializers import ChatQuerySerializer

# Global router instance
_router = None

def get_router():
    """Initialize and return the router instance"""

    global _router
    
    if _router is None:
        try:
            # Initialize SQL Agent
            sql_agent = SQLAgent(
                db_uri=settings.DATABASE_URL,
                gemini_api_key=settings.GEMINI_API_KEY,
                tables_to_use=getattr(settings, 'CHATBOT_TABLES', None),
                model_name="gemini-2.5-flash"
            )
            
            # Load intents data
            intents_path = os.path.join(settings.BASE_DIR, 'chatbot', 'data', 'intents.json')
            with open(intents_path, 'r') as f:
                intents_data = json.load(f)
            
            # Initialize Intent Classifier
            intent_classifier = HybridIntentClassifier(
                intents_data,
                similarity_threshold=getattr(settings, 'INTENT_SIMILARITY_THRESHOLD', 0.85),
                ollama_url=getattr(settings, 'OLLAMA_URL', 'http://localhost:11434'),
                model_name=getattr(settings, 'OLLAMA_MODEL', 'llama3.2:3b')
            )
            
            # Initialize Router with Gemini 2.5 Pro
            _router = QueryRouter(
                sql_agent=sql_agent,
                intent_classifier=intent_classifier,
                model_name="gemini-2.5-pro"
            )
            
            print("Query Router initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize router: {e}")
            raise
    
    return _router

class ChatbotQueryView(APIView):
    """Main chatbot endpoint for single queries"""
    
    def post(self, request):
        serializer = ChatQuerySerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(
                {'error': 'Invalid input', 'details': serializer.errors},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            router = get_router()
            query = serializer.validated_data['query']
            
            # Process the query
            result = router.process_query(query)
            
            # Add SQL query to response if requested and available
            if (serializer.validated_data.get('return_sql') and 
                result.get('type') == 'sql_response' and 
                result.get('sql_query')):
                result['debug_sql'] = result.get('sql_query')
            
            return Response(result, status=status.HTTP_200_OK)
            
        except Exception as e:
            return Response(
                {
                    'error': 'Internal server error',
                    'message': str(e),
                    'type': 'error_response'
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )