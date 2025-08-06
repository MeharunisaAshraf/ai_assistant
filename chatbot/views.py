import logging
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from chatbot.services.gemini import AIAssistant
from chatbot.serializers import ChatQuerySerializer, ChatResponseSerializer, ErrorResponseSerializer

logger = logging.getLogger(__name__)


@swagger_auto_schema(
    method='post',
    request_body=ChatQuerySerializer,
    responses={
        200: ChatResponseSerializer,
        400: ErrorResponseSerializer,
        500: ErrorResponseSerializer,
    },
    operation_description="Process a user query using the AI Assistant",
    operation_summary="Chat with AI Assistant",
    tags=['Chat API']
)
@api_view(['POST'])
@permission_classes([AllowAny])
def chat_query(request):
    """
    Process a user query using the AI Assistant.
    
    This endpoint accepts a user query and returns an AI-generated response.
    The AI can handle both FAQ questions and SQL queries based on the content.
    """
    try:
        # Validate input
        serializer = ChatQuerySerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {
                    'success': False,
                    'error': 'Invalid input',
                    'details': serializer.errors
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        query = serializer.validated_data['query']
        logger.info(f"Processing chat query: {query}")

        # Initialize AI Assistant
        ai_assistant = AIAssistant()
        
        # Test connection first
        if not ai_assistant.test_connection():
            logger.error("AI Assistant connection failed")
            return Response(
                {
                    'success': False,
                    'error': 'AI service is currently unavailable. Please try again later.',
                    'details': {'connection_error': 'Failed to connect to Gemini API'}
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        # Process the query
        result = ai_assistant.process_query(query)
        
        # Prepare response
        response_data = {
            'user_query': result.get('user_query', query),
            'response': result.get('response', ''),
            'response_type': result.get('response_type', 'unknown'),
            # 'classification': result.get('classification', {}),
            'success': True,
            'error': None
        }

        logger.info(f"Query processed successfully: {result.get('response_type')}")
        return Response(response_data, status=status.HTTP_200_OK)

    except Exception as e:
        logger.error(f"Error processing chat query: {str(e)}", exc_info=True)
        return Response(
            {
                'success': False,
                'error': 'An internal error occurred while processing your request.',
                'details': {'exception': str(e)}
            },
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@swagger_auto_schema(
    method='get',
    responses={
        200: openapi.Response(
            description="Health check response",
            schema=openapi.Schema(
                type=openapi.TYPE_OBJECT,
                properties={
                    'status': openapi.Schema(type=openapi.TYPE_STRING),
                    'ai_connection': openapi.Schema(type=openapi.TYPE_BOOLEAN),
                    'message': openapi.Schema(type=openapi.TYPE_STRING),
                }
            )
        )
    },
    operation_description="Check the health status of the AI Assistant service",
    operation_summary="Health Check",
    tags=['Health']
)
@api_view(['GET'])
@permission_classes([AllowAny])
def health_check(request):
    """
    Check the health status of the AI Assistant service.
    
    This endpoint verifies that the AI service is running and can connect to external APIs.
    """
    try:
        ai_assistant = AIAssistant()
        connection_status = ai_assistant.test_connection()
        
        return Response({
            'status': 'healthy' if connection_status else 'degraded',
            'ai_connection': connection_status,
            'message': 'AI Assistant is ready' if connection_status else 'AI service connection issues'
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return Response({
            'status': 'unhealthy',
            'ai_connection': False,
            'message': f'Service error: {str(e)}'
        }, status=status.HTTP_503_SERVICE_UNAVAILABLE)
