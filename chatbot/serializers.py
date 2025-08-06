from rest_framework import serializers


class ChatQuerySerializer(serializers.Serializer):
    """Serializer for incoming chat query requests"""
    query = serializers.CharField(
        max_length=1000,
        help_text="The user's query to process"
    )

    def validate_query(self, value):
        """Validate the query field"""
        if not value or not value.strip():
            raise serializers.ValidationError("Query cannot be empty")
        return value.strip()


class ChatResponseSerializer(serializers.Serializer):
    """Serializer for chat response"""
    user_query = serializers.CharField(help_text="The original user query")
    response = serializers.CharField(help_text="The AI assistant's response")
    response_type = serializers.CharField(help_text="Type of response (faq, sql_query, etc.)")
    # classification = serializers.DictField(help_text="Classification details from AI")
    success = serializers.BooleanField(default=True, help_text="Whether the request was successful")
    error = serializers.CharField(required=False, allow_null=True, help_text="Error message if any")


class ErrorResponseSerializer(serializers.Serializer):
    """Serializer for error responses"""
    success = serializers.BooleanField(default=False)
    error = serializers.CharField(help_text="Error message")
    details = serializers.DictField(required=False, help_text="Additional error details")
