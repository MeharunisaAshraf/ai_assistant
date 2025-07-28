from rest_framework import serializers

class ChatQuerySerializer(serializers.Serializer):
    query = serializers.CharField(max_length=1000, help_text="User query to process")
    return_sql = serializers.BooleanField(default=False, help_text="Include SQL query in response for debugging")
    
    def validate_query(self, value):
        if not value.strip():
            raise serializers.ValidationError("Query cannot be empty")
        return value.strip()

class BatchChatQuerySerializer(serializers.Serializer):
    queries = serializers.ListField(
        child=serializers.CharField(max_length=1000),
        max_length=10,
        help_text="List of queries to process (max 10)"
    )
    return_sql = serializers.BooleanField(default=False)