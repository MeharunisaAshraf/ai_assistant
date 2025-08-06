from django.urls import path
from chatbot.views import chat_query, health_check

app_name = 'chatbot'

urlpatterns = [
    path('api/chat/', chat_query, name='chat_query'),
    path('api/health/', health_check, name='health_check'),
]