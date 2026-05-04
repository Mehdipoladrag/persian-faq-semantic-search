from django.urls import path
from django.views.decorators.csrf import csrf_exempt
from . import views

app_name = 'rag_app'

urlpatterns = [
    path('', views.chat_ui, name='chat_ui'),
    path('ask/', csrf_exempt(views.ask_api), name='ask_api'),
]