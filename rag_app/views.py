"""
Views Module - Handle HTTP requests and responses for chat interface
"""

import json
from typing import Dict, Any

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .rag_utils import get_rag_chain


class ChatView:
    """
    Base class for chat-related views.
    Handles UI rendering and API endpoints for Q&A.
    """

    @staticmethod
    def render_chat_ui(request) -> render:
        """
        Render the chat interface HTML page.

        Args:
            request: HTTP request object

        Returns:
            Rendered HTML response
        """
        return render(request, "rag_app/chat.html")

    @staticmethod
    @csrf_exempt
    @require_http_methods(["POST"])
    def handle_ask_api(request) -> JsonResponse:
        """
        Process user question and return RAG-generated answer.

        Args:
            request: HTTP POST request with JSON body containing 'query'

        Returns:
            JsonResponse with 'answer' and 'sources' fields
        """
        try:
            # Parse request body
            data: Dict[str, Any] = json.loads(request.body)
            query: str = data.get("query", "").strip()

            # Validate input
            if not query:
                return JsonResponse(
                    {"error": "Empty query. Please provide a question."}, status=400
                )

            # Get RAG chain and generate answer
            rag_chain = get_rag_chain()
            result = rag_chain.ask(query)

            # Return response
            return JsonResponse(
                {"answer": result["answer"], "sources": result["sources"]}
            )

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON format"}, status=400)
        except Exception as error:
            return JsonResponse({"error": f"Server error: {str(error)}"}, status=500)


# Module-level convenience functions (Django URL patterns expect functions)
def chat_ui(request):
    """Convenience function for chat UI view"""
    return ChatView.render_chat_ui(request)


def ask_api(request):
    """Convenience function for ask API endpoint"""
    return ChatView.handle_ask_api(request)
