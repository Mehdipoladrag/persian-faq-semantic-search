import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .rag_utils import get_rag_chain


def chat_ui(request):
    return render(request, 'rag_app/chat.html')


@csrf_exempt
def ask_api(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            query = data.get("query", "").strip()
        except:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
    elif request.method == 'GET':
        query = request.GET.get("query", "").strip()
    else:
        return JsonResponse({"error": "Method not allowed"}, status=405)
    
    if not query:
        return JsonResponse({"error": "Empty query"}, status=400)
    
    try:
        rag_chain = get_rag_chain()
        result = rag_chain.ask(query)
        
        return JsonResponse({
            "answer": result["answer"],
            "sources": result["sources"]
        })
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)