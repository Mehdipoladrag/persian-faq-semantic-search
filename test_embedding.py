import sys
sys.path.append('.')

from rag_app.rag_utils import get_rag_chain

# دریافت RAG chain (که دیتابیس را هم می‌سازد)
rag = get_rag_chain()

# تست جستجو مستقیم
query = "مراحل دریافت تسهیلات ودیعه رهن محل کار چگونه است؟"
results = rag.vector_store.search(query, top_k=3)

print(f"تعداد نتایج: {len(results)}")
for i, r in enumerate(results):
    print(f"\n--- نتیجه {i+1} (امتیاز: {r['score']:.4f}) ---")
    print(r['text'][:300])  # فقط 300 کاراکتر اول