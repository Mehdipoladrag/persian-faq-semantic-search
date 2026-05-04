import logging
from typing import Dict, Any

from openai import OpenAI

logger = logging.getLogger(__name__)


class RAGChain:
    """
    RAG chain that combines document retrieval with LLM generation.
    """

    def __init__(self, vector_store: Any, config: Any) -> None:
        """
        Initialize RAG chain with vector store and LLM configuration.

        Args:
            vector_store: Vector database instance for document search
            config: Configuration object with LLM and retrieval settings
        """
        self.vector_store = vector_store
        self.llm_config = config.llm
        self.top_k = config.retrieval["top_k"]
        self.system_prompt = config.chat["system_prompt"]

        self.client = OpenAI(
            base_url=self.llm_config["base_url"],
            api_key=self.llm_config.get("api_key", "not-needed"),
        )

    def ask(self, query: str) -> Dict[str, Any]:
        """
        Process a user query and return generated answer based on retrieved documents.

        Args:
            query: User's question in Persian

        Returns:
            Dictionary with 'answer', 'sources', and 'retrieved_chunks' keys
        """
        retrieved_docs = self.vector_store.search(query, top_k=self.top_k)

        if not retrieved_docs:
            return {
                "answer": "هیچ سند مرتبطی یافت نشد. لطفاً سوال دیگری بپرسید.",
                "sources": [],
                "retrieved_chunks": [],
            }

        context = "\n\n".join(
            [f"{i+1}:\n{doc['text']}" for i, doc in enumerate(retrieved_docs)]
        )

        user_prompt = f"""اطلاعات موجود:
{context}

سوال: {query}

پاسخ را فقط بر اساس اطلاعات بالا بده. اگر پاسخ در اطلاعات نبود، بگو «اطلاعاتی ندارم»."""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.llm_config["model_name"],
                messages=messages,
                temperature=self.llm_config["temperature"],
                max_tokens=self.llm_config["max_tokens"],
            )
            answer = response.choices[0].message.content
        except Exception as error:
            logger.error(f"LLM call failed: {error}")
            answer = "خطا در ارتباط با مدل. لطفاً LM Studio را بررسی کنید."

        sources = list(
            set(
                [
                    doc["metadata"].get("source", "unknown")
                    for doc in retrieved_docs
                    if "source" in doc["metadata"]
                ]
            )
        )

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": retrieved_docs,
        }