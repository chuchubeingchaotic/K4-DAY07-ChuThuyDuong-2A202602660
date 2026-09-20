from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        if self.store.get_collection_size() == 0:
            return "Cơ sở tri thức hiện chưa có dữ liệu. Không tìm thấy thông tin để trả lời."

        results = self.store.search(question, top_k=top_k)
        if not results:
            return "Không tìm thấy thông tin phù hợp trong cơ sở tri thức để trả lời câu hỏi."

        context_lines: list[str] = []
        for i, res in enumerate(results, start=1):
            source = res.get("metadata", {}).get("source", res.get("id", f"doc_{i}"))
            context_lines.append(f"[{i}] (Nguồn: {source}):\n{res['content']}")

        context_str = "\n\n".join(context_lines)

        prompt = (
            f"Bạn là một trợ lý thông minh trả lời câu hỏi dựa trên cơ sở tri thức được cung cấp.\n"
            f"Ràng buộc:\n"
            f"- Chỉ sử dụng thông tin trong phần Ngữ cảnh dưới đây để trả lời câu hỏi.\n"
            f"- Trích dẫn rõ ràng số thứ tự nguồn [1], [2],... khi đưa ra thông tin.\n"
            f"- Nếu thông tin không có trong ngữ cảnh, hãy trả lời trung thực rằng không tìm thấy thông tin, tuyệt đối không bịa đặt.\n\n"
            f"Ngữ cảnh:\n{context_str}\n\n"
            f"Câu hỏi: {question}\n\n"
            f"Câu trả lời:"
        )

        return self.llm_fn(prompt)
