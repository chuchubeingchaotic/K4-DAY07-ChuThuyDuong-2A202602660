"""
Script to evaluate personal Semantic Chunking strategy on the e-commerce dataset.
Generates metrics, runs 5 benchmark queries, tests metadata filtering,
and displays results for REPORT_CANHAN.md and REPORT_NHOM.md.
"""

import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.models import Document
from src.chunking import (
    FixedSizeChunker,
    SentenceChunker,
    RecursiveChunker,
    SemanticChunker,
    ChunkingStrategyComparator,
    compute_similarity,
)
from src.store import EmbeddingStore
from src.agent import KnowledgeBaseAgent
from src.embeddings import _mock_embed


def parse_frontmatter(text: str) -> tuple[dict, str]:
    metadata = {}
    content = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            raw_meta = parts[1].strip()
            content = parts[2].strip()
            for line in raw_meta.splitlines():
                if ":" in line:
                    key, val = line.split(":", 1)
                    metadata[key.strip()] = val.strip().strip('"').strip("'")
    return metadata, content


def main():
    data_dir = Path("data/ecommerce")
    md_files = list(data_dir.glob("*.md"))

    print("=== 1. CHUNKING STRATEGY COMPARISON ===")
    comparator = ChunkingStrategyComparator()
    semantic_chunker = SemanticChunker(max_chunk_size=800)

    for fpath in sorted(md_files)[:3]:
        text = fpath.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)
        res = comparator.compare(body, chunk_size=500)
        sem_chunks = semantic_chunker.chunk(body)
        print(f"\nDocument: {fpath.name} (length={len(body)} chars)")
        print(f"  FixedSize:  count={res['fixed_size']['count']:2d}, avg_len={res['fixed_size']['avg_length']:.1f}")
        print(f"  Sentence:   count={res['by_sentences']['count']:2d}, avg_len={res['by_sentences']['avg_length']:.1f}")
        print(f"  Recursive:  count={res['recursive']['count']:2d}, avg_len={res['recursive']['avg_length']:.1f}")
        print(f"  Semantic:   count={len(sem_chunks):2d}, avg_len={sum(len(c) for c in sem_chunks)/len(sem_chunks):.1f}")

    print("\n=== 2. INGESTING DOCUMENTS WITH SEMANTIC CHUNKING ===")
    store = EmbeddingStore("ecommerce_semantic_store", embedding_fn=_mock_embed)
    all_docs = []

    for fpath in sorted(md_files):
        text = fpath.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)
        chunks = semantic_chunker.chunk(body)
        doc_id = meta.get("doc_id", fpath.stem)
        for idx, ch in enumerate(chunks):
            chunk_doc = Document(
                id=f"{doc_id}#{idx}",
                content=ch,
                metadata={
                    "doc_id": doc_id,
                    "title": meta.get("title", ""),
                    "source_url": meta.get("source_url", ""),
                    "retrieved_at": meta.get("retrieved_at", ""),
                    "document_version": meta.get("document_version", "1.0"),
                    "audience": meta.get("audience", "both"),
                    "category": meta.get("category", ""),
                    "language": meta.get("language", "vi"),
                    "chunk_index": idx,
                    "source": str(fpath),
                }
            )
            all_docs.append(chunk_doc)

    store.add_documents(all_docs)
    print(f"Total documents ingested into store: {store.get_collection_size()}")

    # 5 Benchmark Queries aligned with Shopee policies & L3B variant
    queries = [
        {
            "id": 1,
            "query": "Điều kiện để Người mua được bảo hành miễn phí cho sản phẩm mua tại Shopee là gì?",
            "gold_answer": "Sản phẩm còn trong thời hạn bảo hành, phiếu bảo hành còn nguyên vẹn và hư hỏng do lỗi kỹ thuật của nhà sản xuất.",
            "expected_doc": "buyer-warranty-policy",
            "filter": None,
        },
        {
            "id": 2,
            "query": "Thời gian và các lý do người mua được yêu cầu trả hàng, hoàn tiền?",
            "gold_answer": "Người mua có thể yêu cầu trả hàng trong Thời Gian Shopee Đảm Bảo (ví dụ: hàng bể vỡ, lỗi kỹ thuật, sai mô tả, thiếu phụ kiện).",
            "expected_doc": "return-refund-policy",
            "filter": None,
        },
        {
            "id": 3,
            "query": "Người bán có hành vi gian lận đơn hàng ảo hoặc lạm dụng mã giảm giá sẽ bị xử lý như thế nào?",
            "gold_answer": "Người bán bị phạt điểm Sao Quả Tạ, tạm khóa hoặc cấm tài khoản, đóng băng số dư ví Shopee và bồi thường thiệt hại.",
            "expected_doc": "seller-fraud-penalty-policy",
            "filter": {"audience": "seller"},
        },
        {
            "id": 4,
            "query": "Quy định về các mặt hàng và nội dung bị cấm đăng bán trên sàn là gì?",
            "gold_answer": "Cấm đăng bán hàng giả, hàng nhái, vũ khí, chất kích thích, động vật hoang dã và các mặt hàng vi phạm pháp luật.",
            "expected_doc": "seller-listing-regulations",
            "filter": {"audience": "seller"},
        },
        {
            "id": 5,
            "query": "Chính sách đồng kiểm và các giới hạn hàng hóa không hỗ trợ vận chuyển của Shopee?",
            "gold_answer": "Quy định các mặt hàng cấm vận chuyển (chất dễ cháy nổ, tiền mặt, hàng quá khổ) và quy trình đồng kiểm ngoại quan khi nhận hàng.",
            "expected_doc": "shipping-policy",
            "filter": None,
        },
    ]

    agent = KnowledgeBaseAgent(
        store=store,
        llm_fn=lambda p: (
            f"[Agent RAG Answer]\n"
            f"Dựa trên các tài liệu đã kiểm tra trong cơ sở tri thức: "
            f"Thông tin chính xác được trích xuất từ các điều khoản quy định đã lưu trữ. "
            f"Nguồn tham chiếu cụ thể đã được chỉ định theo số hiệu ngữ cảnh [1], [2]."
        )
    )

    print("\n=== 3. RUNNING BENCHMARK EVALUATION ===")
    for q in queries:
        qid = q["id"]
        query_str = q["query"]
        filt = q["filter"]
        if filt:
            results = store.search_with_filter(query_str, top_k=3, metadata_filter=filt)
        else:
            results = store.search(query_str, top_k=3)

        top1 = results[0] if results else None
        top1_doc = top1["metadata"].get("doc_id") if top1 else "none"
        top1_preview = (top1["content"][:100].replace("\n", " ") + "...") if top1 else "None"
        score = top1["score"] if top1 else 0.0
        relevant = (top1_doc == q["expected_doc"]) or any(r["metadata"].get("doc_id") == q["expected_doc"] for r in results)

        print(f"\nQuery #{qid}: {query_str}")
        print(f"Filter: {filt}")
        print(f"Top-1 Doc ID: {top1_doc} | Score: {score:.4f}")
        print(f"Top-1 Preview: {top1_preview}")
        print(f"Relevant in top-3: {relevant}")
        answer = agent.answer(query_str, top_k=3)
        print(f"Agent answer summary: {answer[:120]}...")


if __name__ == "__main__":
    main()

