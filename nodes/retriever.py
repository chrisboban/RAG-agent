from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import re

# Load once
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def _keyword_overlap_score(query: str, text: str) -> float:
    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0.0
    t_tokens = _tokenize(text)
    return len(q_tokens & t_tokens) / len(q_tokens)


def retriever(state):
    query = state["messages"][-1].content

    # Stage 1: semantic retrieval with scores
    scored_docs = vectorstore.similarity_search_with_score(query, k=20)

    # Stage 2: clean and rerank with lexical overlap
    ranked = []
    for doc, distance in scored_docs:
        text = doc.page_content.strip()

        if len(text) < 100:
            continue

        overlap = _keyword_overlap_score(query, text)
        score = overlap - (distance * 0.5)
        ranked.append((score, text))

    ranked.sort(key=lambda x: x[0], reverse=True)
    cleaned_docs = []
    for _, text in ranked[:6]:
        cleaned_docs.append(text)

    # Fallback: keep top semantic chunks if filters were too strict
    if not cleaned_docs:
        for doc, _ in scored_docs[:4]:
            text = doc.page_content.strip()
            if len(text) >= 80:
                cleaned_docs.append(text)

    if len(cleaned_docs) < 3:
        print("Weak retrieval: fewer than 3 strong chunks.")

    print("\n=== RETRIEVED DOCS ===")
    for i, d in enumerate(cleaned_docs):
        print(f"[{i}] {d[:200]}")

    return {
        "_docs": cleaned_docs
    }
