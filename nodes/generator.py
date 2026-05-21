from groq import Groq
from langchain_core.messages import AIMessage, HumanMessage
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

FALLBACK_TEXT = "I don't have enough information from the provided documents."
MEMORY_WINDOW = 6  # last 3 human+AI pairs


def _build_history(messages) -> list[dict]:
    role_map = {HumanMessage: "user", AIMessage: "assistant"}
    history = []
    for m in messages:
        if type(m) in role_map:
            history.append({"role": role_map[type(m)], "content": m.content})
    return history


def generator(state):
    all_messages = state["messages"]
    query = all_messages[-1].content
    docs = state.get("_docs", [])

    prior = all_messages[:-1]
    history = _build_history(prior[-MEMORY_WINDOW:])

    if docs:
        context = "\n\n".join(docs)
        print("\n=== FINAL CONTEXT ===")
        print(context[:1000])
        system_prompt = (
            "You are a grounded RAG assistant. "
            "Answer using the provided context. "
            "You also have access to the recent conversation history — use it to answer follow-up or meta questions about what was said earlier. "
            "Paraphrase and synthesize evidence; handle synonyms. "
            "If evidence is partial, answer with what is known and end with: \"Uncertain: <what is missing>\". "
            "Only say you lack information if the context is completely unrelated AND history has no answer."
        )
        user_content = f"Context:\n{context}\n\nQuestion:\n{query}"
    else:
        system_prompt = (
            "You are a helpful assistant with memory of the recent conversation. "
            "No document context was retrieved. "
            "Answer the user's question using only the conversation history above. "
            f"If the answer is not in the history either, say: \"{FALLBACK_TEXT}\""
        )
        user_content = query

    messages = [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": user_content}
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70B-versatile",
        messages=messages
    )

    answer = response.choices[0].message.content

    error_trigger = bool(docs) and len(answer.strip()) < 20
    if error_trigger:
        retry_messages = [
            {"role": "system", "content": "Do not refuse. Use only context and give a concise best-effort answer with one uncertainty line if needed."},
            *history,
            {"role": "user", "content": user_content}
        ]
        retry = client.chat.completions.create(
            model="llama-3.3-70B-versatile",
            messages=retry_messages
        )
        answer = retry.choices[0].message.content

    return {"messages": [AIMessage(content=answer)]}
