def router(state):
    query = state["messages"][-1].content.lower()

    keywords = ["what", "who", "why", "how", "about", "does"]

    if any(k in query for k in keywords):
        return {"route": "retrieve"}

    return {"route": "answer"}
