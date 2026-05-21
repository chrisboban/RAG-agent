from graph import build_graph
from langchain_core.messages import HumanMessage


def main():
    # Build LangGraph
    graph = build_graph()

    # Initialize state with empty messages
    state = {
        "messages": []
    }

    print("Chat started (type 'exit' to quit)\n")

    while True:
        # Take user input
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat...")
            break

        # Append user message to state
        
        state["messages"].append(
            HumanMessage(content=user_input)
        )

        # Invoke graph
        state = graph.invoke(state)

        # Get last assistant message
        last_message = state["messages"][-1].content

        print("\n\nBot:", last_message)
        print()


if __name__ == "__main__":
    main()