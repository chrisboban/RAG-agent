from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    route: str
    _docs: list[str]
