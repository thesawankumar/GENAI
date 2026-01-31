from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    streaming=True
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState) -> ChatState:
    # state["messages"] already contains FULL history
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# ✅ Create checkpointer
checkpoint_saver = InMemorySaver()

# ✅ Pass checkpointer HERE (important)
graph = StateGraph(ChatState, checkpointer=checkpoint_saver)

graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# ✅ compile WITHOUT arguments
chatbot = graph.compile()


