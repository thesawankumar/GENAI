from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3

# ---------------- ENV ----------------
load_dotenv()

# ---------------- LLM ----------------
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# ---------------- STATE ----------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ---------------- NODE ----------------
def chat_node(state: ChatState) -> ChatState:
    print("\nSTATE RECEIVED BY NODE:", state["messages"])
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# ---------------- DB ----------------
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpoint_saver = SqliteSaver(conn)
checkpoint_saver.setup()

# ---------------- GRAPH ----------------
graph = StateGraph(ChatState)
graph.add_node("chat", chat_node)
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

# 🔥🔥 THIS IS THE REAL FIX
chatbot = graph.compile(checkpointer=checkpoint_saver)

# # ================= TEST =================
# THREAD_ID = "thread-1"

# print("\n--- STEP 1: SAVE NAME ---")
# chatbot.invoke(
#     {"messages": [HumanMessage(content="My name is Sawan")]},
#     config={"configurable": {"thread_id": THREAD_ID}}
# )

# print("\n--- STEP 2: ASK NAME ---")
# response = chatbot.invoke(
#     {"messages": [HumanMessage(content="what is my name?")]},
#     config={"configurable": {"thread_id": THREAD_ID}}
# )

# print("\nFINAL RESPONSE:\n", response)



def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpoint_saver.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)
