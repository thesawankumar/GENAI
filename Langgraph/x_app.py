import streamlit as st
from typing import TypedDict, Literal
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# =====================================================
# LLM
# =====================================================
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7
)

# =====================================================
# Structured Output Schema (Evaluator)
# =====================================================
class EvaluationSchema(BaseModel):
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str

structured_evaluator_llm = llm.with_structured_output(EvaluationSchema)

# =====================================================
# State
# =====================================================
class TweetState(TypedDict):
    topic: str
    tweet: str
    evaluation: str
    feedback: str
    feedback_history: list[str]
    iteration: int
    max_iteration: int

# =====================================================
# Nodes
# =====================================================
def generate_tweet(state: TweetState):
    response = llm.invoke(
        f"Write a short, funny, original X (Twitter) post about: {state['topic']}"
    )
    return {"tweet": response.content}
def evaluate_tweet(state: TweetState):

    messages = [
        SystemMessage(
            content="You evaluate tweets for quality."
        ),
        HumanMessage(
            content=f"""
Tweet:
"{state['tweet']}"

Is this tweet good enough to post on X?

Approve ONLY if it is:
- original
- funny or clever
- short and punchy
- feels shareable
- not a Q&A or setup-punchline joke

Otherwise mark it as needs_improvement.

Respond ONLY in structured format:
evaluation: approved | needs_improvement
feedback: one short paragraph
"""
        )
    ]

    response = structured_evaluator_llm.invoke(
        messages,
        config={"tool_choice": "required"}
    )

    return {
        "evaluation": response.evaluation,
        "feedback": response.feedback,
        "feedback_history": [response.feedback],
    }

def improve_tweet(state: TweetState):
    response = llm.invoke(
        f"""
Improve the following tweet based on feedback:

Tweet:
"{state['tweet']}"

Feedback:
{state['feedback']}

Rewrite the tweet to be better, funnier, and more punchy.
"""
    )
    return {"tweet": response.content}

def increment_iteration(state: TweetState):
    return {"iteration": state["iteration"] + 1}

def should_continue(state: TweetState):
    if state["evaluation"] == "approved":
        return END
    if state["iteration"] >= state["max_iteration"]:
        return END
    return "improve_tweet"

# =====================================================
# LangGraph
# =====================================================
graph = StateGraph(TweetState)

graph.add_node("generate_tweet", generate_tweet)
graph.add_node("evaluate_tweet", evaluate_tweet)
graph.add_node("improve_tweet", improve_tweet)
graph.add_node("increment_iteration", increment_iteration)

graph.add_edge(START, "generate_tweet")
graph.add_edge("generate_tweet", "evaluate_tweet")

graph.add_conditional_edges("evaluate_tweet", should_continue)

graph.add_edge("improve_tweet", "increment_iteration")
graph.add_edge("increment_iteration", "evaluate_tweet")

workflow = graph.compile()

# =====================================================
# Streamlit UI
# =====================================================
st.set_page_config(page_title="AI X Post Generator", layout="centered")

st.title("🐦 AI X Post Generator (LangGraph + Groq)")

topic = st.text_input("Enter topic for X post")
max_iter = st.slider("Max improvement attempts", 1, 10, 5)

if st.button("Generate X Post"):
    if not topic.strip():
        st.warning("Please enter a topic")
    else:
        with st.spinner("Generating X post..."):
            result = workflow.invoke({
                "topic": topic,
                "tweet": "",
                "evaluation": "",
                "feedback": "",
                "feedback_history": [],
                "iteration": 1,
                "max_iteration": max_iter
            })

        st.subheader("✅ Final X Post")
        st.success(result["tweet"])

        st.subheader("🧠 Evaluation Result")
        st.write(result["evaluation"])

        st.subheader("📝 Final Feedback")
        st.write(result["feedback"])

        st.subheader("📜 Feedback History")
        for i, fb in enumerate(result["feedback_history"], 1):
            st.write(f"{i}. {fb}")
