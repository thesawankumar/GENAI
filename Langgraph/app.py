import streamlit as st
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv(find_dotenv())

st.set_page_config(page_title="AI Review Reply Assistant", layout="centered")

# -------------------- LLM --------------------

model = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

# -------------------- Schemas --------------------

class SentimentSchema(BaseModel):
    sentiment: Literal["positive", "negative"]

class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"]
    tone: Literal["angry", "frustrated", "disappointed", "calm"]
    urgency: Literal["low", "medium", "high"]

structured_model = model.with_structured_output(SentimentSchema)
structured_model2 = model.with_structured_output(DiagnosisSchema)

# -------------------- State --------------------

class ReviewState(TypedDict):
    review: str
    sentiment: Literal["positive", "negative"]
    diagnosis: dict
    response: str

# -------------------- Nodes --------------------

def find_sentiment(state):
    sentiment = structured_model.invoke(
        f"Find sentiment of this review:\n{state['review']}"
    ).sentiment
    return {"sentiment": sentiment}

def check_sentiment(state):
    return "positive_response" if state["sentiment"] == "positive" else "run_diagnosis"

def positive_response(state):
    response = model.invoke(
        f"""Write a warm thank-you response for this review:
"{state['review']}"

Also ask the user to leave feedback."""
    ).content
    return {"response": response}

def run_diagnosis(state):
    diagnosis = structured_model2.invoke(
        f"Diagnose this review:\n{state['review']}"
    )
    return {"diagnosis": diagnosis.model_dump()}

def negative_response(state):
    d = state["diagnosis"]
    response = model.invoke(
        f"""User issue: {d['issue_type']}
Tone: {d['tone']}
Urgency: {d['urgency']}

Write an empathetic support reply."""
    ).content
    return {"response": response}

# -------------------- Graph --------------------

graph = StateGraph(ReviewState)
graph.add_node("find_sentiment", find_sentiment)
graph.add_node("positive_response", positive_response)
graph.add_node("run_diagnosis", run_diagnosis)
graph.add_node("negative_response", negative_response)

graph.add_edge(START, "find_sentiment")
graph.add_conditional_edges("find_sentiment", check_sentiment)
graph.add_edge("positive_response", END)
graph.add_edge("run_diagnosis", "negative_response")
graph.add_edge("negative_response", END)

workflow = graph.compile()

# -------------------- UI --------------------

st.title("🧠 AI Review Reply Generator")

review = st.text_area("Paste customer review here")

if st.button("Generate Reply") and review:
    with st.spinner("Analyzing review..."):
        result = workflow.invoke({"review": review})

    st.subheader("💬 AI Response")
    st.success(result["response"])
