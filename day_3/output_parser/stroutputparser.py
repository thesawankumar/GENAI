from dotenv import load_dotenv,find_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

load_dotenv(find_dotenv())


# -------- CHAT-COMPATIBLE MODEL --------
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",  # ✅ chat/instruct model
    task="text-generation",
    max_new_tokens=600,
    temperature=0.7
)

model = ChatHuggingFace(llm=llm)

# -------- PROMPT 1: DETAILED REPORT --------
template1 = PromptTemplate(
    template="Write a detailed report on {topic}.",
    input_variables=["topic"]
)

# -------- PROMPT 2: SUMMARY --------
template2 = PromptTemplate(
    template="Write a 5 line summary of the following text:\n{text}",
    input_variables=["text"]
)

# -------- STEP 1 --------
prompt1 = template1.invoke({"topic": "black hole"})
result = model.invoke(prompt1)

print("===== DETAILED REPORT =====")
print(result.content)

# -------- STEP 2 --------
prompt2 = template2.invoke({"text": result.content})
result2 = model.invoke(prompt2)

print("\n===== SUMMARY =====")
print(result2.content)
