# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from langchain_huggingface import HuggingFacePipeline
# # from langchain_core.prompts import PromptTemplate

# model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model = AutoModelForCausalLM.from_pretrained(model_id)

# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=100,
#     temperature=0.1
# )

# llm = HuggingFacePipeline(pipeline=pipe)

# # prompt = PromptTemplate(
# #     input_variables=["question"],
# #     template="Answer clearly:\nQuestion: {question}"
# # )

# # chain = prompt | llm

# print(llm.invoke("What is the capital of India?"))




from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    temperature=0.1,
    return_full_text=False   # ✅ FIX 1
)

llm = HuggingFacePipeline(pipeline=pipe)

# ✅ FIX 2 (minimal token hint, NOT full prompt)
question = "What is the capital of India?\n<|assistant|>"

print(llm.invoke(question))
