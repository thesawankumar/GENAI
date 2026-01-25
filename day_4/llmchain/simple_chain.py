# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv,find_dotenv
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# load_dotenv(find_dotenv())

# prompt = PromptTemplate(
#     template='Generate 5 interesting facts about {topic}',
#     input_variables=['topic']
# )

# model = ChatOpenAI()

# parser = StrOutputParser()

# chain = prompt | model | parser

# result = chain.invoke({'topic':'cricket'})

# print(result)

# chain.get_graph().print_ascii()





from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(find_dotenv())

# -------- PROMPT --------
prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=["topic"]
)

# -------- HF CHAT MODEL --------
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=300,
    temperature=0.7
)

model = ChatHuggingFace(llm=llm)

# -------- PARSER --------
parser = StrOutputParser()

# -------- CHAIN --------
chain = prompt | model | parser

# -------- RUN --------
result = chain.invoke({"topic": "cricket"})
print(result)

# -------- GRAPH --------
chain.get_graph().print_ascii()
