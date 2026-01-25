# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# load_dotenv()

# prompt1 = PromptTemplate(
#     template='Generate a detailed report on {topic}',
#     input_variables=['topic']
# )

# prompt2 = PromptTemplate(
#     template='Generate a 5 pointer summary from the following text \n {text}',
#     input_variables=['text']
# )

# model = ChatOpenAI()

# parser = StrOutputParser()

# chain = prompt1 | model | parser | prompt2 | model | parser

# result = chain.invoke({'topic': 'Unemployment in India'})

# print(result)

# chain.get_graph().print_ascii()


from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv(find_dotenv())
# -------- PROMPTS --------
prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template="Generate a 5 pointer summary from the following text \n {text}",
    input_variables=["text"]
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
chain = prompt1 | model | parser | prompt2 | model | parser
# -------- RUN --------
result = chain.invoke({"topic": "Unemployment in India"})
print(result)
chain.get_graph().print_ascii()