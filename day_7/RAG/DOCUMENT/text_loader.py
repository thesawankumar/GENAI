# from langchain_community.document_loaders import TextLoader
# from langchain_openai import ChatOpenAI
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import PromptTemplate
# from dotenv import load_dotenv

# load_dotenv()

# model = ChatOpenAI()

# prompt = PromptTemplate(
#     template='Write a summary for the following poem - \n {poem}',
#     input_variables=['poem']
# )

# parser = StrOutputParser()

# loader = TextLoader('cricket.txt', encoding='utf-8')

# docs = loader.load()

# print(type(docs))

# print(len(docs))

# print(docs[0].page_content)

# print(docs[0].metadata)

# chain = prompt | model | parser

# print(chain.invoke({'poem':docs[0].page_content}))


from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv, find_dotenv

# Load HF token
load_dotenv(find_dotenv())

# -------- HF CHAT MODEL --------
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=300,
    temperature=0.7
)
model = ChatHuggingFace(llm=llm)

# -------- PROMPT --------
prompt = PromptTemplate(
    template="Write a concise summary for the following poem:\n{poem}",
    input_variables=["poem"]
)

# -------- PARSER --------
parser = StrOutputParser()

# -------- DOCUMENT LOADER --------
loader = TextLoader("cricket.txt", encoding="utf-8")
docs = loader.load()

# Debug prints (same as tumhara)
print(type(docs))               # list
print(len(docs))                # number of documents
print(docs[0].page_content)     # text content
print(docs[0].metadata)         # metadata

# -------- CHAIN --------
chain = prompt | model | parser

# -------- RUN --------
result = chain.invoke({"poem": docs[0].page_content})
print(result)
