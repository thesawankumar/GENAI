# from langchain_community.document_loaders import WebBaseLoader
# from langchain_openai import ChatOpenAI
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import PromptTemplate
# from dotenv import load_dotenv

# load_dotenv()

# model = ChatOpenAI()

# prompt = PromptTemplate(
#     template='Answer the following question \n {question} from the following text - \n {text}',
#     input_variables=['question','text']
# )

# parser = StrOutputParser()

# url = 'https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421'
# loader = WebBaseLoader(url)

# docs = loader.load()


# chain = prompt | model | parser

# print(chain.invoke({'question':'What is the product that we are talking about?', 'text':docs[0].page_content}))

from langchain_community.document_loaders import WebBaseLoader
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
    template="Answer the following question:\n{question}\nfrom the following text:\n{text}",
    input_variables=["question", "text"]
)
# -------- PARSER --------
parser = StrOutputParser()
# -------- LOADER --------
url = "https://www.flipkart.com/apple-macbook-air-m2-16-gb-256-gb-ssd-macos-sequoia-mc7x4hn-a/p/itmdc5308fa78421"
loader = WebBaseLoader(url)
docs = loader.load()
# -------- CHAINING --------
chain = prompt | model | parser
print(
    chain.invoke(
        {
            "question": "What is the product that we are talking about?",
            "text": docs[0].page_content,
        }
    )
)