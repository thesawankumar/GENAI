# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from dotenv import load_dotenv
# from langchain.schema.runnable import RunnableSequence

# load_dotenv()

# prompt1 = PromptTemplate(
#     template='Write a joke about {topic}',
#     input_variables=['topic']
# )

# model = ChatOpenAI()

# parser = StrOutputParser()

# prompt2 = PromptTemplate(
#     template='Explain the following joke - {text}',
#     input_variables=['text']
# )

# chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

# print(chain.invoke({'topic':'AI'}))

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# -------- PROMPT 1 --------
prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

# -------- HF CHAT MODEL --------
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    max_new_tokens=200,
    temperature=0.7
)
model = ChatHuggingFace(llm=llm)

# -------- PARSER --------
parser = StrOutputParser()

# -------- PROMPT 2 --------
prompt2 = PromptTemplate(
    template="Explain the following joke:\n{text}",
    input_variables=["text"]
)

# -------- RUNNABLE SEQUENCE --------
chain = RunnableSequence(
    prompt1,
    model,
    parser,
    prompt2,
    model,
    parser
)

# -------- RUN --------
result = chain.invoke({"topic": "AI"})
print(result)
