# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from dotenv import load_dotenv
# from langchain.schema.runnable import RunnableSequence, RunnableParallel

# load_dotenv()

# prompt1 = PromptTemplate(
#     template='Generate a tweet about {topic}',
#     input_variables=['topic']
# )

# prompt2 = PromptTemplate(
#     template='Generate a Linkedin post about {topic}',
#     input_variables=['topic']
# )

# model = ChatOpenAI()

# parser = StrOutputParser()

# parallel_chain = RunnableParallel({
#     'tweet': RunnableSequence(prompt1, model, parser),
#     'linkedin': RunnableSequence(prompt2, model, parser)
# })

# result = parallel_chain.invoke({'topic':'AI'})

# print(result['tweet'])
# print(result['linkedin'])

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
# -------- PROMPT 1 --------
prompt1 = PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=["topic"]
)
# -------- PROMPT 2 --------
prompt2 = PromptTemplate(
    template="Generate a Linkedin post about {topic}",
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
# -------- RUNNABLE PARALLEL --------
parallel_chain = RunnableParallel({
    "tweet": RunnableSequence(prompt1, model, parser),
    "linkedin": RunnableSequence(prompt2, model, parser)
})
result = parallel_chain.invoke({"topic": "AI"})
print("\n========== TWEET ==========\n")
print(result["tweet"])

print("\n======= LINKEDIN POST =======\n")
print(result["linkedin"])

