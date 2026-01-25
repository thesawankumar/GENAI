# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from dotenv import load_dotenv
# from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough

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

# joke_gen_chain = RunnableSequence(prompt1, model, parser)

# parallel_chain = RunnableParallel({
#     'joke': RunnablePassthrough(),
#     'explanation': RunnableSequence(prompt2, model, parser)
# })

# final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

# print(final_chain.invoke({'topic':'cricket'}))

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
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
# -------- JOKE GENERATION CHAIN --------
joke_gen_chain = RunnableSequence(prompt1, model, parser)
# -------- RUNNABLE PARALLEL --------
parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "explanation": RunnableSequence(prompt2, model, parser)
})
# -------- FINAL CHAIN --------
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
print(final_chain.invoke({"topic": "cricket"}))