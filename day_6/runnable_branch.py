# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from dotenv import load_dotenv
# from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch, RunnableLambda

# load_dotenv()

# prompt1 = PromptTemplate(
#     template='Write a detailed report on {topic}',
#     input_variables=['topic']
# )

# prompt2 = PromptTemplate(
#     template='Summarize the following text \n {text}',
#     input_variables=['text']
# )

# model = ChatOpenAI()

# parser = StrOutputParser()

# report_gen_chain = prompt1 | model | parser

# branch_chain = RunnableBranch(
#     (lambda x: len(x.split())>300, prompt2 | model | parser),
#     RunnablePassthrough()
# )

# final_chain = RunnableSequence(report_gen_chain, branch_chain)

# print(final_chain.invoke({'topic':'Russia vs Ukraine'}))


from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough,RunnableBranch, RunnableLambda
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
# -------- PROMPT 1 --------
prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)
# -------- PROMPT 2 --------
prompt2 = PromptTemplate(
    template="Summarize the following text:\n{text}",
    input_variables=["text"]
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
# -------- REPORT GENERATION CHAIN --------
report_gen_chain = prompt1 | model | parser
# -------- BRANCH CHAIN --------
branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300, prompt2 | model | parser),
    RunnablePassthrough()
)
# -------- FINAL CHAIN --------
final_chain = RunnableSequence(report_gen_chain, branch_chain)
print(final_chain.invoke({"topic": "Russia vs Ukraine"}))
