# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from dotenv import load_dotenv
# from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnablePassthrough, RunnableParallel

# load_dotenv()

# def word_count(text):
#     return len(text.split())

# prompt = PromptTemplate(
#     template='Write a joke about {topic}',
#     input_variables=['topic']
# )

# model = ChatOpenAI()

# parser = StrOutputParser()

# joke_gen_chain = RunnableSequence(prompt, model, parser)

# parallel_chain = RunnableParallel({
#     'joke': RunnablePassthrough(),
#     'word_count': RunnableLambda(word_count)
# })

# final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

# result = final_chain.invoke({'topic':'AI'})

# final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])

# print(final_result)


from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableLambda, RunnablePassthrough,RunnableParallel
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
# -------- WORD COUNT FUNCTION --------
def word_count(text):
    return len(text.split())
# -------- PROMPT --------
prompt = PromptTemplate(
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
# -------- JOKE GENERATION CHAIN --------
joke_gen_chain = RunnableSequence(prompt, model, parser)
# -------- PARALLEL CHAIN --------
parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "word_count": RunnableLambda(word_count)
})
# -------- FINAL CHAIN --------
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
# -------- RUN --------
result = final_chain.invoke({"topic": "AI"})
final_result = "{} \n word count - {}".format(result["joke"], result["word_count"])
print(final_result)