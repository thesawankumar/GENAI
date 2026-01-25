# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnable import RunnableParallel, RunnableBranch, RunnableLambda
# from langchain_core.output_parsers import PydanticOutputParser
# from pydantic import BaseModel, Field
# from typing import Literal

# load_dotenv()

# model = ChatOpenAI()

# parser = StrOutputParser()

# class Feedback(BaseModel):

#     sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')

# parser2 = PydanticOutputParser(pydantic_object=Feedback)

# prompt1 = PromptTemplate(
#     template='Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}',
#     input_variables=['feedback'],
#     partial_variables={'format_instruction':parser2.get_format_instructions()}
# )

# classifier_chain = prompt1 | model | parser2

# prompt2 = PromptTemplate(
#     template='Write an appropriate response to this positive feedback \n {feedback}',
#     input_variables=['feedback']
# )

# prompt3 = PromptTemplate(
#     template='Write an appropriate response to this negative feedback \n {feedback}',
#     input_variables=['feedback']
# )

# branch_chain = RunnableBranch(
#     (lambda x:x.sentiment == 'positive', prompt2 | model | parser),
#     (lambda x:x.sentiment == 'negative', prompt3 | model | parser),
#     RunnableLambda(lambda x: "could not find sentiment")
# )

# chain = classifier_chain | branch_chain

# print(chain.invoke({'feedback': 'This is a beautiful phone'}))

# chain.get_graph().print_ascii()


from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv(find_dotenv())
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

class Feedback(BaseModel):

    sentiment: Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')
parser2 = PydanticOutputParser(pydantic_object=Feedback)
# -------- CLASSIFIER CHAIN --------
prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into postive or negative \n {feedback} \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction": parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2
# -------- BRANCH CHAIN --------
prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback \n {feedback}",
    input_variables=["feedback"]
)
prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback \n {feedback}",
    input_variables=["feedback"]
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | parser),
    (lambda x: x.sentiment == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)
# -------- FINAL CHAIN --------
chain = classifier_chain | branch_chain
# -------- RUN --------
print(chain.invoke({"feedback": "This is a beautiful phone"}))
chain.get_graph().print_ascii()
