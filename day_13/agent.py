from langchain_community.chat_models import ChatOllama
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
import requests

# -------- LLM (OLLAMA) --------
llm = ChatOllama(
    model="mistral",
    temperature=0
)

# -------- Tools --------
search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """Get current weather for a city"""
    try:
        return requests.get(
            f"https://wttr.in/{city}?format=3",
            timeout=5
        ).text
    except Exception as e:
        return str(e)

# -------- Prompt --------
prompt = hub.pull("hwchase17/react")

# -------- Agent --------
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True
)

# -------- Run --------
response = agent_executor.invoke({
        "input": "Find the capital of Madhya Pradesh and tell its current weather"
    })
print(response)
