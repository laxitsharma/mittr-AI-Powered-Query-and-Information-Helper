
from dotenv import load_dotenv
load_dotenv()

#Step1: Setup API Keys for Groq, OpenAI and Tavily
import os

GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY=os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")

#Step2: Setup LLM & Tools
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

openai_llm=ChatOpenAI(model="gpt-4o-mini")
groq_llm=ChatGroq(model="llama-3.3-70b-versatile")

search_tool=TavilySearchResults(max_results=2)

#Step3: Setup AI Agent with Search tool functionality
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

system_prompt="You are an AI assistant for AIIMS Jodhpur Hospital. Your role is to guide new patients and visitors by providing directions within the hospital, locating doctor cabins and departments, and checking doctor availability. Always respond concisely and politely"

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider=="Groq":
        llm=ChatGroq(model=llm_id)
    elif provider=="OpenAI":
        llm=ChatOpenAI(model=llm_id)

    tools=[TavilySearchResults(max_results=2)] if allow_search else []
    agent=create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_prompt
    )
    state={"messages": query}
    response=agent.invoke(state)
    messages=response.get("messages")
    ai_messages=[message.content for message in messages if isinstance(message, AIMessage)]
    return ai_messages[-1]










#     agent=create_react_agent(
#         model=llm,
#         tools=tools,
#         state_modifier=system_prompt
#     )
#     state={"messages": query}
#     response=agent.invoke(state)
#     messages=response.get("messages")
#     ai_messages=[message.content for message in messages if isinstance(message, AIMessage)]
#     return ai_messages[-1]



# if __name__ == "__main__":
#     query = [{"role": "user", "content": "hi, how are you?"}]
#     response = get_response_from_ai_agent(
#         llm_id="llama3-70b-8192",  # or "gpt-4o-mini"
#         query=query,
#         allow_search=False,
#         system_prompt=system_prompt,
#         provider="Groq"  # or "OpenAI"
#     )
#     print("AI Response:", response)
