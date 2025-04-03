#recommend a stock of nvidia
#chatbot should contact agents where 1st agent will be ai agenst which will be collectimng the details of stiock
#sennd agenst: information from news
#interact with llm model
#come with recommendation

from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
#BEST PART ABOUT PHIDATA IT HAS LOT OIF INTERACTIOON WITH LOT OF TOOLS

import openai

import os
from dotenv import load_dotenv
load_dotenv()
#openai.api_key=os.getenv("OPENAI_API_KEY")



os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

os.environ["PHI_API_KEY"]=os.getenv("PHI_API_KEY")


#call tools inside any agenst

from phi.tools.duckduckgo import DuckDuckGo

#duckduckgo enables an agent to search the web for imforation
#every tool we are going to use, backened it will be a LLM model
#that LLM model will get some input from the tool that we will use

#whenever a query is done , it will hit the tool and then tool will use LLM and output will get generated
#webseach agent 
#llama-3.3-70b-versatile

#llama3-groq-70b-8192-tool-use-preview   depreciated: https://console.groq.com/docs/deprecations
web_search_agent=Agent(
    name="We search agent",
    role="Search the web for the information",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True


)





#WE GAVE PERSON A TASK, EXPLORE NVIDEA, HE IS EXPLORING MULTIPLE SOURCE TO COME UP WITH CONCLUSION

#FINANCIAL AGENT

finance_agent=Agent(
    name="Finance AI Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True,
                      company_news=True),
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,

)



multi_ai_agent=Agent(
    team=[web_search_agent,finance_agent],
    instructions=["Always include sources","Use table to display the data"],
    show_tool_calls=True,
    markdown=True,
)

#initiate the multiai agent

multi_ai_agent.print_response("Summarize and list recommendation and share the latest news for NVDA", stream=True)