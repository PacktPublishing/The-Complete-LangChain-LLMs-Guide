import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain.llms import OpenAI
from langchain.agents import Tool, initialize_agent, load_tools
from langchain import SerpAPIWrapper



## Must: pip install google-search-results
load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")
SERP_API_KEY = os.getenv("SERPAPI_API_KEY") # must get the api key and add to .env go to https://serpapi.com/


#==== Using OpenAI Chat API =======
llm_model = "gpt-3.5-turbo"
llm = OpenAI(temperature=0.0)

search = SerpAPIWrapper(serpapi_api_key=SERP_API_KEY)


# tools
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="google search"
    )
]

# initialize our agent
self_ask_with_search = initialize_agent(
    tools,
    llm,
    agent='self-ask-with-search',
    verbose=True
)

query = "who has travelled the most: Justin Timberlake, Alicia Keys, or Jason Mraz?"
result = self_ask_with_search(query)


