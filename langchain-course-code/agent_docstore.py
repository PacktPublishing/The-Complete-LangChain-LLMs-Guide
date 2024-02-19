import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain.llms import OpenAI
from langchain.agents import Tool, initialize_agent, load_tools


from langchain import Wikipedia
from langchain.agents.react.base import DocstoreExplorer


load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

#==== Using OpenAI Chat API =======
llm_model = "gpt-3.5-turbo"
llm = OpenAI(temperature=0.0)

docstore = DocstoreExplorer(Wikipedia())
tools = [
    Tool(
        name="Search",
        func=docstore.search,
        description="search wikipedia"
    ),
    Tool(
        name="Lookup",
        func=docstore.lookup,
        description="lookup a term in wikipedia"
    )
]

# initialize our agent
docstore_agent = initialize_agent(
    tools,
    llm,
    agent="react-docstore",
    verbose=True,
    max_iterations=4
)

query = "What were Einstein's main beliefs?"
result = docstore_agent.run(query)
# print(docstore_agent.agent.llm_chain.prompt.template)


