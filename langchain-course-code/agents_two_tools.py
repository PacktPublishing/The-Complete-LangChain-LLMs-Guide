import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain.llms import OpenAI
from langchain.agents import Tool, initialize_agent, load_tools

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

#==== Using OpenAI Chat API =======
llm_model = "gpt-3.5-turbo"
llm = OpenAI(temperature=0.0)

# Second Generic Tool
prompt = PromptTemplate(
    input_variables=["query"],
    template="{query}"
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

# Initialize the LLM Tool
llm_tool = Tool(
    name="Language Model",
    func=llm_chain.run,
    description="Use this tool for general queries and logic"
)
 
tools = load_tools(
    ['llm-math'],
    llm=llm
)
tools.append(llm_tool) # adding the new tool to our tools list


#ReAct framework = Reasoning and Action
agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3 # to avoid high bills from the LLM
)
query = "if I have 54 eggs and Mary has 10, and 5 more people have 12 eggs each.  \
    How many eggs to we have in total?"
    
print(agent.agent.llm_chain.prompt.template)

result = agent(query)
print(result['output'])






