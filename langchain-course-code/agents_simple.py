import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain.llms import OpenAI
from langchain.agents import Tool, initialize_agent, load_tools



load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

#==== Using OpenAI Chat API =======
llm_model = "gpt-3.5-turbo"
llm = OpenAI(temperature=0.0)
 
# llm_math = LLMMathChain.from_llm(llm=llm)
# math_tool = Tool(
#     name="Calculator",
#     func=llm_math.run,
#     description="Useful for when you need to answer questions related to Math."
# )
tools = load_tools(
    ['llm-math'],
    llm=llm
)

print(tools[0].name, tools[0].description)

#ReAct framework = Reasoning and Action
agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3 # to avoid high bills from the LLM
)
query = "If James is currently 45 years old, how old will he be in 50 years? \
    If he has 4 kids and adopted 7 more, how many children does he have?"
result = agent(query)
print(result['output'])

# print(f" ChatGPT ::: {llm.predict('what is 3.1^2.1')}")



