import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

#==== Using OpenAI Chat API =======
llm_model = "gpt-3.5-turbo"

llm = ChatOpenAI(temperature=0.6, 
                 model=llm_model)

print(llm.predict("My name is Paulo. What is yours?"))
print(llm.predict("Great!  What's my name?")) # we have memory issues!

# How to solve llms memory issues?
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

conversation.predict(input="Hello there, I am Paulo")
conversation.predict(input="Why is the sky blue?")
conversation.predict(input="If phenomenon called Rayleigh didn't exist, what color would the sky be?")
conversation.predict(input="What's my name?")


print(f"Memory ===> {memory.buffer} <====")

# print(memory.load_memory_variables({}))








