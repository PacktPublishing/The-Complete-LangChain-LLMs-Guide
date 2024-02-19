import os 
import openai 
from dotenv import find_dotenv, load_dotenv
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage



load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

llm_model = "gpt-3.5-turbo"


prompt = "How old is the Universe"

messages = [HumanMessage(content=prompt)]

llm = OpenAI(temperature=0.7)
chat_model = ChatOpenAI(temperature=0.7)


# print(llm.predict("What is the weather in WA DC"))
# print("==========")

print(chat_model.predict_messages(messages).content)
# print(chat_model.predict("What is the weather in WA DC"))




