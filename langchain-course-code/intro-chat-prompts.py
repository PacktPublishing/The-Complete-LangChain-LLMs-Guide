import os 
import openai 
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate




load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")


llm_model = "gpt-3.5-turbo"


# OpenAI Completion Endpoint
def get_completion(prompt, model=llm_model):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, 
    )
    return response.choices[0].message["content"]

# Translate text, review 
customer_review = """
 Your product is terrible!  I don't know how 
 you were able to get this to the market.
 I don't want this! Actually no one should want this.
 Seriously!  Give me money now!
 
"""
tone = """ Proper British English in a nice, warm, respectful tone """
language = "Turkish"

promp = f""" 
  Rewrite the following {customer_review} in {tone}, and then
  please translate the new review message into {language}.
"""

rewrite = get_completion(prompt=promp)

# print(rewrite)

# ====== Using LangChain & prompt templates - Still ChatAPI ====
chat_model = ChatOpenAI(temperature=0.7, 
                        model=llm_model)

template_string = """ 
 Translate the following text {customer_review}
 into italiano in a polite tone.
 And the company name is {company_name}
"""

prompt_template = ChatPromptTemplate.from_template(template_string)

translation_message = prompt_template.format_messages(
    customer_review = customer_review,
    company_name="Google"
)

response = chat_model(translation_message)
print(response.content)

