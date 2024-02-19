import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain


load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

#==== Using OpenAI Chat API =======
llm_model = "gpt-3.5-turbo"
llm = ChatOpenAI(temperature=0.0, model=llm_model)

#### === packages to install ====
# pip install langchain pypdf openai chromadb tiktoken docx2txt

# load the pdf file
pf_loader = PyPDFLoader('./docs/RachelGreenCV.pdf')
documents = pf_loader.load()


#set up qa chain
chain = load_qa_chain(llm, verbose=True)
query = 'Where did Rachel go to school?'
response = chain.run(input_documents=documents,
                     question=query)

print(response)