import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain 
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA






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

# Now we split the data into chunks
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(documents)

# create our vector db chromadb
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    persist_directory='./data'
)
vectordb.persist()

# Use RetrievalQA chain to get info from the vectorstore
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(search_kwargs={'k':3}),
    return_source_documents=True
)

result = qa_chain("whe did Rachel graduate?")
#results = qa_chain({'query': 'Who is the CV about?'}) # the other way of doing the same thing
print(result['result'])



