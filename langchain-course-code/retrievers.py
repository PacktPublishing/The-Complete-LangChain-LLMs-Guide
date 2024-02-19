import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain.chat_models import ChatOpenAI



#!New Imports
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

#==== Using OpenAI Chat API =======
llm_model = "gpt-3.5-turbo"
embeddings = OpenAIEmbeddings()

llm = ChatOpenAI(temperature=0.0, model=llm_model) #changed to openAI


# 1. Load a pdf file
loader = PyPDFLoader("./data/react-paper.pdf")
docs = loader.load()

# 2. Split the document into chunks
# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)
splits = text_splitter.split_documents(docs)

# Install faiss vector store...or chroma! pip install chromadb
from langchain.vectorstores import Chroma
persist_directory = './data/db/chroma/'
# !rm -rf ./data/db/chroma  # remove old database files if any
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings, # openai embeddings
    persist_directory= persist_directory

)
vectorstore.persist() # save this for later usage!

## load the persisted db
vector_store = Chroma(persist_directory=persist_directory,
                      embedding_function=embeddings)


# make a retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
docs = retriever.get_relevant_documents("Tell me more about ReAct prompting")
# print(retriever.search_type)
print(docs[0].page_content)


# Make a chain to answer questions
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True,
    return_source_documents=True
    
)

## Cite sources - helper function to prettyfy responses
def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])

query = "tell me more about ReAct prompting"
llm_response = qa_chain(query)
print(process_llm_response(llm_response=llm_response))


