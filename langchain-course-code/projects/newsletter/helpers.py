import os
from dotenv import find_dotenv, load_dotenv
import openai
import json
import requests
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.utilities import GoogleSerperAPIWrapper

openai.api_key = os.getenv("OPENAI_API_KEY")
SERP_API_KEY = os.getenv("SERPER_API_KEY")

load_dotenv(find_dotenv())

embeddings = OpenAIEmbeddings()


# 1. Serp request to get list of relevant articles
def search_serp(query):
    search = GoogleSerperAPIWrapper(k=5, type="search")
    response_json = search.results(query)
    
    print(f"Response=====>, {response_json}")
    
    return response_json

# 2. llm to choose the best articles, and return urls
def pick_best_articles_urls(response_json, query):
    # turn json to string
    response_str = json.dumps(response_json)
    
    # create llm to choose best articles
    llm = ChatOpenAI(temperature=0.7)
    template = """ 
      You are a world class journalist, researcher, tech, Software Engineer, Developer and a online course creator
      , you are amazing at finding the most interesting and relevant, useful articles in certain topics.
      
      QUERY RESPONSE:{response_str}
      
      Above is the list of search results for the query {query}.
      
      Please choose the best 3 articles from the list and return ONLY an array of the urls.  
      Do not include anything else -
      return ONLY an array of the urls. 
      Also make sure the articles are recent and not too old.
      If the file, or URL is invalid, show www.google.com.
    """
    prompt_template = PromptTemplate(
        input_variables=["response_str", "query"],
        template= template
    )
    article_chooser_chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        verbose=True
    )
    urls = article_chooser_chain.run(response_str=response_str,
                                     query=query)
    
    # Convert string to list
    url_list = json.loads(urls)
    #print(url_list)
    return url_list

# 3. Get content for each article from urls and make summaries
def extract_content_from_urls(urls):
    # use unstructuredURLLoader
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    docs = text_splitter.split_documents(data)
    db = FAISS.from_documents(docs, embeddings) # if libmagic issues: https://github.com/Yelp/elastalert/issues/1927
    
    return db 

# 4. summarize the articles...
def summarizer(db, query, k=4):
    
    docs = db.similarity_search(query, k=k)
    
    # Join the content of the page_content attribute from eac document...
    docs_page_content = " ".join([d.page_content for d in docs])
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=.7)
    template = """
       {docs}
        As a world class journalist, researcher, article, newsletter and blog writer, 
        you will summarize the text above in order to create a 
        newsletter around {query}.
        This newsletter will be sent as an email.  The format is going to be like
        Tim Ferriss' "5-Bullet Friday" newsletter.
        
        Please follow all of the following guidelines:
        1/ Make sure the content is engaging, informative with good data
        2/ Make sure the conent is not too long, it should be the size of a nice newsletter bullet point and summary
        3/ The content should address the {query} topic very well
        4/ The content needs to be good and informative
        5/ The content needs to be written in a way that is easy to read, digest and understand
        6/ The content needs to give the audience actinable advice & insights including resouces and links if necessary
        
        SUMMARY:
    """
    prompt_template = PromptTemplate(input_variables=["docs", "query"],
                                     template=template)
    
    summarizer_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
    
    response = summarizer_chain.run(docs=docs_page_content, query=query)
    
    return response.replace("\n", "") # reponse, docs

# 5. Turn summarization into newsletter (or article...)
def generate_newsletter(summaries, query):
    summaries_str = str(summaries)
    llm = ChatOpenAI(model="gpt-3.5-turbo", 
                     temperature=.7)
    template = """
    {summaries_str}
        As a world class journalist, researcher, article, newsletter and blog writer, 
        you'll use the text above as the context about {query}
        to write an excellent newsletter to be sent to subscribers about {query}.
        
        This newsletter will be sent as an email.  The format is going to be like
        Tim Ferriss' "5-Bullet Friday" newsletter.
        
        Make sure to write it informally - no "Dear" or any other formalities.  Start the newsletter with
        `Hi All!
          Here is your weekly dose of the Tech Newsletter, a list of what I find interesting
          and worth and exploring.`
          
        Make sure to also write a backstory about the topic - make it personal, engaging and lighthearted before
        going into the meat of the newsletter.
        
        Please follow all of the following guidelines:
        1/ Make sure the content is engaging, informative with good data
        2/ Make sure the conent is not too long, it should be the size of a nice newsletter bullet point and summary
        3/ The content should address the {query} topic very well
        4/ The content needs to be good and informative
        5/ The content needs to be written in a way that is easy to read, digest and understand
        6/ The content needs to give the audience actinable advice & insights including resouces and links if necessary.
        
        If there are books, or products involved, make sure to add amazon links to the products or just a link placeholder.
        
        As a signoff, write a clever quote related to learning, general wisdom, living a good life.  Be creative with this one - and then,
        Sign with "Paulo 
          - Learner and Teacher"
        
        NEWSLETTER-->:
    """
    prompt_template = PromptTemplate(input_variables=["summaries_str", "query"], 
                                     template=template)
    news_letter_chain = LLMChain(
        llm=llm,
        prompt=prompt_template,
        verbose=True)
    news_letter = news_letter_chain.predict(
        summaries_str=summaries_str,
        query=query)
    
    return news_letter

