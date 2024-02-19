import os
import streamlit as st 
from helpers import *




def main():
    st.set_page_config(page_title="Researcher...", 
                       page_icon=":parrot:", 
                       layout="wide")
    
    st.header("Generate a Newsletter :parrot:")
    query = st.text_input("Enter a topic...")
    
    if query:
        print(query)
        st.write(query)
        with st.spinner(f"Generating newsletter for {query}"):
            #st.write("Generating newsletter for: ", query)
            
            search_results = search_serp(query=query)
            urls = pick_best_articles_urls(response_json=search_results, query=query)
            data = extract_content_from_urls(urls)
            summaries = summarizer(data, query)
            newsletter_thread = generate_newsletter(summaries, query)
            
            with st.expander("Search Results"):
                st.info(search_results)
            with st.expander("Best URLs"):
                st.info(urls)
            with st.expander("Data"):
                # iterate through data in the FAISS db and return the similiarity search data to show!
                data_raw = " ".join(d.page_content for d in data.similarity_search(query,k=4))
                st.info(data_raw)
            with st.expander("Summaries"):
                st.info(summaries)
            with st.expander("Newsletter:"):
                st.info(newsletter_thread)
        st.success("Done!")
    # query = "Flutter development news"
    # resp = search_serp(query=query)
    # urls = pick_best_articles_urls(response_json=resp, query=query)
    # data = extract_content_from_urls(urls=urls)
    # summaries = summarizer(db=data, query=query)
    # newsletter = generate_newsletter(summaries=summaries, query=query)
    # print(newsletter)



#Invoking main function
if __name__ == '__main__':
    main()