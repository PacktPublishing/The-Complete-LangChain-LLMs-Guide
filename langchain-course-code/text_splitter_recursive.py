import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

# 1. CharacterTextSplitter
with open("./data/i-have-a-dream.txt") as paper:
    speech = paper.read()
    
text_splitter = RecursiveCharacterTextSplitter(
    
    chunk_size = 40,
    chunk_overlap = 12,
    length_function = len,
    add_start_index=True
)

docs = text_splitter.create_documents([speech])

# print(len(docs))
# print(f"Doc 1: {docs[0]}")
# print(f"Doc 2: {docs[1]}")

s1 = "abcdefghijklmnopqrstuvwxyz"
s = "Python can be easy to pick up whether you're a professional or a beginner."

text = text_splitter.split_text(s)
print(text)






