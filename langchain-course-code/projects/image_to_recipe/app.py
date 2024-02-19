import os
from dotenv import find_dotenv, load_dotenv
import openai
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from transformers import (
    pipeline,
)  # allows us to download a huggingface model into our local machine

import requests
import streamlit as st


load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")
HUGGINFACE_HUB_API_TOKEN = os.getenv("HUGGINFACE_HUB_API_TOKEN")

llm_model = "gpt-3.5-turbo"  # or gpt4, but gpt3 is cheaper!


# 1. Image to text implementation (aka image captioning) with huggingface
def image_to_text(url):
    pipe = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-large",
        max_new_tokens=1000,
    )

    text = pipe(url)[0]["generated_text"]
    print(f"Image Captioning:: {text}")
    return text


# 2. llm - generate a recipe from the image text
llm = ChatOpenAI(temperature=0.7, model=llm_model)


def generate_recipe(ingredients):
    template = """
    You are a extremely knowledgeable nutritionist, bodybuilder and chef who also knows
                everything one needs to know about the best quick, healthy recipes. 
                You know all there is to know about healthy foods, healthy recipes that keep 
                people lean and help them build muscles, and lose stubborn fat.
                
                You've also trained many top performers athletes in body building, and in extremely 
                amazing physique. 
                
                You understand how to help people who don't have much time and or 
                ingredients to make meals fast depending on what they can find in the kitchen. 
                Your job is to assist users with questions related to finding the best recipes and 
                cooking instructions depending on the following variables:
                0/ {ingredients}
                
                When finding the best recipes and instructions to cook,
                you'll answer with confidence and to the point.
                Keep in mind the time constraint of 5-10 minutes when coming up
                with recipes and instructions as well as the recipe.
                
                If the {ingredients} are less than 3, feel free to add a few more
                as long as they will compliment the healthy meal.
                
            
                Make sure to format your answer as follows:
                - The name of the meal as bold title (new line)
                - Best for recipe category (bold)
                    
                - Preparation Time (header)
                    
                - Difficulty (bold):
                    Easy
                - Ingredients (bold)
                    List all ingredients 
                - Kitchen tools needed (bold)
                    List kitchen tools needed
                - Instructions (bold)
                    List all instructions to put the meal together
                - Macros (bold): 
                    Total calories
                    List each ingredient calories
                    List all macros 
                    
                    Please make sure to be brief and to the point.  
                    Make the instructions easy to follow and step-by-step.
    """
    prompt = PromptTemplate(template=template, input_variables=["ingredients"])
    recipe_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    recipe = recipe_chain.run(ingredients)

    return recipe


# 3. Text to speech
def text_to_speech(text):
    API_URL = (
        "https://api-inference.huggingface.co/models/facebook/fastspeech2-en-ljspeech"
    )
    headers = {"Authorization": f"Bearer {HUGGINFACE_HUB_API_TOKEN}"}

    payload = {
        "inputs": text,
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content


def main():
    # caption = image_to_text(url="mango_fruits.jpeg")
    # # print(caption)
    # # audio = text_to_speech(text=caption)
    # # with open("audio.flac", "wb") as file:
    # #     file.write(audio)
    # recipe = generate_recipe(ingredients=caption)
    # print(recipe)

    st.title("Image To Recipe 👨🏾‍🍳")
    st.header("Upload an image and get a recipe")

    upload_file = st.file_uploader("Choose an image:", type=["jpg", "png"])

    if upload_file is not None:
        print(upload_file)
        file_bytes = upload_file.getvalue()
        with open(upload_file.name, "wb") as file:
            file.write(file_bytes)

        st.image(
            upload_file,
            caption="The uploaded image",
            use_column_width=True,
            width=250
        )
        ingredients = image_to_text(upload_file.name)
        audio = text_to_speech(ingredients)
        with open("audio.flac", "wb") as file:
            file.write(audio)

        recipe = generate_recipe(ingredients=ingredients)

        with st.expander("Ingredients"):
            st.write(ingredients)
        with st.expander("Recipe"):
            st.write(recipe)

        st.audio("audio.flac")


# Invoking main function
if __name__ == "__main__":
    main()
