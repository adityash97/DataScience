# Create a search bar through with user will interact with Open AI API.
# from langchain.llms.openai import OpenAI
from langchain_openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

st.title("""
          Open AI implemented with LanngChain
          """)
query = st.text_input("Enter Your Query")

## Initilize OPEN AI LLM Model
llm = OpenAI(temperature=0.8)

if query:
    st.write(llm(query))