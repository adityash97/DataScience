from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.sequential import SimpleSequentialChain,SequentialChain
from langchain.chains.llm import LLMChain
import streamlit as st



load_dotenv()

def main():
    st.title("Search About Any Celebrity  (Langchain Chain Example)")
    person_name = st.text_input("Enter the person name")
    
    name_detail_prompt = PromptTemplate(input_variables=['name'],template="Tell me about celebrity {name}",output_key='person')
    dob_finder_prompt = PromptTemplate(input_variables=['dob'],)
    


if __name__ == '__main__':
    main()