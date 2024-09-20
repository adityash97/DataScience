# When we wanted to create our own use case,custom searches instead of generic searches.
import streamlit as st
from dotenv import load_dotenv
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

load_dotenv()

st.title(
    """
         Using PromptTemplate for specific/custom searches/result.
         
         """
)

query = st.text_input("Enter any celebrity name to know about")

llm = OpenAI(temperature=0.8)
# Creating Prompt Template
sample_template = PromptTemplate(
    input_variables=["name"], template="Tell me about {name}"
)
"""
Chains allow you to connect multiple tasks that need to be executed in a sequence. For example, you can first generate a prompt, then use an LLM to process it, and finally store the response in memory.
"""
chain = LLMChain(llm=llm, prompt=sample_template, verbose=True)

if query:
    result = chain.run(query)
    st.write(result)
