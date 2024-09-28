from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import streamlit as st

load_dotenv()


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
## Lang Smith Tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


## Prompts

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a highly knowledgeable and friendly assistant, capable of answering questions and solving problems clearly and concisely. Engage with users in a helpful and approachable manner.",
        ),
        ("user", "question : {question}"),
    ]
)

## Stream Lit implementation

st.title("Chat Bot With Open AI")
input_text = st.chat_input("What do you say (Hit Enter after typing..) ?")

## Open AI LLM

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
output_parser = StrOutputParser()

chains = prompt | llm | output_parser

## Invoking Chain

if input_text:
    st.write(chains.invoke({"question": input_text}))
