"""Chat Bot Implementation Using  OLLAMA"""

from langchain.prompts.chat import ChatPromptTemplate  # why?
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms.ollama import Ollama
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# For Langsmith Dashboard
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


def main():
    st.title("Chat Bot Using Ollama")
    chatinput = st.chat_input("Enter Your Query here")

    prompts = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a highly knowledgeable and friendly assistant, capable of answering questions and solving problems"
                +"clearly and concisely. Engage with users in a helpful and approachable manner.",
            ),
            ("user", "query : {query}"),
        ]
    )

    llm = Ollama(model="llama3.1")

    ouput = StrOutputParser()

    chain = prompts | llm | ouput

    def stream_output(text, chunk=1):
        for i in range(0, len(text), chunk):
            yield text[i : i + chunk]

    if chatinput:
        print("Waiting for reply..")
        st.write("Your Query : ", chatinput)
        st.write_stream(stream_output(chain.invoke({"query": chatinput}), chunk=2))
        print("Please check the reply.")


if __name__ == "__main__":
    main()
