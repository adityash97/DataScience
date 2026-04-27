import streamlit as st
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import json
import os

load_dotenv()

mistral_api_key = os.getenv("MISTRAL_API_KEY")

sysMessage = "You are a funny assistant with a good memory of previous conversations. You can also use emojis in your responses."

llm = ChatMistralAI(
    model="mistral-small-latest",
    api_key=mistral_api_key,
    temperature=0.7
)

st.title("Mistral Chatbot")
st.write("Type 'exit' to stop the conversation and save the chat history.\n")

if "init" not in st.session_state:
    st.session_state.init = False

if "history" not in st.session_state:
    st.session_state.history = []
    systemMessage = SystemMessage(content=sysMessage)
    st.session_state.history.append(systemMessage)

systemMessage = st.session_state.history[0]

user_input = st.text_input("You :")

if st.button("Send"):

    if user_input.strip() != "":

        humanMessage = HumanMessage(content=user_input)
        message = [systemMessage, humanMessage]

        if humanMessage.content.lower() == "exit":

            save_history = []

            for msg in st.session_state.history:
                save_history.append({
                    "role": msg.__class__.__name__,
                    "content": msg.content
                })

            with open("mistral_chat_history2.json", "w") as f:
                json.dump(save_history, f, indent=4)

            st.write("Chat history saved.")

        else:

            if st.session_state.history and st.session_state.init:
                message = st.session_state.history + message

            response = llm.invoke(message)

            st.session_state.init = True

            st.session_state.history.append(humanMessage)
            st.session_state.history.append(
                AIMessage(content=response.content)
            )

            st.write("Mistral :", response.content)

