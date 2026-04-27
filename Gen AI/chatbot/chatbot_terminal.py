from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv
import json
import os

load_dotenv()
mistral_api_key = os.getenv('MISTRAL_API_KEY')
print("Type 'exit' to stop the conversation\n")
sysMessage = "You are a funny assistant with a good memory of previous conversations. You can also use emojis in your responses."


llm = ChatMistralAI(
    model="mistral-small-latest",
    api_key=mistral_api_key,
    temperature=0.7
)
init = False
history = [] # It could be a database 
systemMessage = SystemMessage(content = sysMessage) #set the context for the model.
history.append(systemMessage)
while True:
    humanMessage = input("You : ") # Taking user input
    if humanMessage.strip() == "":
        continue
    humanMessage = HumanMessage(content=humanMessage) # Taking user input
    message = [systemMessage,humanMessage]


    if humanMessage.content.lower() == "exit": # saving history
        save_history = []
        for msg in history:
            save_history.append({
                "role":msg.__class__.__name__,
                "content":msg.content
            })
        with open('mistral_chat_history.json','w') as f:
            json.dump(save_history, f, indent=4)
        break
    if history and init:
        message = history + message # adding history to the current message
    response = llm.invoke(message)
    init = True

    history.append(humanMessage)
    history.append(AIMessage(content = response.content))

    print("Mistral : ",response.content,"\n")



