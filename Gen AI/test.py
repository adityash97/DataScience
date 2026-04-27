from dotenv import load_dotenv
import os
import json
import typing

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
mistral_api_key = os.getenv("MISTRAL_API_KEY")
hugging_face_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")



# mistral ai access

from langchain_mistralai import ChatMistralAI
llm = ChatMistralAI(
    model="mistral-small-latest",
    api_key=mistral_api_key,
    temperature=0.7
)


response = llm.invoke("What is the capital of India?")

print(response)
content = {'response':response.content}

with open('mistral_response.json','w') as f:
    json.dump(content, f, indent=4)

