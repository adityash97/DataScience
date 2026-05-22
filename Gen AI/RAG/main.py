
# retrive on query
# pass the reterived item to llm
# show response

import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from embeddings import documents

load_dotenv()
mistral_api_key = os.getenv("MISTRAL_API_KEY")

prompt = ChatPromptTemplate([
('system',"""You are a helpful AI assistant. Your job is to answer the user query on the basis of provided context.
 If you don't have enough context to answer the user query then simply say : "I don't have enough context to answer this query."
 
 """),
("human","""
Here is the context on wich you have to answer the below query :
context : {context}
query : {query}
 """)

])

llm = ChatMistralAI( model="mistral-small-latest",
        api_key=mistral_api_key,
        temperature=0)

embedding = MistralAIEmbeddings(model="mistral-embed",api_key=mistral_api_key)

# vector = Chroma(collection_name="foo",persist_directory="embedding_store",embedding_function=embedding)
vector_store = Chroma(
    collection_name="foo",
    persist_directory="embedding_store",
    embedding_function=embedding
)



retriever = vector_store.as_retriever(search_type="mmr",search_kwargs={"k":2,"fetch_k": 2, "lambda_mult": 0.5})
query = input("Enter your query: ")

# query = "Enter your query: what is vector database?"

retrieved_documents = [d.page_content for d in retriever.invoke(query)]


# print("retieved_documents : ", retrieved_documents)

formatted_prompt = prompt.invoke({
    "context": retrieved_documents,
    "query": query
})

# print("\n\n\n",formatted_prompt)
response = llm.invoke(input=formatted_prompt)

print("*"*20,response.content,"*"*20)



