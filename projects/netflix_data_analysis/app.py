# Stremlit Application(frontend)
"""
This project aims to replicate pandas.ai using Langgraph.
"""
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import pandas as pd
from langgraph.graph import StateGraph,START,END
from langchain_core.messages import HumanMessage
from typing import Optional,Type,Literal,TypedDict
from pydantic import BaseModel
from typing import List
from IPython.display import display,Image
import streamlit as st

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")

# dataset
movie_data_set = pd.read_csv('./sample_100mb.csv')
movie_data_set.drop(columns=['Unnamed: 0'],inplace=True)

"""get colums and other details"""


"""Create Graph"""
#state
class DataState(TypedDict):
    user_query : str # at invoke
    columns :List[str] # at invoke
    dataset_name : str # at invoke
    llm_response : str
    user_to_response: List[dict]
    error_message : str | None = None
    retry_on_error : int = 2

# Prompts
def process_user_query(state:DataState):
    return f"""
        You are an AI assistant for data analysis. Your task is to generate Python code directly without any additional text, formatting, or prefixes like "python" or "```". 

        Do not include imports, comments, or any other text. Assume the following:
        1. Pandas and NumPy are already installed and imported.
        2. The dataset is loaded into a DataFrame with the name: {state['dataset_name']}.
        3. The dataset has the following columns: {state['columns']}.

        Based on the user query: "{state['user_query']}", generate the exact Python code to execute the query on the dataset.
    """
def generate_error_prompt(state:DataState):
    return f"""
    The generated code caused the following error: {state['error_message']}.
    Please rectify the error and regenerate the correct Python code for the query: "{state['user_query']}".
    """
def summarize_prompt(state:DataState):
    return f"""
        You are an AI assistant. The user asked the following query: "{state['user_query']}".

        The response retrieved from the dataset is as follows:
        {state['user_to_response']}

        Please summarize this response in a concise and clear manner relevant to the user query.
    """ 
# Nodes
def generate_llm_response(state : DataState):
    print("generate_llm_response called")
    response = llm.invoke([HumanMessage(content=process_user_query(state=state))])
    # print("response : ",response)
    # response = "movie_data_set.hea()" # wrong query
    state['llm_response'] = response.content
    return state

def call_llm_again(state: DataState):
    print("call_llm_again called ")
    response = llm.invoke([HumanMessage(content=generate_error_prompt(state=state))])
    print("response again : ",response)
    # response = "movie_data_set.head()" # correct query
    state['llm_response'] = response.content
    return state


def summarize_response(state:DataState):
    print("summarize_response called ")
    if state['retry_on_error'] > 0:
        response  = llm.invoke([HumanMessage(content=summarize_prompt(state=state))])
        state['user_to_response'] = response.content
    else:
        state['user_to_response'] = f"Exhausted the limit of retry. Please write the code manually. The recent error is {state['error_message']}"
    return state


def fail_condition(state:DataState):
        
    if state['error_message']:
        if state['retry_on_error'] > 0:
            return 'call_llm_again'
        return 'summarize_response'
    return 'summarize_response'


        
def execute_query(state:DataState):
    try: 
        local_vars = {"movie_data_set": movie_data_set}
        exec(f"result = {state['llm_response']}", {}, local_vars)
        state['user_to_response']  = local_vars['result'].to_dict(orient='records')
        state['error_message'] =  None
        return state
        
    except Exception as e:
        print("LLm Response :",state['llm_response'] ,"\n\n")
        print("Error : ", e)
        state['error_message'] = e
        return state

# df.head().to_string(index=False)



"""Get user query"""

# Display a text input box for the user query
user_query = st.text_input("Enter your query:")

# Store the input in a variable and display it
if user_query:
    st.write("You entered:", user_query)

"""Output to frontend"""
