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

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")

# dataset
# movie_data_set = pd.read_csv('./sample_100mb.csv')
# movie_data_set.drop(columns=['Unnamed: 0'],inplace=True)

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
    

# df.head().to_string(index=False)



"""Get user query"""



"""Output to frontend"""
