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




"""Get user query"""



"""Output to frontend"""
