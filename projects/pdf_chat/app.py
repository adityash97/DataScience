import streamlit as st
import pickle
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms.openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

# sidebar
with st.sidebar:
    st.title("PDF Chat App(Using LLM)")
    st.markdown(
        """
                ## About 
                This app is LLM powered chatbot built using :
                -[StreamLit](https://streamlit.io/)
                -[LangChain](https://www.langchain.com/)
                -[OpenAI](https://openai.com/)LLM Model
                
                """
    )
    add_vertical_space(5)
load_dotenv()


def main():
    st.header("Chat With PDF ᯓ  ✈︎")
    pdf = st.file_uploader("Upload Your PDF", type="pdf")
    if pdf:
        pdf_reader = PdfReader(pdf)
        text = ""
        for pages in pdf_reader.pages:
            text += pages.extract_text()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        import os

        file_name = pdf.name[:-4]
        # if os.path.exists(f"{file_name}.pkl"):
        #     with open(f"{file_name}.pkl",'rb') as f:
        #         VectorStore = pickle.loads(f)
        # else:
        # import pdb;pdb.set_trace()
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        # file_name = pdf.name[:-4]
        # from joblib import dump
        # dump(VectorStore, f"{file_name}.pkl")
        # with open(f"{file_name}.pkl",'wb') as f:
        #     pickle.dump(VectorStore,f)
        st.write(chunks)
        query = st.text_input("Enter Your Query")
        if query:
            docs = VectorStore.similarity_search(
                query=query, k=3
            )  # get top matched document chunk
            # feed the chunk to LLM to get the response
            llm = OpenAI()  # model_name= gpt-3.5-turbo
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)


if __name__ == "__main__":
    main()
