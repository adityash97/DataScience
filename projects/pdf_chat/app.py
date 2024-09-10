import streamlit as st
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space


#sidebar
with st.sidebar:
    st.title('PDF Chat App(Using LLM)')
    st.markdown('''
                ## About 
                This app is LLM powered chatbot built using :
                -[StreamLit](https://streamlit.io/)
                -[LangChain](https://www.langchain.com/)
                -[OpenAI](https://openai.com/)LLM Model
                
                ''')
    add_vertical_space(5)

def main():
    st.header("Chat With PDF ᯓ  ✈︎")
    pdf = st.file_uploader("Upload Your PDF", type='pdf')
    if pdf:
        pdf_reader = PdfReader(pdf)
        text = ""
        for pages in pdf_reader.pages:
            text += pages.extract_text()

        
        st.write(text)
    
    
if __name__ == '__main__':
    main()
    
    