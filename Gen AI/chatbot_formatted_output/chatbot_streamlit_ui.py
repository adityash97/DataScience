import streamlit as st
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from data_model import ResumeDataModel
import os

load_dotenv()

mistral_api_key = os.getenv("MISTRAL_API_KEY")

st.title("Resume Parser")

resume_text = st.text_area("Enter Resume Text", height=300)

if st.button("Extract Data"):
    if resume_text.strip():

        prompt = f"""
        You are a helpful assistant that extracts structured information from resumes.
        Please provide the following details:
        name, email, phone, summary, experience, education, and skills.

        Here is the resume text:
        {resume_text}
        """

        llm = ChatMistralAI(
            model="mistral-small-latest",
            api_key=mistral_api_key,
            temperature=0
        )

        structured_output = llm.with_structured_output(ResumeDataModel)

        try:
            response = structured_output.invoke(prompt)

            response_json = response.model_dump_json()

            resume_data = ResumeDataModel.model_validate_json(response_json)

            st.subheader("Extracted Resume Data")
            st.json(resume_data.model_dump())

        except Exception as e:
            st.error(f"Error validating structured output: {e}")

    else:
        st.warning("Please enter resume text.")