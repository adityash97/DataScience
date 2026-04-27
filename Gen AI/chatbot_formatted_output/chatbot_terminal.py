from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
from data_model import ResumeDataModel
import os

load_dotenv()

mistral_api_key = os.getenv("MISTRAL_API_KEY")

print("Type 'exit' to quit the program. \n")
while True:
    resume_text = input("Enter the resume text or 'exit' to quit: ")
    if resume_text.lower() == 'exit':
        print("Exiting the program. Goodbye!")
        break   

    prompt = f"You are a helpful assistant that extracts structured information from resumes. Please provide the following details: name, email, phone, summary, experience, education, and skills.\
    Here is the resume text: {resume_text}"

    llm = ChatMistralAI(
        model="mistral-small-latest",
        api_key=mistral_api_key,
        temperature=0
    )

    structured_output = llm.with_structured_output(ResumeDataModel)
    response = structured_output.invoke(prompt) # it returns a pydantic model instance of ResumeDataModel
    response = ResumeDataModel.model_dump_json(response)  # Convert the response to JSON format

    try:
        resume_data = ResumeDataModel.model_validate_json(response)
        print("\nExtracted Resume Data:", resume_data)
    except Exception as e:
        print(f"Error validating structured output: {e}")
    print("____"*20)
    
