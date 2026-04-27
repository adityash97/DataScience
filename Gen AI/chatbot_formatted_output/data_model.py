from pydantic import BaseModel

class ResumeDataModel(BaseModel):
    name: str
    email: str
    phone: str
    summary: str
    experience: list
    education: list
    skills: list