import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

template = "Explain {topic} in simple terms."

prompt = PromptTemplate(
    template=template,
    input_variables=["topic"]
)

final_prompt = prompt.format(topic="AI")
print(final_prompt)