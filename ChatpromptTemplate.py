import os
from dotenv import load_dotenv
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

model= ChatOpenAI(model="gpt-4o-mini", temperature=0.2
 )
chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant who explains things clearly."),
        ("human", "who is {person}?"),
        ("assistant", "I explain who {person} is without mentioning the name {person}."),

    ]
)
print(type(chat_prompt_template))

formatted_prompt = chat_prompt_template.format_messages(person="Mahatama Gandhi")
#format_messages is used to fill in the template with the user-provided person. Because until you call .format_messages(), this is just a template, not a usable prompt.

print(formatted_prompt)
print(type(formatted_prompt))

response = model.invoke(formatted_prompt)

print("The AI response is : \n", response.content)