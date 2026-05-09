from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_classic.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain_classic.chains import ConversationChain

load_dotenv()
llm = ChatOpenAI(model="gpt-5-nano")
# chat_history = []
memory = ConversationSummaryBufferMemory(input_key="input", llm=llm, max_token_limit=100) # max_token_limit is the maximum number of tokens to keep in the conversation summary.
#ConverationChain is a chain that maintains a conversation history and can generate responses based on that history. 
conversation=ConversationChain(llm=llm, memory=memory, verbose=True)
# Example interaction
print(conversation.predict(input="Hi there!"))#predict method is used to generate a response based on the input provided.
print(conversation.predict(input="I want to know more about Agenti Ai."))
print(conversation.predict(input="Tell me how fast they are developing?"))
print(conversation.predict(input="Tell me about their future?"))
print(conversation.predict(input="What are some platforms they are using?"))