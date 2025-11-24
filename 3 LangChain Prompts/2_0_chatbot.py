from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash', temperature = 0.5)

while(True): #This loop doesn't remember previous conversations
    user_input = input("You: ")
    if user_input == 'exit':
        break 
    result = model.invoke(user_input)
    print("AI: ", result.content)

