from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash', temperature = 0.5)

chat_history = []

while(True): #This loop remember previous conversations
    user_input = input("You: ")
    chat_history.append(user_input)
    if user_input == 'exit':
        break 
    result = model.invoke(chat_history) #Passing the whole conversation to model
    chat_history.append(result.content)
    print("AI: ", result.content)
print('-'*75)
print(chat_history) #

