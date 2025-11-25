from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash'
    , temperature = 0.5
    , max_completion_tokens = 10
    )

result = model.invoke('Write a 5 line poem on Solar System')

print(result.content)