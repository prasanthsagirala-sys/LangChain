from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

print('hello')

# llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation"
# )

# model = ChatHuggingFace(llm=llm)

# result = model.invoke("What is the capital of India")

# print(result.content)

model = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-Nemo-Base-2407",
        max_new_tokens=100
)

# model = ChatHuggingFace(llm=model)

result = model.invoke("What is the capital of India?")

print(result)