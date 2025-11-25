from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI(model = 'gpt-5-mini')

#1st promt -> detailed report
template1 = PromptTemplate(
    template = 'Write a detailed report on {topic}',
    input_variables = ['topic']
)

#2nd prompt -> summarized report
template2 = PromptTemplate(
    template = 'Write a 5 line summary on following text. /n{text}',
    input_variables = ['text']
)

# prompt1 = template1.invoke({'topic':'black hole'})
# result1 = model.invoke(prompt1)
# prompt2 = template2.invoke({'text':result1.content})
# result2 = model.invoke(prompt2)
# print(result2.content)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser #Output will be just text

chain2 = template1 | model | template2 | model  #Output will be response with content,  response_metadata, 


result = chain.invoke({'topic':'black hole'})

print(result)