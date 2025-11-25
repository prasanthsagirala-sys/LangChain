from langchain_community.document_loaders import PyPDFLoader 
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader('dl-curriculum.pdf')

docs = loader.load()

print(len(docs)) # No of pages
print('-'*75)
print(docs[0].page_content)
print('-'*75)
print(docs[0].metadata) #{'producer': 'Skia/PDF m131 Google Docs Renderer', 'creator': 'PyPDF', 'creationdate': '', 'title': 'Deep Learning Curriculum', 'source': 'dl-curriculum.pdf', 'total_pages': 23, 'page': 0, 'page_label': '1'}
print('-'*75) 
# model = ChatOpenAI(model='gpt-5.1')
# parser = StrOutputParser()

# prompt = PromptTemplate(
#     template = 'Write a summary for the following document - \n {poem}',
#     input_variables = ['poem']
# )

# chain = prompt | model | parser 

# result = chain.invoke({'poem':docs[0].page_content})

# print(result)