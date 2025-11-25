from langchain_community.document_loaders import TextLoader 
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader('cricket.txt', encoding = 'utf-8')

docs = loader.load()

print(docs) # [Document(metadata={'source': 'cricket.txt'}, page_content='Beneath the sun...')]
print('-'*75)
model = ChatOpenAI(model='gpt-5.1')
parser = StrOutputParser()

prompt = PromptTemplate(
    template = 'Write a summary for the following poen - \n {poem}',
    input_variables = ['poem']
)

chain = prompt | model | parser 

result = chain.invoke({'poem':docs[0].page_content})

print(result)