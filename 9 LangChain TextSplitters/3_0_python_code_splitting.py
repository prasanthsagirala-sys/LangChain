from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = '''
#Genarate a topic and summarize it if it is greater than 500 words else print as it is. 
from langchain_core.runnables.base import Runnable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch
from dotenv import load_dotenv
import re

load_dotenv()

model = ChatOpenAI()
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = 'Generate a summary on topic: {topic}',
    input_variables = ['topic']
)

topic_chain = prompt1 | model | parser

parallel_chain = RunnableParallel(
    {
        'summary' : RunnablePassthrough(),
        'count_words' : RunnableLambda(lambda text: len(re.findall(r'\b\w+\b', text)))
    }
)

prompt2 = PromptTemplate(
    template = 'Summarize the following text to less than 500 words: \n {summary}',
    input_variables = ['summary']
)

conditional_chain = RunnableBranch(
  (lambda x: x['count_words']>=500, prompt2 | model | parser),
  RunnableLambda(lambda x: x['summary'])
) 

parallel_chain_2 = RunnableParallel(
    {
        'original_summary': RunnableLambda(lambda x:x['summary']),
        'words_in_original_summary': RunnableLambda(lambda x:x['count_words']),
        'new_summary': conditional_chain 
    }
)

final_chain = topic_chain | parallel_chain | parallel_chain_2

print(final_chain.invoke({'topic':'AI'}))

final_chain.get_graph().print_ascii()
'''

splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.PYTHON, #Similarly for Markdown etc.
    chunk_size = 137, #Paste here and check for optimal chunk size https://chunkviz.up.railway.app/
    chunk_overlap = 0
)

chunks = splitter.split_text(text)

print(len(chunks))
print('-'*75)

for i in chunks:
    print(i)
    print('-'*75)