#Generate and explain a joke on a topic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template = 'Write a joke about {topic} ',
    input_variables = ['topic']
)

prompt2  = PromptTemplate(
    template = 'Explain the joke \n {joke}',
    input_variables = ['joke'] #As there is only one expected input, no need to worry about naming. 
                                #Whatever is output of first model, will be input of second model if only one o/p and i/p
)

model = ChatOpenAI(model='gpt-5.1')

parser = StrOutputParser()

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

print(chain.invoke({'topic':'AI'}))