# Customer Feedback -> check if Pos or Neg Sentiment -> Reply based on Sentiment
#from re import template
from typing import Literal
#from langchain_core.prompts.string import PromptTemplateFormat
from langchain_openai import ChatOpenAI
#from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI(model = 'gpt-5.1')

str_parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment: Literal['Positive', 'Negative'] = Field(description='Sentiment of feedback')

pydantic_parser = PydanticOutputParser(pydantic_object = Feedback)

prompt1 = PromptTemplate(
    template = 'Classify the sentiment of following feedback text. \n Feedback -> {feedback} \n\n {format_instruction} ',
    input_variables = ['feedback'],
    partial_variables = {'format_instruction':pydantic_parser.get_format_instructions()}
)

classifier_chain = prompt1 | model | pydantic_parser 

result = classifier_chain.invoke({'feedback':'The product is really good!'})

print(result) #sentiment='Positive'

prompt2 = PromptTemplate(
    template = 'Write an appropriate response to this positive feedback \n {feedback}',
    input_variables = ['feedback']
)

prompt3 = PromptTemplate(
    template = 'Write an appropriate response to this negative feedback \n {feedback}',
    input_variables = ['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment=='Positive', prompt2 | model | str_parser), #Condition 1
    (lambda x:x.sentiment=='Negative', prompt3 | model | str_parser), #Condition 2
    RunnableLambda(lambda x: "Could not find the sentiment") #Default
)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback':"The product is amazing!"})

print(result)

chain.get_graph().print_ascii()

'''
Thank you so much for your positive feedback—it really means a lot. I’m glad you had a good experience, and I appreciate you taking the time to share it. If there’s anything more I can help with or improve on, I’d be happy to hear it.
    +-------------+      
    | PromptInput |
    +-------------+
            *
            *
            *
   +----------------+
   | PromptTemplate |
   +----------------+
            *
            *
            *
     +------------+
     | ChatOpenAI |
     +------------+
            *
            *
            *
+----------------------+
| PydanticOutputParser |
+----------------------+
            *
            *
            *
       +--------+
       | Branch |
       +--------+
            *
            *
            *
    +--------------+
    | BranchOutput |
    +--------------+
'''