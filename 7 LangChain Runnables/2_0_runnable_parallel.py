#Generate a tweet and linkedin post on a given topic parallely
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template = 'Generate a LinkedIn post about {topic}',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template = 'Generate a Twitter post about {topic}',
    input_variables = ['topic']
)

model = ChatOpenAI()
parser = StrOutputParser()

parallel_chain = RunnableParallel( #Takes Input as dictionary and returns Output as dictionary
    {
        'tweet': prompt1 | model | parser,
        'linkedin': prompt2 | model | parser
    }
)

print(parallel_chain.invoke({'topic':'AI'}))

parallel_chain.get_graph().print_ascii()