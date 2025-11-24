from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'google/gemma-2-2b-it',  #"TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

parser = JsonOutputParser()

#1st promt -> detailed report
template = PromptTemplate(
    template = 'Give me 5 facts about {topic} \n {format_instruction}',
    input_variables = ['topic'],
    partial_variables = {'format_instruction': parser.get_format_instructions()}
)


chain = template | model | parser

result = chain.invoke({'topic':'black hole'}) #If there are no input variables, send a empty dictionary

print(result) #Output will be very unstructured