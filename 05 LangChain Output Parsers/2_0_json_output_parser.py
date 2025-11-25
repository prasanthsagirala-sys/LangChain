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
    template = 'Give me the name, age and city of a fictional person \n {format_instruction}',
    input_variables = [],
    partial_variables = {'format_instruction': parser.get_format_instructions()}
)

prompt = template.format() #format for static prompts, invoke for dynamic prompt

print(prompt)
# Output:
# Give me the name, age and city of a fictional person 
#  Return a JSON object.

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result) #{'name': 'Amelia Croft', 'age': '32', 'city': 'Seattle'}

chain = template | model | parser

result = chain.invoke({}) #If there are no input variables, send a empty dictionary

print(result) #{'name': 'Anika Sharma', 'age': 32, 'city': 'Tokyo, Japan'}

#Datatype for age is not fixed. So seeing once as int and once as str, i.e., no schema format