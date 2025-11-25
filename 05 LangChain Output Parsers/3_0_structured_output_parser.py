from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'google/gemma-2-2b-it',  #"TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

#1st promt -> detailed report
template = PromptTemplate(
    template = 'Give me 5 facts about {topic} \n {format_instruction}',
    input_variables = ['topic'],
    partial_variables = {'format_instruction': parser.get_format_instructions()}
)

print(template.invoke({'topic':'black hole'}))
# Output:
# text='Give me 5 facts about black hole \n The output should be a markdown code snippet formatted in the following schema, including the leading and trailing "```json" and "```":\n\n```json\n{\n\t"fact_1": string  // Fact 1 about the topic\n\t"fact_2": string  // Fact 2 about the topic\n\t"fact_3": string  // Fact 3 about the topic\n}\n```'

chain = template | model | parser

result = chain.invoke({'topic':'black hole'}) #If there are no input variables, send a empty dictionary

print(result) 

#Disadvantage -> No data validation