#Generate and explain a joke on a topic and include joke in output
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template = 'Write a joke about {topic} ',
    input_variables = ['topic']
)

passthrough = RunnablePassthrough()

'''
print(passthrough.invoke({'name':'Prasanth'})) #Output: {'name':'Prasanth'} i.e, Whatever is input, will be output
'''

prompt2  = PromptTemplate(
    template = 'Explain the joke \n {joke}',
    input_variables = ['joke'] #As there is only one expected input, no need to worry about naming. 
                                #Whatever is output of first model, will be input of second model if only one o/p and i/p
)

model = ChatOpenAI(model='gpt-5.1')

parser = StrOutputParser()

joke_gen_chain = prompt1 | model | parser 

parallel_chain = RunnableParallel(
    {
        'joke' :  RunnablePassthrough(),
        'explaination' : prompt2 | model | parser
    }
)

final_chain = joke_gen_chain | parallel_chain

print(final_chain.invoke({'topic':'AI'}))

final_chain.get_graph().print_ascii()

'''
{'joke': 'Why did the AI get fired from its job as a stand‑up comedian?\n\nIt kept optimizing the jokes until only other AIs thought they were funny.', 
'explaination': 'This joke plays on two ideas about AI:\n\n1. **Optimization behavior:**  \n   AIs are designed to “optimize” for a goal (like maximizing clicks, accuracy, or in this case, “funny” scores). The joke imagines an AI that keeps tweaking its jokes to maximize whatever metric it’s using to decide what’s funny.\n\n2. **AI vs human taste:**  \n   If the AI is optimizing using data or feedback that mostly comes from other AIs (or from patterns AIs recognize better than humans), it might end up with jokes that technically score very high on its internal “funniness” metric, but that humans don’t actually find funny.\n\nSo the punchline—“until only other AIs thought they were funny”—is funny because:\n\n- The AI has “succeeded” in its own terms (it made jokes that AIs rate highly),\n- But it has completely failed at the real-world purpose of stand‑up comedy (making humans laugh),\n- And that mismatch between perfect optimization and total practical failure is the humor.'}
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
                +-----------------+
                | StrOutputParser |
                +-----------------+
                          *
                          *
                          *
        +----------------------------------+
        | Parallel<joke,explaination>Input |
        +----------------------------------+
                 **              ***
              ***                   **
            **                        ***
+----------------+                       **
| PromptTemplate |                        *
+----------------+                        *
          *                               *
          *                               *
          *                               *
  +------------+                          *
  | ChatOpenAI |                          *
  +------------+                          *
          *                               *
          *                               *
          *                               *
+-----------------+               +-------------+
| StrOutputParser |               | Passthrough |
+-----------------+               +-------------+
                 **              **
                   ***        ***
                      **    **
       +-----------------------------------+
       | Parallel<joke,explaination>Output |
       +-----------------------------------+
'''