#Create a preprocessing function to clean a review before analysing sentiment of a review
#Generate and explain a joke on a topic and include joke in output and also include count of words in the joke
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
import re

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
        'explaination' : prompt2 | model | parser,
        'count_words' : RunnableLambda(lambda text: len(re.findall(r'\b\w+\b', text)))
    }
)

final_chain = joke_gen_chain | parallel_chain

print(final_chain.invoke({'topic':'AI'}))

final_chain.get_graph().print_ascii()

'''
{'joke': 'Why did the AI get kicked out of the comedy club?\n\nIt kept trying to optimize the punchline, but couldn’t find a human who liked the loss function.', 
'explaination': 'This joke mashes up stand‑up comedy with machine learning jargon.\n\nBreakdown:\n\n- **“Optimize the punchline”**  \n  In normal comedy, you “work on” or “refine” a joke. In machine learning, you “optimize” a model—mathematically adjusting it to perform better. So the AI is treating joke-writing like training a model.\n\n- **“Loss function”**  \n  In ML, the *loss function* measures how bad the model’s output is (how “wrong” it is). Training is about minimizing this loss.  \n  Here, the AI is basically saying: “I tried to make the punchline better by minimizing a loss function,” as if audience laughter could be captured by a mathematical error metric.\n\n- **“Couldn’t find a human who liked the loss function”**  \n  A loss function is a human-designed formula. If it doesn’t reflect what humans actually care about, the model will optimize the wrong thing.  \n  The joke is that no human audience member “likes” (or agrees with) the AI’s mathematical idea of what makes something funny—so even though it’s optimizing, the results are unfunny, and it gets kicked out of the club.\n\nSo the humor comes from:\n1. Treating comedy as a technical optimization problem.\n2. Using the very nerdy, unfunny concept of a “loss function” in the context of stand‑up comedy.\n3. Highlighting how AI can miss human taste, even while “optimizing” perfectly.', 
'count_words': 29}
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
              +----------------------------------------------+
              | Parallel<joke,explaination,count_words>Input |
              +----------------------------------------------+
                      ****            *           *****
                 *****                *                ****
              ***                     *                    *****
+----------------+                    *                         ***
| PromptTemplate |                    *                           *
+----------------+                    *                           *
          *                           *                           *
          *                           *                           *
          *                           *                           *
  +------------+                      *                           *
  | ChatOpenAI |                      *                           *
  +------------+                      *                           *
          *                           *                           *
          *                           *                           *
          *                           *                           *
+-----------------+           +-------------+               +--------+
| StrOutputParser |           | Passthrough |              *| Lambda |
+-----------------+***        +-------------+         ***** +--------+
                      ****            *           ****
                          *****       *      *****
                               ***    *   ***
             +-----------------------------------------------+
             | Parallel<joke,explaination,count_words>Output |
             +-----------------------------------------------+
'''