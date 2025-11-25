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
{'original_summary': 'AI or artificial intelligence is a rapidly advancing field of technology that aims to create machines capable of performing tasks that would typically require human intelligence. This includes tasks such as speech recognition, decision making, problem-solving, and learning. AI has the potential to revolutionize various industries by increasing efficiency, accuracy, and productivity. However, there are also ethical concerns surrounding AI, particularly regarding privacy, bias, and the potential loss of jobs to automation. Overall, AI holds great promise for the future but also requires careful consideration and regulation to ensure it is used responsibly and ethically.', 
'words_in_original_summary': 96, 
'new_summary': 'AI or artificial intelligence is a rapidly advancing field of technology that aims to create machines capable of performing tasks that would typically require human intelligence. This includes tasks such as speech recognition, decision making, problem-solving, and learning. AI has the potential to revolutionize various industries by increasing efficiency, accuracy, and productivity. However, there are also ethical concerns surrounding AI, particularly regarding privacy, bias, and the potential loss of jobs to automation. Overall, AI holds great promise for the future but also requires careful consideration and regulation to ensure it is used responsibly and ethically.'}
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
                  +------------------------------------+
                  | Parallel<summary,count_words>Input |
                  +------------------------------------+
                               **         **
                             **             **
                            *                 *
                  +-------------+          +--------+
                  | Passthrough |          | Lambda |
                  +-------------+          +--------+
                               **         **
                                 **     **
                                   *   *
                  +-------------------------------------+
                  | Parallel<summary,count_words>Output |
                  +-------------------------------------+
                                     *
                                     *
                                     *
+-----------------------------------------------------------------------+
| Parallel<original_summary,words_in_original_summary,new_summary>Input |
+-----------------------------------------------------------------------+
                          ***        *        ***
                      ****           *           ****
                    **               *               **
            +--------+          +--------+          +--------+
            | Lambda |          | Lambda |          | Branch |
            +--------+****      +--------+       ***+--------+
                          ***        *        ***
                             ****    *    ****
                                 **  *  **
+------------------------------------------------------------------------+
| Parallel<original_summary,words_in_original_summary,new_summary>Output |
+------------------------------------------------------------------------+
'''