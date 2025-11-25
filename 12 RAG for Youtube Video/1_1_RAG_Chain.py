from yt_transcript import get_yt_video_en_transcript

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

from dotenv import load_dotenv

load_dotenv()

video_id = "Gfr50f6ZBvo"

def border():
    print('-'*100)

#1a Get YT Video English Transcript
transcript = get_yt_video_en_transcript(video_id)

print("Transcript:", transcript[:100])
border()

#1b Text Splitting
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

# print(len(chunks))
# print(chunks[0])
# border()

#1c & 1d - Indexing (Embedding generation and storing in Vector)
embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')
vector_store = FAISS.from_documents(chunks, embeddings)


#2 Retriever
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':4})


#3 Augmentation
prompt = PromptTemplate(
    template = """
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

#4 Generation
llm = ChatOpenAI(model='gpt-5.1')

parser = StrOutputParser()

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

context_combiner = RunnableLambda(format_docs)

context_user_query_chain = RunnableParallel(
    {
        'question': RunnablePassthrough(),
        'context': retriever | context_combiner
    }
)


final_chain = context_user_query_chain | prompt | llm | parser

user_query = 'is there any discussion about Deepmind?'

result = final_chain.invoke(user_query)

print(result) 
#Output:
# Yes. The transcript discusses DeepMind, including its founding tenets: focusing on algorithmic advances like deep learning and reinforcement learning, drawing inspiration from human brain research (e.g., fMRI), leveraging commoditized compute and GPUs, and using mathematical and theoretical definitions of intelligence.
