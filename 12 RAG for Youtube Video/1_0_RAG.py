from yt_transcript import get_yt_video_en_transcript

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

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

print(len(chunks))
print(chunks[0])
border()

#1c & 1d - Indexing (Embedding generation and storing in Vector)
embeddings = OpenAIEmbeddings(model = 'text-embedding-3-small')
vector_store = FAISS.from_documents(chunks, embeddings)

# print(vector_store.index_to_docstore_id)
# border()
# print(vector_store.get_by_ids(['932365cb-0b63-40fd-80c8-7408cf0bc9b8']))
# border()

#2 Retriever
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':4})
# query = 'What is DeepMind?'
# print(query)
# print(retriever.invoke(query))
# border()

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

question          = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs    = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({'context':context_text, 'question':question})

#4 Generation
llm = ChatOpenAI(model='gpt-5.1')
answer = llm.invoke(final_prompt)
border()
print(answer)
border()

