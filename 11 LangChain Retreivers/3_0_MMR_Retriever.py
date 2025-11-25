from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Sample documents
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

emebedding_model = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=emebedding_model
)

retriever = vectorstore.as_retriever(
    search_type = 'mmr', #This enables the MMR
    search_kwargs={'k':3, 'lambda_mult':0.5} #K is top results, lambda_mult[0 to 1] : 1 means normal search, 0 means very divergent results
)

query = 'What is lanchain?'
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f'\n--- Result {i+1} ---')
    print(doc.page_content)
