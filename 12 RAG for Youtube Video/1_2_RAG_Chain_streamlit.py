from yt_transcript import get_yt_video_en_transcript
from youtube_transcript_api import NoTranscriptFound, CouldNotRetrieveTranscript

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate, prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch

from dotenv import load_dotenv
import streamlit as st 

load_dotenv()

def border():
    print('-'*100)

def get_retriever_for_video_id(video_id):
    print('Loading Retriever...')

    #1a Get YT Video English Transcript
    transcript = get_yt_video_en_transcript(video_id)
    if transcript is None:
        raise CouldNotRetrieveTranscript(video_id)
    print('Transcript:', transcript[:100],"...")

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
    print('Retriever ready')
    return retriever


#3 Augmentation
def get_context_question_prompt():
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
    return prompt

#4 Generation
def get_rag_chain_for_video_id(video_id):
    print('Creating RAG Chain...')
    try:
        retriever = get_retriever_for_video_id(video_id)
    except CouldNotRetrieveTranscript:
        return None, 'No Transcript found'

    llm = ChatOpenAI(model='gpt-5.1')

    parser = StrOutputParser()

    def format_docs(retrieved_docs):
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        return context_text

    context_combiner = RunnableLambda(format_docs)

    prompt = get_context_question_prompt()

    context_user_query_chain = RunnableParallel(
        {
            'question': RunnablePassthrough(),
            'context': retriever | context_combiner
        }
    )


    final_chain = context_user_query_chain | prompt | llm | parser

    print('RAG Chain Ready')
    border()

    return final_chain, None

def invoke_chain(chain, user_query):
    if chain is None: return {'user_query': user_query, 'response':'No data available'}
    result = chain.invoke(user_query)
    return result

# video_id = "J5_-l7WIO_w&t" #"Gfr50f6ZBvo"
# user_query = 'is there any discussion about Deepmind?'

# rag_chain = get_rag_chain_for_video_id(video_id)

# print(invoke_chain(rag_chain, user_query))
# border()

# print(invoke_chain(rag_chain, 'Summarize the video'))
# border()


# ---------------- Streamlit App ---------------- #

st.set_page_config(page_title="YouTube RAG Q&A", page_icon="🎬")
st.header("🎬 RAG-based YouTube Video Q&A")

# Initialize session state
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "summary" not in st.session_state:
    st.session_state.summary = None

if "video_id" not in st.session_state:
    st.session_state.video_id = ""


# --- Video ID input and Generate button --- #
video_id = st.text_input(
    "Enter YouTube Video ID",
    value=st.session_state.video_id,
    placeholder="e.g. Gfr50f6ZBvo",
)

generate_clicked = st.button("Generate Summary")

if generate_clicked:
    if not video_id.strip():
        st.warning("Please enter a valid YouTube video ID.")
    else:
        st.session_state.video_id = video_id.strip()
        with st.spinner("Building RAG chain and generating summary..."):
            chain, error_msg = get_rag_chain_for_video_id(st.session_state.video_id)

            if error_msg:
                st.session_state.rag_chain = None
                st.session_state.summary = None
                st.error(error_msg)
            else:
                st.session_state.rag_chain = chain
                # Ask the chain to summarize the video
                summary_query = (
                    "Provide a concise summary of this video's key points "
                    "in 6-8 sentences."
                )
                try:
                    summary_text = invoke_chain(chain, summary_query)
                    st.session_state.summary = summary_text
                except Exception as e:
                    st.session_state.summary = None
                    st.error(f"Error while generating summary: {e}")


# --- Show Summary if available --- #
if st.session_state.summary:
    st.subheader("📌 Summary of the Video")
    st.write(st.session_state.summary)

# --- Show Question box only after summary is generated --- #
if st.session_state.rag_chain:
    st.subheader("❓ Ask a Question about this Video")

    user_question = st.text_input(
        "Your question",
        placeholder="e.g. Is there any discussion about DeepMind?",
        key="user_question_input",
    )

    answer_clicked = st.button("Get Answer", key="get_answer_btn")

    if answer_clicked:
        if not user_question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                try:
                    answer = invoke_chain(st.session_state.rag_chain, user_question)
                    st.markdown("**Answer:**")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error while generating answer: {e}")




