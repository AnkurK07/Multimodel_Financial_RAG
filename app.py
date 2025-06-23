import os
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.vectorstores import Weaviate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from document_processors import load_multimodal_data, load_data_from_directory

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Initialize session state variables for managing chat history and document index
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'retriever' not in st.session_state:
    st.session_state['retriever'] = None

st.set_page_config(layout="wide")

# Initialize LLM (LLaMA 3.3-70B via Groq)
@st.cache_resource
def initialize_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama3-70b-8192",
        temperature=0.2
    )

@st.cache_resource
def initialize_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Create Weaviate vector store and retriever

def create_retriever(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    texts = [chunk for doc in documents for chunk in splitter.split_text(doc.page_content)]

    embeddings = initialize_embeddings()
    vectorstore = Weaviate(
        url="http://localhost:8080",
        index_name="MultimodalRAG",
        embedding=embeddings,
    )
    vectorstore.add_texts(texts)
    return vectorstore.as_retriever()

def main():
    llm = initialize_llm()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.title("Multimodal Financial RAG")

        input_method = st.radio("Choose input method:", ("Upload Files", "Enter Directory Path"))

        if input_method == "Upload Files":
            uploaded_files = st.file_uploader("Drag and drop files here", accept_multiple_files=True)
            if uploaded_files and st.button("Process Files"):
                with st.spinner("Processing files..."):
                    documents = load_multimodal_data(uploaded_files, llm)
                    st.session_state['retriever'] = create_retriever(documents)
                    st.success("Files processed and retriever created!")
        else:
            directory_path = st.text_input("Enter directory path:")
            if directory_path and st.button("Process Directory"):
                if os.path.isdir(directory_path):
                    with st.spinner("Processing directory..."):
                        documents = load_data_from_directory(directory_path, llm)
                        st.session_state['retriever'] = create_retriever(documents)
                        st.success("Directory processed and retriever created!")
                else:
                    st.error("Invalid directory path. Please enter a valid path.")

    with col2:
        if st.session_state['retriever'] is not None:
            st.title("Chat")
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=st.session_state['retriever'],
                return_source_documents=False
            )

            chat_container = st.container()
            with chat_container:
                for message in st.session_state['history']:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            user_input = st.chat_input("Enter your query:")

            if user_input:
                with st.chat_message("user"):
                    st.markdown(user_input)
                st.session_state['history'].append({"role": "user", "content": user_input})

                with st.chat_message("assistant"):
                    with st.spinner("Generating response..."):
                        result = qa_chain.run(user_input)
                        st.markdown(result)
                st.session_state['history'].append({"role": "assistant", "content": result})
                st.rerun()

            if st.button("Clear Chat"):
                st.session_state['history'] = []
                st.rerun()

if __name__ == "__main__":
    main()
