# Multimodel_Financial_RAG
![image](https://github.com/user-attachments/assets/012e444a-3b7e-4632-b05c-57d30bb5629b)

## 🧩 Problem Statement
Financial analysts and decision-makers often face challenges in extracting insights from large volumes of unstructured and multimodal financial data,such as reports, charts, scanned documents, and earnings call recordings. Traditional search and NLP tools lack the ability to seamlessly understand and retrieve context across text, image, and audio formats.

To address this, we propose building a ***Multimodal Retrieval-Augmented Generation (RAG)*** system that can intelligently query and summarize financial information from diverse sources using advanced LLMs and vector-based retrieval.

## 🔁 RAG Workflow: Steps and Tools Used

### 🔹 Step 1: Data Ingestion
Goal: Load text, images, and audio from financial documents.

Tools:

- ***PyMuPDF, python-docx, python-pptx*** – Parse PDFs, DOCX, PPTs

- ***Whisper (OpenAI)*** – Transcribe audio

- ***LLaMA 3.2–11B Vision*** – Summarize image-based content like chart, tables

### 🔹 Step 2: Chunking & Embedding
Goal: Break documents into meaningful chunks and embed them.

Tools:

- ***LangChain*** → RecursiveCharacterTextSplitter

- ***OpenAIEmbeddings from LangChain*** → Generates text embeddings

```
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
```

### 🔹 Step 3: Vector Indexing
Goal: Store embeddings for semantic retrieval.

Tools:

- ***Weaviate from LangChain*** → Connects directly to Weaviate DB

```
from langchain.vectorstores import Weaviate
vectorstore = Weaviate(
    url='http://localhost:8080',
    index_name='multimodal_rag',
    embedding=embeddings,
)
```

### 🔹 Step 4: Query Handling
Goal: Retrieve top-k relevant chunks for a given user query.

Tools:

- ***vectorstore.as_retriever()*** – Retrieves documents based on similarity

- ***LangChain RetrievalQA*** – Combines retriever with LLM

### 🔹 Step 5: Response Generation
Goal: Generate an accurate, grounded response from retrieved context.

Tools:

- ***ChatGroq with LLaMA 3.3–70B*** (LangChain wrapper)

- ***LangChain RetrievalQA Chain***

### 🔹 Step 6: Frontend Interface
Goal: Let users upload files, ask questions, and receive answers.

Tools:

Streamlit – Frontend for uploading documents and chatting with the RAG agent
