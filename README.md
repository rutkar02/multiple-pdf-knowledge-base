Multi PDF Knowledge Base

A Retrieval-Augmented Generation (RAG) application built with Streamlit and OpenAI that allows users to upload multiple PDF documents and ask questions across all of them.

Features

* Upload multiple PDF documents
* Automatic text extraction
* Document chunking
* OpenAI embeddings generation
* Cosine similarity search
* Top-K retrieval
* Source attribution
* Multi-document question answering

Tech Stack

* Python
* Streamlit
* OpenAI API
* PyPDF
* NumPy
* OpenAI Embeddings (text-embedding-3-small)

How It Works

1. Upload one or more PDF files.
2. Text is extracted from each PDF.
3. Documents are split into chunks.
4. Embeddings are generated for each chunk.
5. Embeddings are cached in session state.
6. User question is converted into an embedding.
7. Cosine similarity finds the most relevant chunks.
8. Top matching chunks are sent to GPT.
9. GPT generates an answer along with source documents.

Architecture

PDFs
→ Text Extraction
→ Chunking
→ Embeddings
→ Vector Store (Session State)
→ Similarity Search
→ Top-K Retrieval
→ GPT Response

Run Locally

pip install -r requirements.txt
streamlit run app.py

Future Improvements

* Persistent vector database (ChromaDB / FAISS)
* Metadata filtering
* Better chunking strategies
* Citation highlighting
* PDF page references
* Hybrid search (keyword + vector search)
* Chat history support
* Streaming responses

Learning Outcomes

This project demonstrates:

* Retrieval-Augmented Generation (RAG)
* Embeddings
* Cosine Similarity
* Vector Search
* Chunking
* Metadata Management
* Caching
* Multi-document Retrieval
* LLM Application Development
