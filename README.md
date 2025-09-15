# RAG-chatbot-with-real-time-web-search
This project implements a Retrieval-Augmented Generation (RAG) chatbot that can answer user queries using both local documents and real-time web search. Unlike traditional chatbots that rely solely on pre-trained data, this system dynamically retrieves relevant information to provide more accurate and up-to-date responses.

Tech stack

Python
Streamlit – Web interface
Ollama / Hugging Face – Embeddings & LLM integration
Vector Store (Chroma) – Document indexing
DuckDuckGo / Custom Search API – Web search integration

Key features
Document Processing: Load and split documents into chunks, then generate embeddings for semantic search.
Hybrid Retrieval: Perform similarity search over local documents and augment with web search results.
Context Assembly: Combine retrieved chunks and web content into a structured context.
Answer Generation: Use a Large Language Model (LLM) to generate coherent answers based on the assembled context.
Source Attribution: Provide clear references to retrieved documents and web sources.
Web Interface: Simple Streamlit UI for interactive queries and responses.




https://github.com/user-attachments/assets/c95f255e-5bd4-4aa7-b941-896120615060

