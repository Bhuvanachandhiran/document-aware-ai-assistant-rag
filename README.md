# Document-Aware AI Assistant (RAG)

A Generative AI application that answers questions using user-provided documents through Retrieval-Augmented Generation (RAG).

## Problem Statement
Large Language Models can generate incorrect or hallucinated answers when they lack domain-specific knowledge. This project addresses that issue by grounding responses in external documents.

## Solution Overview
The system retrieves relevant document content using embeddings and vector similarity search, then passes that context to a language model to generate accurate, grounded answers.

## Features
- Document ingestion and indexing
- Semantic retrieval using embeddings
- Context-aware answer generation
- Hallucination control
- Interactive web interface

## Tech Stack
- Python
- Sentence Transformers
- FAISS
- Hugging Face Transformers
- Streamlit
- PyTorch

## Architecture
1. Document text is split into chunks  
2. Chunks are converted into embeddings  
3. FAISS indexes embeddings for similarity search  
4. Relevant chunks are retrieved for each query  
5. LLM generates answers using retrieved context  


