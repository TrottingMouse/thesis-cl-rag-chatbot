"""
src – Continual-Learning RAG Chatbot

Top-level package for the thesis CL-RAG chatbot.
Contains two main sub-packages:

- offline: document preprocessing, chunking, embedding and FAISS index building
- online:  query processing, retrieval, reranking and answer generation
- models:  shared Pydantic / dataclass models used across both pipelines
"""
