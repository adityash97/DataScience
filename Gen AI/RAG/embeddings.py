# text
# embedding
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.documents import Document

load_dotenv()
mistral_api_key = os.getenv('MISTRAL_API_KEY')

documents = [
    Document(
        page_content="RAG (Retrieval Augmented Generation) is a technique that combines retrieval of documents with text generation using LLMs.",
        metadata={"source": "rag_intro", "topic": "RAG", "level": "beginner"}
    ),
    Document(
        page_content="Vector databases like FAISS and Pinecone are used to store embeddings for efficient similarity search.",
        metadata={"source": "vector_db_guide", "topic": "embeddings", "tool": "FAISS"}
    ),
    Document(
        page_content="LangChain provides tools for building LLM applications such as chains, agents, and memory systems.",
        metadata={"source": "langchain_docs", "topic": "framework", "library": "LangChain"}
    ),
    Document(
        page_content="Chunking is important in RAG because it ensures that large documents are split into smaller, meaningful pieces.",
        metadata={"source": "rag_best_practices", "topic": "chunking"}
    ),
    Document(
        page_content="Embedding models convert text into numerical vectors so that semantic similarity can be measured.",
        metadata={"source": "embedding_basics", "topic": "embeddings"}
    ),
    Document(
        page_content="This document is irrelevant and should ideally not be retrieved for most queries.",
        metadata={"source": "noise", "topic": "irrelevant"}
    ),
    Document(
        page_content="The capital of France is Paris. It is known for the Eiffel Tower.",
        metadata={"source": "general_knowledge", "topic": "geography"}
    ),
    Document(
        page_content="Python is a popular programming language used in AI, machine learning, and backend development.",
        metadata={"source": "programming", "topic": "python"}
    ),
]


if __name__ == '__main__':
    embedding = MistralAIEmbeddings(model="mistral-embed",api_key=mistral_api_key)

    vector_store = Chroma(
        collection_name="foo",
        embedding_function=embedding,
        persist_directory="embedding_store"

    )

    vector_store.add_documents(documents=documents)

    retriever = vector_store.as_retriever(search_type="mmr",search_kwargs={"k": 2, "fetch_k": 2, "lambda_mult": 0.5})
    found_document = retriever.invoke("What is vector database")

    print("\n\n\n\n\n",found_document)




