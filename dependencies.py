
def install_dependencies():
    import os

    commands = [

        "pip install pinecone-client",
        "pip install chromadb langchain tiktoken",
        "pip install Chroma",
        "pip install -U langchain-community",
        "pip install langchain_together",
        "pip install sentence-transformers",
        "pip install boto3 requests",
        "pip install transformers datasets faiss-cpu",
        "pip install python-docx pandas",
        "pip install qdrant-client",
        "pip install chromadb",
        "pip install pymilvus",
        "pip install weaviate-client",
        "pip install langchain-weaviate",
        "pip install PyPDF2"

    ]

    for command in commands:
        os.system(command)

install_dependencies()
