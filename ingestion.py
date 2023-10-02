import os
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from decouple import Config

config = Config(
    "D:\VIT Material\VIT material\Projects\Langchain Projects\documentation-helper\.env"
)

pinecone.init(
    api_key=config.get("PINECONE_API_KEY"),
    environment=config.get("PINECONE_ENVIRONMENT"),
)


def ingest_docs() -> None:
    loader = ReadTheDocsLoader(path="langchain-docs/langchain.readthedocs.io/en/latest")
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents) }documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Split the document into {len(documents)} chunks")

    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Inserting {len(documents)} chunks to Pinecone")
    embeddings = OpenAIEmbeddings(openai_api_key=config.get("OPENAI_API_KEY"))
    Pinecone.from_documents(documents, embeddings, index_name="langchain-doc-index")
    print("Added to Pinecone vectorstore vectors!")


if __name__ == "__main__":
    ingest_docs()
