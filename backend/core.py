import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
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


def run_llm(query: str, chat_history: list[tuple(str, any)] = []) -> any:
    embeddings = OpenAIEmbeddings(openai_api_key=config.get("OPENAI_API_KEY"))
    docsearch = Pinecone.from_existing_index(
        index_name="langchain-doc-index", embedding=embeddings
    )
    chat_llm = ChatOpenAI(
        openai_api_key=config.get("OPENAI_API_KEY"), verbose=True, temperature=0
    )

    ''' qa_chain = RetrievalQA.from_chain_type(
        llm=chat_llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )'''

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_llm, retriever=docsearch.as_retriever(), return_source_documents=True
    )

    return qa_chain({"question": query, "chat_history":chat_history})


if __name__ == "__main__":
    run_llm(query="What is a langchain chain?")
