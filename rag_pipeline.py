from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline


def load_docs(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    return splitter.split_documents(docs)


def create_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def create_vectorstore(chunks, embeddings):
    return Chroma.from_documents(chunks, embeddings)


def load_llm():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=512
    )
    return HuggingFacePipeline(pipeline=pipe)


def create_qa_chain(vectorstore, llm):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )