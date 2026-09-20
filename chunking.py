from langchain_voyageai import VoyageAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os

load_dotenv()


def load_pdf(pdf_file: str):
    if not os.path.isfile(pdf_file):
        raise FileNotFoundError(f"PDF file not found: {pdf_file}")
    loader = PyPDFLoader(pdf_file)
    return loader.load()

def get_chunks(pdf_file: str, size: int, overlap: int):
    docs = load_pdf(pdf_file)
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    return splitter.split_documents(docs)

def create_embeddings(pdf_file: str, size: int = 1000, overlap: int = 200):
    chunks = get_chunks(pdf_file, size, overlap)
    api_key = os.getenv("VOYAGEAI_API_KEY")
    if not api_key:
        raise RuntimeError("VOYAGEAI_API_KEY is required to index a PDF.")

    embeddings = VoyageAIEmbeddings(
        voyage_api_key=api_key, model="voyage-law-2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def create_retriever(pdf_file: str, size: int = 1000, overlap: int = 200):
    """Build an in-memory retriever for one uploaded PDF."""
    vector_store = create_embeddings(pdf_file, size, overlap)
    return vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )
