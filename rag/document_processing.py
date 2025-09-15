from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

def load_and_split_documents():
    loader = DirectoryLoader("documents", glob="**/*", loader_cls=TextLoader, 
                           loader_kwargs={'encoding': 'utf-8'})
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)
    return split_docs

def create_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory="chroma_db")
    vectordb.persist()
    return vectordb
