import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import AzureOpenAIEmbeddings

# Load environment variables
load_dotenv()

# Set up Azure OpenAI - THIS IS REQUIRED
os.environ["OPENAI_API_TYPE"] = "azure"


def load_and_chunk_data(file_path, loader_class):
    """Loads and chunks data from a file."""
    loader = loader_class(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    return texts


def create_vector_store(texts):
    """Creates a vector store from the provided text chunks."""
    embeddings = AzureOpenAIEmbeddings(
        model=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        chunk_size=2000,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )
    db = FAISS.from_documents(texts, embeddings)
    print(db)
    return db


def create_retrieval_qa_chain(db):
    """Creates a retrieval QA chain."""
    llm = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT"),
        temperature=0.7,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )

    retriever = db.as_retriever()

    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=retriever, chain_type_kwargs={"prompt": PROMPT}
    )

    return qa_chain


def build_rag_pipeline(file_path="data.txt", db=None, loader_class=None):
    """Builds the entire RAG pipeline."""
    print("Building RAG pipeline...")
    if loader_class is None:
        from langchain_community.document_loaders import TextLoader
        loader_class = TextLoader

    if db is None:
        try:
            texts = load_and_chunk_data(file_path, loader_class)
        except Exception as e:
            print(f"Error loading or chunking data: {e}")
            raise e  # Re-raise the exception to be handled upstream

        try:
            db = create_vector_store(texts)
        except Exception as e:
            print(f"Error creating vector store or QA chain: {e}")
            raise e
    try:
        qa_chain = create_retrieval_qa_chain(db)
        return qa_chain
    except Exception as e:
        print(f"Error creating vector store or QA chain: {e}")
        raise e  # Re-raise the exception to be handled upstream
