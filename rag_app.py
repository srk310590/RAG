import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
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


def load_and_chunk_data(file_path):
    """Loads and chunks data from a text file."""
    loader = TextLoader(file_path)
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
    # print("Generated embeddings: ")
    # print("Embeddings Generated")
    db = FAISS.from_documents(texts, embeddings)
    
    #db = FAISS.from_documents(texts, embeddings)
    print(db)
    return db


def create_retrieval_qa_chain(db):
    """Creates a retrieval QA chain."""
    llm = AzureChatOpenAI(
        deployment_name=os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT"),
        temperature=0.7,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY")
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


def main():
    """Main function to run the RAG application."""
    print("Main function to run the RAG application.")
    # Load and chunk data (replace with your data source)
    file_path = "data.txt"  # change this
    try:
      print("Loading and chunking data...")
      texts = load_and_chunk_data(file_path)
      print(texts)
    except Exception as e:
      print(f"Error loading or chunking data: {e}")
      return

    try:
      # Create the vector store
      print("Creating the vector store...")
      print(os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"))
      print(os.getenv("AZURE_OPENAI_ENDPOINT"))
      print(os.getenv("AZURE_OPENAI_API_KEY"))
      print(os.getenv("AZURE_OPENAI_API_VERSION"))
      print(os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT"))
      print(texts)
      db = create_vector_store(texts)
      print(db)
      # Create the retrieval QA chain
      print("Creating the retrieval QA chain...")
      qa_chain = create_retrieval_qa_chain(db)
      print(qa_chain)
      # Ask questions
      while True:
        query = input("Enter your question (or type 'exit' to quit): ")
        if query.lower() == "exit":
            break
        result = qa_chain.run(query)
        print(f"Answer: {result}\n")


    except Exception as e: 
      print(f"An error occurred during vector store creation or QA chain setup: {e}")

if __name__ == "__main__":
    main()
