import streamlit as st
from rag_app import build_rag_pipeline, load_and_chunk_data, create_vector_store
import tempfile
import os
from langchain_community.document_loaders import (
    TextLoader,
    Docx2txtLoader,
    PyPDFLoader,
)

# Set page configuration
st.set_page_config(page_title="RAG App", page_icon=":book:")

st.title("RAG Application")

# Dictionary to map file types to loaders
FILE_LOADER_MAPPING = {
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
    ".doc": Docx2txtLoader,
    ".pdf": PyPDFLoader,
}


def handle_uploaded_file(uploaded_file):
    """Handles the uploaded file, processes it, and updates the RAG pipeline."""
    if uploaded_file is not None:
        try:
            # Determine the file extension
            file_extension = os.path.splitext(uploaded_file.name)[1]
            # Check if the file type is supported
            if file_extension not in FILE_LOADER_MAPPING:
                st.error(
                    f"Unsupported file type: {file_extension}. Please upload a .txt, .doc, .docx, or .pdf file."
                )
                return False
            # Get the appropriate loader class
            loader_class = FILE_LOADER_MAPPING[file_extension]

            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=file_extension
            ) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            # load and create vector store from the uploaded file
            texts = load_and_chunk_data(temp_file_path, loader_class)
            db = create_vector_store(texts)

            # Update the RAG pipeline in session state
            st.session_state.qa_chain = build_rag_pipeline(db=db)

            st.success("File uploaded and processed successfully!")
            return True

        except Exception as e:
            st.error(f"Error processing file: {e}")
            return False

        finally:
            # Remove the temporary file
            if "temp_file_path" in locals() and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    return False


# Sidebar for file upload
st.sidebar.header("Upload File")
uploaded_file = st.sidebar.file_uploader(
    "Choose a file", type=["txt", "doc", "docx", "pdf"]
)

# Initialize the RAG pipeline with default data if no file is uploaded
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = build_rag_pipeline()

if uploaded_file is not None:
    handle_uploaded_file(uploaded_file)

# Input box for the question
query = st.text_input("Ask a question about the data:")

# Get and display the answer
if query:
    try:
        with st.spinner("Thinking..."):
            result = st.session_state.qa_chain.run(query)
        st.markdown(f"**Answer:** {result}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

