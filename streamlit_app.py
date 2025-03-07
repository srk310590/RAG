import streamlit as st
from rag_app import build_rag_pipeline

# Set page configuration
st.set_page_config(page_title="RAG App", page_icon=":book:")

# Initialize the RAG pipeline
if "qa_chain" not in st.session_state:
    try:
        st.session_state.qa_chain = build_rag_pipeline()
    except Exception as e:
        st.error(f"Failed to initialize RAG pipeline: {e}")
        st.stop()

st.title("RAG Application")

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
