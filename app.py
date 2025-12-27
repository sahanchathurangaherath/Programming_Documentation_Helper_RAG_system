import streamlit as st
import os
from dotenv import load_dotenv
from rag_system import RAGSystem
from document_loader import load_pdf, load_url
import tempfile

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="Programming Doc Helper",
    page_icon="📚",
    layout="wide"
)

# Initialize RAG System
@st.cache_resource
def get_rag_system():
    try:
        return RAGSystem()
    except ValueError as e:
        return None

def main():
    st.title("📚 Programming Documentation Helper")
    
    rag = get_rag_system()
    
    if rag is None:
        st.error("⚠️ GEMINI_API_KEY not found in environment variables. Please check your .env file.")
        st.info("You can get an API key from Google AI Studio.")
        return

    # Sidebar for Document Loading
    with st.sidebar:
        st.header("Add Documentation")
        
        tab1, tab2 = st.tabs(["URL", "PDF"])
        
        with tab1:
            url_input = st.text_input("Enter URL")
            if st.button("Load URL"):
                if url_input:
                    with st.spinner("Loading URL..."):
                        try:
                            text = load_url(url_input)
                            rag.add_document(text, source=url_input)
                            st.success(f"Successfully loaded {url_input}")
                        except Exception as e:
                            st.error(f"Error loading URL: {str(e)}")
        
        with tab2:
            uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
            if st.button("Load PDF"):
                if uploaded_file:
                    with st.spinner("Processing PDF..."):
                        try:
                            # Read file content
                            text = load_pdf(uploaded_file)
                            rag.add_document(text, source=uploaded_file.name)
                            st.success(f"Successfully loaded {uploaded_file.name}")
                        except Exception as e:
                            st.error(f"Error processing PDF: {str(e)}")

        st.markdown("---")
        st.markdown("### 🛠️ System Control")
        if st.button("Clear Knowledge Base"):
            rag.clear_database()
            st.success("Knowledge base cleared!")

    # Main Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = rag.query(prompt)
                    answer = response["answer"]
                    sources = list(set(response["source_documents"]))
                    
                    full_response = f"{answer}\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in sources])
                    
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
