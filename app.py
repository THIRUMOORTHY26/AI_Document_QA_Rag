import streamlit as st
from src.document_loader import load_pdf
from src.text_splitter import split_documents
from src.vector_store import create_vector_store
from src.rag_pipeline import create_llm, generate_answer

st.set_page_config(
    page_title="AI Document Q&A",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Document Q&A")
st.write("Ask questions about your PDF using Retrieval-Augmented Generation (RAG).")

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "llm" not in st.session_state:
    st.session_state.llm = None


# PDF upload
uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"]
)
if uploaded_file is not None:

    if st.button("Process Document"):

        with st.spinner("Processing document..."):

            # Save uploaded PDF temporarily
            pdf_path = "data/uploaded_document.pdf"

            with open(pdf_path, "wb") as file:
                file.write(uploaded_file.getbuffer())

            # Load document
            documents = load_pdf(pdf_path)

            # Split document
            chunks = split_documents(documents)

            # Create vector store
            vector_store = create_vector_store(chunks)

            # Create LLM
            llm = create_llm()

            # Store in session
            st.session_state.vector_store = vector_store
            st.session_state.llm = llm

        st.success(
            f"Document processed successfully! "
            f"Created {len(chunks)} chunks."
        )

# Question section
if st.session_state.vector_store is not None:

    st.subheader("Ask a Question")

    question = st.text_input(
        "Enter your question"
    )

    if st.button("Ask Question"):

        if question.strip():

            with st.spinner("Finding answer..."):

                # Retrieve relevant documents
                retrieved_documents = (
                    st.session_state.vector_store.similarity_search(
                        question,
                        k=2
                    )
                )

                # Generate answer
                answer = generate_answer(
                    st.session_state.llm,
                    question,
                    retrieved_documents
                )

            st.subheader("💡 Answer")
            st.write(answer)

            # Sources
            st.subheader("📚 Sources")

            sources = set()

            for document in retrieved_documents:

                source = document.metadata.get(
                    "source",
                    "Unknown source"
                )

                page = document.metadata.get(
                    "page",
                    None
                )

                if page is not None:
                    source_info = (
                        f"{source} — Page {page + 1}"
                    )
                else:
                    source_info = source

                sources.add(source_info)

            for source in sources:
                st.write(f"- {source}")

        else:
            st.warning("Please enter a question.")