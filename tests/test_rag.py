from src.document_loader import load_pdf
from src.text_splitter import split_documents
from src.vector_store import create_vector_store
from src.rag_pipeline import create_llm, generate_answer

# 1. Load PDF
pdf_path = "data/machine_learning.pdf"
documents = load_pdf(pdf_path)

# 2. Split into chunks
chunks = split_documents(documents)

# 3. Create vector store
vector_store = create_vector_store(chunks)

# 4. Create local LLM
llm = create_llm()

# 5. Ask a question
question = "What are the different types of machine learning?"

# 6. Retrieve relevant chunks
retrieved_documents = vector_store.similarity_search(
    question,
    k=2
)

# 7. Generate answer
answer = generate_answer(
    llm,
    question,
    retrieved_documents
)

print("\nQuestion:")
print(question)

print("\nAnswer:")
print(answer)

print("\nSources:")

for document in retrieved_documents:
    source = document.metadata.get("source", "Unknown source")
    page = document.metadata.get("page", None)

    if page is not None:
        page = page + 1
        print(f"- {source} — Page {page}")
    else:
        print(f"- {source}")