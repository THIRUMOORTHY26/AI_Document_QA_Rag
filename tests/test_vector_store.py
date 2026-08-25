from src.document_loader import load_pdf
from src.text_splitter import split_documents
from src.vector_store import create_vector_store

pdf_path = "data/machine_learning.pdf"

documents = load_pdf(pdf_path)

chunks = split_documents(documents)

vector_store = create_vector_store(chunks)

print(f"Number of chunks stored: {len(chunks)}")

query = "What is overfitting?"

results = vector_store.similarity_search(query, k=2)

print("\nRetrieved chunks:\n")

for i, result in enumerate(results, start=1):
    print(f"--- Result {i} ---")
    print(result.page_content)
    print()