from src.document_loader import load_pdf
from src.text_splitter import split_documents


pdf_path = "data/machine_learning.pdf"

documents = load_pdf(pdf_path)

chunks = split_documents(documents)

print(f"Number of pages: {len(documents)}")
print(f"Number of chunks: {len(chunks)}")

print("\nFirst chunk:")
print(chunks[0].page_content)