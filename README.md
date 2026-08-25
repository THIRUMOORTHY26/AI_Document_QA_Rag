#  AI Document Q&A — RAG

An AI-powered document question-answering application built using
Retrieval-Augmented Generation (RAG).

Users can upload a PDF document and ask questions about its content.
The application retrieves relevant document chunks using semantic
search and generates answers using a locally hosted Qwen 2.5 3B model.

##  Features

- Upload PDF documents
- Extract text from PDF files
- Split documents into smaller chunks
- Generate semantic embeddings
- Store and search embeddings using FAISS
- Retrieve relevant document content
- Generate answers using Qwen 2.5 3B
- Display document source and page information
- Interactive Streamlit interface
- Runs locally without requiring an OpenAI API key

##  RAG Architecture

```text
                PDF Document
                     │
                     ▼
              PyPDFLoader
                     │
                     ▼
             Text Chunking
                     │
                     ▼
       Sentence Transformer
       (all-MiniLM-L6-v2)
                     │
                     ▼
                   FAISS
              Vector Store
                     │
                     ▼
             User Question
                     │
                     ▼
            Similarity Search
                     │
                     ▼
            Relevant Chunks
                     │
                     ▼
          Qwen 2.5 3B / Ollama
                     │
                     ▼
              Final Answer
                     │
                     ▼
              Source / Page

### Tech Stack

- Python 3.12
- LangChain
- Streamlit
- FAISS
- Sentence Transformers
- Hugging Face
- Ollama
- Qwen 2.5 3B
- PyPDF
- Git & GitHub

####  Installation

1. Clone the repository
```bash
git clone https://github.com/THIRUMOORTHY26/AI_Document_QA_Rag.git

2. Navigate to the project
```bash
cd AI_Document_QA_Rag

3. Create a virtual environment
```bash
python -m venv venv

4. Activate the virtual environment
```bash
venv\Scripts\activate

5. Install Python dependencies
```bash
pip install -r requirements.txt

6. Install Ollama
```bash
ollama pull qwen2.5:3b

7. Run the application
```bash
python -m streamlit run app.py
