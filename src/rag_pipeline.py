from langchain_ollama import ChatOllama

def create_llm():
    llm = ChatOllama(
        model="qwen2.5:3b",
        temperature=0
    )

    return llm

def generate_answer(llm, question, retrieved_documents):
    context = "\n\n".join(
        document.page_content
        for document in retrieved_documents
    )
    prompt = f"""
You are a helpful document question-answering assistant.

Answer the user's question using ONLY the context provided below.

If the answer cannot be found in the context, say:
"I could not find the answer in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content