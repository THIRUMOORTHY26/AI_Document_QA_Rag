from src.embeddings import create_embedding_model

embedding_model = create_embedding_model()
text = "Overfitting occurs when a model learns the training data too closely."
vector = embedding_model.embed_query(text)
print("Vector type:", type(vector))
print("Vector dimensions:", len(vector))
print("First 10 values:", vector[:10])