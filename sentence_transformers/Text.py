from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Two lists of sentences
sentences1 = [
    "Given JavaScript Code Statement: \n What is name of the value?",
    "Which of the following statements regarding filters and transitions is FALSE?",
    "Which factory is used to load SAX parser dynamically at runtime?",
    "Which data type is used to contain the set of attributes of the element node in DOM?",
    "What does the combination of multi-pipeline and single-pipeline for?"
]

sentences2 = [
    "Which data type is used to contain the set of attributes of the element node in DOM?",
    "Data type is used attributes of the element node in",
    "A woman watches TV",
]

# Compute embeddings for both lists
embeddings1 = model.encode(sentences1)
embeddings2 = model.encode(sentences2)

# Compute cosine similarities
similarities = model.similarity(embeddings1, embeddings2)

# Output the pairs with their score
for idx_i, sentence1 in enumerate(sentences1):
    print(sentence1)
    for idx_j, sentence2 in enumerate(sentences2):
        print(f" - {sentence2: <30}: {similarities[idx_i][idx_j]:.4f}")