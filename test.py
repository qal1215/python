from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re

# Load pre-trained Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')


def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation and non-alphanumeric characters (optional)
    text = re.sub(r'[^\w\s]', '', text)
    return text


def get_text_embedding(text):
    return model.encode(text)


def check_overlap_embedding(text1, text2, threshold=0.8):
    # Get embeddings
    embedding1 = get_text_embedding(text1)
    embedding2 = get_text_embedding(text2)

    # Compute cosine similarity
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]

    # Return 1 if similarity is above the threshold, otherwise 0
    return 1 if similarity >= threshold else 0


# Example usage
text1 = "How can I improve my SEO ranking?"
text2 = "What you name?"

preprocessed_text1 = preprocess_text(text1)
preprocessed_text2 = preprocess_text(text2)

result = check_overlap_embedding(preprocessed_text1, preprocessed_text2)
print("Overlap result (Embedding):", result)
