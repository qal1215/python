import numpy as np
import base64
from sentence_transformers import SentenceTransformer
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


def encode_embedding_to_base64(embedding):
    # Convert the NumPy array to bytes
    embedding_bytes = embedding.tobytes()
    # Encode bytes to base64 string
    base64_encoded = base64.b64encode(embedding_bytes).decode('utf-8')
    return base64_encoded

def decode_base64_to_embedding(base64_string, embedding_shape):
    # Decode the base64 string to bytes
    embedding_bytes = base64.b64decode(base64_string)
    # Convert bytes back to NumPy array with the original shape
    embedding = np.frombuffer(embedding_bytes, dtype=np.float32).reshape(embedding_shape)
    return embedding

# Example usage
text1 = "How can I improve my SEO ranking?"
text2 = "What are some tips for better SEO?"

preprocessed_text1 = preprocess_text(text1)
preprocessed_text2 = preprocess_text(text2)

# Get embeddings
embedding1 = get_text_embedding(preprocessed_text1)
embedding2 = get_text_embedding(preprocessed_text2)

# Encode embeddings
embedding1_base64 = encode_embedding_to_base64(embedding1)
embedding2_base64 = encode_embedding_to_base64(embedding2)

print("Embedding 1 (Base64):", embedding1_base64)
print("Embedding 2 (Base64):", embedding2_base64)

# Decode embeddings (for example purpose)
# Define the shape of embeddings (e.g., (384,) for 'all-MiniLM-L6-v2')
# embedding_shape = (embedding1.size,)

# decoded_embedding1 = decode_base64_to_embedding(embedding1_base64, embedding_shape)
# decoded_embedding2 = decode_base64_to_embedding(embedding2_base64, embedding_shape)

# print("Decoded Embedding 1:", decoded_embedding1)
# print("Decoded Embedding 2:", decoded_embedding2)