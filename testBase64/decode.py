import pandas as pd
import numpy as np
import base64
from sentence_transformers import SentenceTransformer
import re

# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def preprocess_text(text):
    """Preprocess text for embedding."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def encode_embedding_to_base64(embedding):
    """Encode a NumPy array to a base64 string."""
    return base64.b64encode(embedding.tobytes()).decode('utf-8')

def decode_base64_to_embedding(base64_string, embedding_shape):
    """Decode a base64 string to a NumPy array."""
    embedding_bytes = base64.b64decode(base64_string)
    return np.frombuffer(embedding_bytes, dtype=np.float32).reshape(embedding_shape)

# Load the CSV file
input_csv = '/workspaces/python/qbcs/questions.csv'
df = pd.read_csv(input_csv)

# Initialize list to store data for the output CSV
output_data = []

# # Compute and encode embeddings
for _, row in df.iterrows():
    question_id = row['questionId']
    text = preprocess_text(row['text'])
    
    # Compute embedding
    embedding = model.encode(text)
    
    # Encode embedding to base64
    embedding_base64 = encode_embedding_to_base64(embedding)
    
    # Append data to output list
    output_data.append([question_id, text, embedding_base64])

# Create DataFrame for output
output_df = pd.DataFrame(output_data, columns=['questionId', 'text', 'embedding_base64'])

# Save to new CSV file
output_csv = 'questions_with_embeddings.csv'
output_df.to_csv(output_csv, index=False)

print(f"Embeddings saved to {output_csv}")