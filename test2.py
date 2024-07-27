import sqlite3

# Connect to SQLite database
conn = sqlite3.connect('embeddings.db')
cursor = conn.cursor()

# Create table if not exists
cursor.execute('''
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,
    embedding_base64 TEXT
)
''')

# Insert embeddings
cursor.execute('''
INSERT INTO embeddings (text, embedding_base64)
VALUES (?, ?)
''', (text1, embedding1_base64))

# Commit and close
conn.commit()

# Retrieve embeddings
cursor.execute('SELECT embedding_base64 FROM embeddings WHERE text=?', (text1,))
retrieved_base64 = cursor.fetchone()[0]

# Decode retrieved embedding
retrieved_embedding = decode_base64_to_embedding(retrieved_base64, embedding_shape)

print("Retrieved Embedding:", retrieved_embedding)

# Close the connection
conn.close()