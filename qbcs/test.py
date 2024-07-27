from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(text1, text2):
    sentences = [text1, text2]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    similarity_score = similarity_matrix[0][0]
    return similarity_score

import pandas as pd

# Example static data for testing
data = {
    "questionId": [1, 2],
    "text": ["I love programming in Python.", "I enjoy writing code in Python."]
}
input_data = pd.DataFrame(data)

questionId = 1
text = "I love programming in Python."
similarities = []

for index, row in input_data.iterrows():
    if row["questionId"] != questionId:
        score = calculate_similarity(text, row["text"])
        similarities.append((row["questionId"], score))

OutputDataSet = pd.DataFrame(similarities, columns=["questionDuplicatedId", "similarScore"])

print(OutputDataSet)