import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Function to calculate similarity score
def calculate_similarity(text1, text2):
    sentences = [text1, text2]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    similarity_score = similarity_matrix[0][0]
    return similarity_score

# data = InputDataSet
data = pd.read_csv("/workspaces/python/qbcs/questions.csv")
# # Input data from SQL Server
questionId = data.iloc[0]["questionId"]
text = data.iloc[0]["text"]
# # Fetch all questions except the current one
other_questions = data[data["questionId"] != questionId]
# # Calculate similarity scores
similarities = []

for index, row in other_questions.iterrows():
    score = calculate_similarity(text, row["text"])
    if (score >= 0.2):
        similarities.append((row["questionId"], score))
# # Output DataFrame
OutputDataSet = pd.DataFrame(similarities, columns=["questionDuplicatedId", "similarScore"])
print(OutputDataSet)