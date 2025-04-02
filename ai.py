import nltk
import json
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Sample Medical Knowledge Base
medical_data = {
    "What is diabetes?": "Diabetes is a chronic condition that affects how your body processes blood sugar (glucose).",
    "What are the symptoms of COVID-19?": "Common symptoms include fever, cough, fatigue, and loss of taste or smell.",
    "How to lower blood pressure?": "To lower blood pressure, reduce salt intake, exercise regularly, and manage stress.",
    "What are the side effects of paracetamol?": "Common side effects include nausea, rash, and liver damage in high doses."
}

# Preprocess and Vectorize Data
def preprocess(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

documents = list(medical_data.keys())
responses = list(medical_data.values())
processed_docs = [preprocess(doc) for doc in documents]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_docs)

# Flask API Setup
app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message", "")
    processed_input = preprocess(user_input)
    user_vector = vectorizer.transform([processed_input])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix)
    best_match_idx = np.argmax(similarity_scores)
    best_match_score = similarity_scores[0, best_match_idx]
    
    if best_match_score > 0.2:
        response = responses[best_match_idx]
    else:
        response = "I'm sorry, I don't have information on that. Please consult a doctor."
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
