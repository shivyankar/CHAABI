import pandas as pd
from transformers import pipeline
import requests
import json
from flask import Flask, request, jsonify

# Load the dataset
url = "https://chaabiv2.s3.ap-south-1.amazonaws.com/hiring/bigBasketProducts.csv"
df = pd.read_csv(url)

# Combine product information and description for the language model input
df['input_text'] = df['product'] + " " + df['description']

# Initialize the vectorizer (using a pre-trained BERT model)
vectorizer = pipeline('feature-extraction', model='bert-base-uncased', device=0)

# Vectorize the text data
embeddings = [vectorizer(text)[0].tolist() for text in df['context']]

# Create a new column in the DataFrame for storing the embeddings
df['embeddings'] = embeddings

# Extract relevant information for Qdrant
documents = [
    {
        "id": idx,
        "vector": vector.tolist(),
        "metadata": {"product": row['product'], "category": row['category'], "brand": row['brand'], "description": row['description']}
    }
    for idx, (index, row, vector) in enumerate(zip(df.index, df.iterrows(), embeddings))
]

# Convert documents to Qdrant format
documents_json = json.dumps({"documents": documents})

# Qdrant API endpoint for indexing vectors
qdrant_endpoint = "http://localhost:6333/index"

# Send a POST request to Qdrant to index the vectors
response = requests.post(qdrant_endpoint, data=documents_json, headers={"Content-Type": "application/json"})

# Check the response
if response.status_code == 200:
    print("Vectors indexed successfully.")
else:
    print(f"Failed to index vectors. Status code: {response.status_code}, Response: {response.text}")

# Initialize the question-answering pipeline with a pre-trained model
qa_pipeline = pipeline('question-answering', model='deepset/roberta-base-squad2', device=0)

# Initialize the Flask application
app = Flask(__name__)


# Function to get contextual answers from the database
def get_answer(query, df):
    # Get the most relevant passage (product information + context) for the query
    relevant_passage = df['input_text'][df['input_text'].str.contains(query, case=False)].values.tolist()

    if not relevant_passage:
        return "No relevant information found."

    # Use the question-answering pipeline to extract the answer
    answer = qa_pipeline(question=query, context=relevant_passage[0])
    return answer['answer']


# API endpoint for answering questions
@app.route('/answer', methods=['POST'])
def answer_question():
    try:
        data = request.get_json()
        query = data['question']

        # Get answer from the database
        answer = get_answer(query, df)

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    # Run the Flask application
    app.run(debug=True)
