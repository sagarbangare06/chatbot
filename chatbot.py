from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Step 1: Azure Configuration
API_KEY = "your_azure_api_key"
ENDPOINT = "your_azure_endpoint"
DEPLOYMENT_NAME = "text-embedding-ada-002"

def authenticate_client(api_key, endpoint):
    """Authenticate with Azure Text Analytics API."""
    return TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))

client = authenticate_client(API_KEY, ENDPOINT)

# Step 2: Predefine Parameters
parameters = [
    "New York", "San Francisco", "London",  # Locations
    "January 1st", "next Monday",          # Dates
    "Laptop", "Smartphone",                # Products
    "$100", "$200", "affordable", "cheap"  # Prices
]

# Step 3: Text Preprocessing
def preprocess_text(text):
    """Preprocess text for embedding generation."""
    return re.sub(r"[^\w\s]", "", text.lower().strip())

parameters = [preprocess_text(param) for param in parameters]

# Step 4: Generate Embeddings for Parameters
def generate_embeddings(client, texts):
    """Generate embeddings using Azure's Text Embedding model."""
    response = client.get_embeddings(documents=texts, deployment_name=DEPLOYMENT_NAME)
    embeddings = np.array([doc.embedding for doc in response.documents])
    # Normalize embeddings to unit vectors
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

parameter_embeddings = generate_embeddings(client, parameters)

# Step 5: Capture User Input and Generate Embedding
def get_user_query_embedding(client, query):
    """Generate embedding for the user query."""
    query = preprocess_text(query)
    response = client.get_embeddings(documents=[query], deployment_name=DEPLOYMENT_NAME)
    embedding = np.array(response.documents[0].embedding)
    # Normalize embedding to unit vector
    return embedding / np.linalg.norm(embedding).reshape(1, -1)

# Step 6: Compare Embedding Vectors (Cosine Similarity)
def find_closest_parameters(query_embedding, parameter_embeddings, parameters):
    """Find the most relevant parameters based on cosine similarity."""
    similarities = cosine_similarity(query_embedding, parameter_embeddings).flatten()
    sorted_indices = similarities.argsort()[::-1]  # Sort indices by similarity (descending)
    return [(parameters[idx], similarities[idx]) for idx in sorted_indices]

# Step 7: Handle Multiple Parameters
def handle_multiple_parameters(results, threshold=0.5):
    """Select parameters with similarity above the threshold."""
    return [param for param, score in results if score >= threshold]

# Main Function
def identify_parameters(client, user_query, parameters, parameter_embeddings):
    """Identify relevant parameters for a user query."""
    query_embedding = get_user_query_embedding(client, user_query)
    results = find_closest_parameters(query_embedding, parameter_embeddings, parameters)
    # Dynamically adjust threshold based on query similarity distribution
    max_score = max(score for _, score in results)
    threshold = max(0.5, max_score * 0.8)  # Example: 80% of max score
    relevant_parameters = handle_multiple_parameters(results, threshold=threshold)
    return relevant_parameters, results

# Example Usage
if __name__ == "__main__":
    user_query = "I want to buy a cheap smartphone for under $200."
    relevant_parameters, results = identify_parameters(client, user_query, parameters, parameter_embeddings)
    
    print("User Query:", user_query)
    print("\nIdentified Parameters:", relevant_parameters)
    print("\nDetailed Similarities:")
    for param, score in results:
        print(f"{param}: {score:.4f}")
