import os
import openai
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import chromadb
from scipy.spatial import distance
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

# Load environment variables from .env (OPENAI_API_KEY)
load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"

# Load dataset and drop rows with missing review text
reviews = pd.read_csv("src/data/datalab_export_2026-05-12 16_27_43.csv")
review_texts = reviews["Review Text"].dropna().reset_index(drop=True)

# Generate embeddings for all reviews in a single API call
openai_client = openai.OpenAI()
responses = openai_client.embeddings.create(input=review_texts.tolist(), model=EMBEDDING_MODEL).model_dump()
embeddings = [response["embedding"] for response in responses["data"]]


def apply_tsne(embeddings):
    # Reduce high-dimensional embeddings to 2D for visualization
    # perplexity must be less than n_samples
    perplexity = min(30, len(embeddings) - 1)
    tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity)
    return tsne.fit_transform(embeddings)

embeddings_2d = apply_tsne(np.array(embeddings))


# Topics used to classify each review by semantic similarity
categories = ["Quality", "Fit", "Style", "Comfort"]

# Generate embeddings for category labels
category_responses = openai_client.embeddings.create(input=categories, model=EMBEDDING_MODEL).model_dump()
category_embeddings = [emb["embedding"] for emb in category_responses["data"]]


def categorize_feedback(text_embedding, category_embeddings):
    # Assign the category whose embedding is closest (lowest cosine distance) to the review
    similarities = [{"distance": distance.cosine(text_embedding, cat_emb), "index": i}
                    for i, cat_emb in enumerate(category_embeddings)]
    closest = min(similarities, key=lambda x: x["distance"])
    return categories[closest["index"]]

# Classify every review into one of the four categories
feedback_categories = [categorize_feedback(emb, category_embeddings) for emb in embeddings]


def plot_tsne(tsne_results, labels=None):
    # Plot 2D t-SNE projection; color points by category label when provided
    plt.figure(figsize=(12, 8))
    if labels:
        unique_labels = list(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
        for label in unique_labels:
            indices = [i for i, l in enumerate(labels) if l == label]
            pts = tsne_results[indices]
            plt.scatter(pts[:, 0], pts[:, 1], alpha=0.5, label=label, color=color_map[label])
        plt.legend()
    else:
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5)
    plt.title("t-SNE Visualization of Review Embeddings")
    plt.xlabel("t-SNE feature 1")
    plt.ylabel("t-SNE feature 2")
    plt.show()

plot_tsne(embeddings_2d, labels=feedback_categories)


# Initialize persistent ChromaDB client for vector storage
chroma_client = chromadb.PersistentClient()

# Create collection with OpenAI embedding function so ChromaDB embeds queries automatically
review_embeddings_db = chroma_client.create_collection(
    name="review_embeddings",
    embedding_function=OpenAIEmbeddingFunction(model_name=EMBEDDING_MODEL, api_key=os.environ["OPENAI_API_KEY"]))

# Store all review texts; ChromaDB will embed them using the collection's embedding function
review_embeddings_db.add(
    documents=review_texts.tolist(),
    ids=[str(i) for i in range(len(review_texts))]
)


def find_similar_reviews(input_text, n=3):
    # Query the vector DB for the n most semantically similar reviews
    collection = chroma_client.get_collection(
        name="review_embeddings",
        embedding_function=OpenAIEmbeddingFunction(model_name=EMBEDDING_MODEL, api_key=os.environ["OPENAI_API_KEY"]))
    results = collection.query(query_texts=[input_text], n_results=n)
    return results

example_review = "Absolutely wonderful - silky and sexy and comfortable"
most_similar_reviews = find_similar_reviews(example_review, 3)["documents"][0]
print(most_similar_reviews)

# Remove collection to avoid duplicate key error on next run
chroma_client.delete_collection(name="review_embeddings")
