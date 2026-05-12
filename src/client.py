import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from openai import OpenAI
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# ── Setup ──────────────────────────────────────────────────────────────────────
load_dotenv()

client = OpenAI()

reviews = pd.read_csv("womens_clothing_e-commerce_reviews.csv")
review_texts = reviews["Review Text"].dropna().tolist()

# ── 1. Generate & store embeddings ────────────────────────────────────────────
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

embeddings = [get_embedding(text) for text in review_texts]

# ── 2. Dimensionality reduction → 2D ──────────────────────────────────────────
embeddings_array = np.array(embeddings)

tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
embeddings_2d = tsne.fit_transform(embeddings_array)

plt.figure(figsize=(12, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5, s=10, c="steelblue")
plt.title("2D Visualization of Women's Clothing Reviews (t-SNE)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.tight_layout()
plt.savefig("reviews_tsne.png", dpi=150)
plt.show()

# ── 3. Categorize feedback by theme ───────────────────────────────────────────
themes = {
    "quality":  "The quality of this product is excellent and well made.",
    "fit":      "This item fits perfectly, true to size.",
    "style":    "This is very stylish and fashionable.",
    "comfort":  "This is very comfortable to wear.",
    "price":    "The price is great value for money.",
}

theme_embeddings = {theme: get_embedding(desc) for theme, desc in themes.items()}

def get_top_theme(review_embedding):
    scores = {
        theme: cosine_similarity([review_embedding], [emb])[0][0]
        for theme, emb in theme_embeddings.items()
    }
    return max(scores, key=scores.get)

categorized = pd.DataFrame({
    "review":    review_texts[:50],   # sample first 50 for display
    "theme":     [get_top_theme(embeddings[i]) for i in range(50)],
})

print("\nTheme distribution (first 50 reviews):")
print(categorized["theme"].value_counts())

# ── 4. Similarity search function ─────────────────────────────────────────────
def find_similar_reviews(input_review, top_n=3):
    """Return top_n most similar reviews to input_review."""
    input_emb = np.array(get_embedding(input_review)).reshape(1, -1)
    sims = cosine_similarity(input_emb, embeddings_array)[0]
    # Exclude exact match (index 0 if input is review_texts[0])
    top_indices = np.argsort(sims)[::-1]
    results = []
    for idx in top_indices:
        if review_texts[idx] != input_review:
            results.append(review_texts[idx])
        if len(results) == top_n:
            break
    return results

first_review = "Absolutely wonderful - silky and sexy and comfortable"
most_similar_reviews = find_similar_reviews(first_review)

print("\nFirst review:")
print(f"  {first_review}")
print("\n3 most similar reviews:")
for i, r in enumerate(most_similar_reviews, 1):
    print(f"  {i}. {r[:120]}...")
