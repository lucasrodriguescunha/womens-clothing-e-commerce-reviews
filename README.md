# Women's Clothing E-Commerce Reviews

Analyzes customer reviews from a women's clothing e-commerce dataset using OpenAI text embeddings, with dimensionality reduction, theme categorization, and similarity search for customer service support.

## Overview

Customers leave free-text feedback covering topics like fit, quality, style, and comfort. This pipeline embeds those reviews, visualizes their semantic structure, categorizes them by theme, and finds similar reviews for any given input.

**Key outputs:** embeddings list · 2D t-SNE array · theme categories · top-3 similar reviews

## How It Works

1. Load CSV of customer reviews (`Review Text` column)
2. Call `text-embedding-3-small` per review via OpenAI Embeddings API → store in `embeddings`
3. Apply t-SNE to reduce to 2D → store in `embeddings_2d`, plot scatter chart
4. Compare each review embedding against theme anchor embeddings (quality, fit, style, comfort, price) using cosine similarity → assign dominant theme
5. `find_similar_reviews(input_review)` returns 3 nearest reviews by cosine similarity

## Setup

```bash
pip install -r requirements.txt
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your_key_here
```

## Usage

```bash
python src/client.py
```

## Data

| Column | Description |
|---|---|
| `Review Text` | Free-text customer feedback on shopping experience and product quality |

Source: `src/data/` — CSV exported from DataLab (filename includes export timestamp).
