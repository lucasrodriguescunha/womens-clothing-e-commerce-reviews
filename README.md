# Análise de Tópicos de Avaliações de Roupas com Embeddings / Analysis of Clothing Review Topics with Embeddings

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B35?style=for-the-badge&logo=databricks&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)

---

## PT-BR

Analisa avaliações de clientes de um e-commerce de roupas femininas usando embeddings de texto da OpenAI, com redução de dimensionalidade, categorização por temas e busca por similaridade para suporte ao cliente.

### Visão Geral

Clientes deixam feedback em texto livre cobrindo tópicos como caimento, qualidade, estilo e conforto. Este pipeline incorpora as avaliações, visualiza sua estrutura semântica, as categoriza por tema e encontra avaliações similares para qualquer entrada.

**Saídas principais:** lista de embeddings · array t-SNE 2D · categorias por tema · top-3 avaliações similares

### Como Funciona

1. Carrega CSV de avaliações (`Review Text`)
2. Chama `text-embedding-3-small` por avaliação via OpenAI Embeddings API → salva em `embeddings`
3. Aplica t-SNE para reduzir a 2D → salva em `embeddings_2d`, plota gráfico de dispersão
4. Compara cada embedding com embeddings âncora por tema (qualidade, caimento, estilo, conforto, preço) via similaridade de cosseno → atribui tema dominante
5. `find_similar_reviews(input_review)` retorna 3 avaliações mais próximas por similaridade de cosseno

### Configuração

```bash
pip install -r requirements.txt
```

Defina sua chave da API OpenAI:

```bash
export OPENAI_API_KEY=sua_chave_aqui
```

### Uso

```bash
python src/client.py
```

### Dados

| Coluna | Descrição |
|---|---|
| `Review Text` | Feedback em texto livre sobre experiência de compra e qualidade do produto |

Fonte: `src/data/` — CSV exportado do DataLab (nome do arquivo inclui timestamp de exportação).

---

## EN-US

Analyzes customer reviews from a women's clothing e-commerce dataset using OpenAI text embeddings, with dimensionality reduction, theme categorization, and similarity search for customer service support.

### Overview

Customers leave free-text feedback covering topics like fit, quality, style, and comfort. This pipeline embeds those reviews, visualizes their semantic structure, categorizes them by theme, and finds similar reviews for any given input.

**Key outputs:** embeddings list · 2D t-SNE array · theme categories · top-3 similar reviews

### How It Works

1. Load CSV of customer reviews (`Review Text` column)
2. Call `text-embedding-3-small` per review via OpenAI Embeddings API → store in `embeddings`
3. Apply t-SNE to reduce to 2D → store in `embeddings_2d`, plot scatter chart
4. Compare each review embedding against theme anchor embeddings (quality, fit, style, comfort, price) using cosine similarity → assign dominant theme
5. `find_similar_reviews(input_review)` returns 3 nearest reviews by cosine similarity

### Setup

```bash
pip install -r requirements.txt
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY=your_key_here
```

### Usage

```bash
python src/client.py
```

### Data

| Column | Description |
|---|---|
| `Review Text` | Free-text customer feedback on shopping experience and product quality |

Source: `src/data/` — CSV exported from DataLab (filename includes export timestamp).
