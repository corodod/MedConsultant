import os
import re
import pickle
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
ARTICLES_DIR = 'articles'
VECTOR_FILE = 'vectors.pkl'

model = SentenceTransformer(MODEL_NAME)

def parse_articles():
    articles_data = []

    for filename in os.listdir(ARTICLES_DIR):
        if not filename.endswith('.txt'):
            continue
        with open(os.path.join(ARTICLES_DIR, filename), 'r', encoding='utf-8') as f:
            text = f.read()

        name_match = re.search(r'\{\{NAME\}\}(.*?)\{\{\/NAME\}\}', text, re.DOTALL)
        desc_match = re.search(r'\{\{DESC\}\}(.*?)\{\{\/DESC\}\}', text, re.DOTALL)

        if name_match:
            name = name_match.group(1).strip()
            desc = desc_match.group(1).strip() if desc_match else ''
            embedding = model.encode(name)  # просто название статьи
            articles_data.append({
                'filename': filename,
                'name': name,
                'desc': desc,
                'embedding': embedding
            })

    with open(VECTOR_FILE, 'wb') as f:
        pickle.dump(articles_data, f)

    print(f"Processed {len(articles_data)} articles and saved embeddings to {VECTOR_FILE}")

if __name__ == '__main__':
    parse_articles()
