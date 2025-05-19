from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

app = FastAPI()
# app.mount("/static", StaticFiles(directory="."), name="static")
app.mount("/static", StaticFiles(directory="templates"), name="static")

# Загрузка эмбеддингов
with open('vectors.pkl', 'rb') as f:
    articles_data = pickle.load(f)

embeddings = np.array([a['embedding'] for a in articles_data])
filenames = [a['filename'] for a in articles_data]

# Модель для эмбеддингов
model_emb = SentenceTransformer('cointegrated/rubert-tiny2')

# LLM модель (замените на вашу)
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B-Instruct")

# слабая быстрая
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# tokenizer = AutoTokenizer.from_pretrained("NousResearch/Hermes-2-Pro-Mistral-7B")
# model = AutoModelForCausalLM.from_pretrained(
#     "NousResearch/Hermes-2-Pro-Mistral-7B",
#     torch_dtype=torch.float16,
#     device_map="auto"
# )

# LLM модель: Phi-3-mini
# tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
# model = AutoModelForCausalLM.from_pretrained(
#     "microsoft/Phi-3-mini-4k-instruct",
#     torch_dtype=torch.float16,
#     device_map="auto"
# )



pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.post("/query")
async def query(request: Request):
    data = await request.json()
    user_query = data.get("query", "")

    # Получаем эмбеддинг запроса
    query_emb = model_emb.encode(["query: " + user_query])

    # Сравниваем с нашими статьями
    similarities = cosine_similarity(query_emb, embeddings).flatten()
    best_idx = np.argmax(similarities)
    best_article = articles_data[best_idx]
    
    # Добавляем описание статьи к вопросу
    # extended_query = f"{user_query}\n\nКонтекст:\n{best_article['desc']}"
    # Ограничьте контекст до 2000 символов или токенов
    MAX_CONTEXT_LENGTH = 2000
    context = best_article['desc'][:MAX_CONTEXT_LENGTH]

    # extended_query = f"На основе следующего контекста ответь на вопрос:\n{context}\n\nВопрос: {user_query}"

    extended_query = f"""
    <|system|>: Отвечай на вопросы строго на основе предоставленного контекста.
    <|context|>: {context}
    <|question|>: {user_query}
    <|answer|>:
    """.strip()

    # Генерируем ответ
    response = pipe(extended_query, max_new_tokens=200)[0]['generated_text']

    return {"response": response}