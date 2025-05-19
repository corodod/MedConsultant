from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import pipeline as question_answering_pipeline
from transformers import pipeline

app = FastAPI()
# app.mount("/static", StaticFiles(directory="."), name="static")
app.mount("/static", StaticFiles(directory="templates"), name="static")

# Загрузка эмбеддингов
with open('vectors.pkl', 'rb') as f:
    articles_data = pickle.load(f)

embeddings = np.array([a['embedding'] for a in articles_data])
filenames = [a['filename'] for a in articles_data]

# Модель для эмбеддингов
model_emb = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

# слабая быстрая
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")


# Модель для извлечения ключевого фрагмента (extractive QA)для извлечения фрагмента из {{DESC}}
# qa_pipe = question_answering_pipeline("question-answering", model="cointegrated/rubert-tiny-qa")
qa_pipe = pipeline(
    "question-answering",
    model="abletobetable/distilbert-ru-qa",
    tokenizer="distilbert-base-multilingual-cased"
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.post("/query")
async def query(request: Request):
    print(">>> [LOG] Обработчик /query вызван")
    data = await request.json()
    user_query = data.get("query", "")

    # Шаг 1: Находим ближайшую статью
    query_emb = model_emb.encode(["query: " + user_query])
    similarities = cosine_similarity(query_emb, embeddings).flatten()
    top_k = 3
    top_k_indices = similarities.argsort()[-top_k:][::-1]  # Топ-3 по убыванию
    top_articles = [articles_data[idx] for idx in top_k_indices]
    combined_context = "\n\n".join([a["desc"] for a in top_articles])

    print("Топ-3 выбранные статьи:")
    for idx, article in enumerate(top_articles, 1):
        print(f"{idx}. {article['name']}")


    # Шаг 2: Извлекаем релевантный фрагмент из {{DESC}}
    context = combined_context
    question_for_qa = user_query
    print(">>> QA-пайп начинается")
    result = qa_pipe(question=question_for_qa, context=context)
    print(f"QA pipe результат: {result}")

    answer_start = result["start"]
    answer_end = result["end"] + 150  # Добавляем окружение
    relevant_section = context[max(0, answer_start - 100):min(len(context), answer_end)]
    print(f"\n=== Контекст для ответа ===\n{relevant_section}\n")


    # Шаг 3: Формируем prompt для LLM
    extended_query = f"""
<|system|>: Отвечай на вопросы строго на основе предоставленного контекста.
<|context|>: {relevant_section}
<|question|>: {user_query}
<|answer|>:
""".strip()

    print(">>> Начинается генерация ответа")
    # Шаг 4: Генерируем ответ
    response = pipe(
        extended_query,
        max_new_tokens=100,
        temperature=0.7,
        truncation=True,
        return_full_text=False
    )[0]['generated_text']

    return {"response": response.strip()}