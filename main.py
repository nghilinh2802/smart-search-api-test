
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import firebase_admin
from firebase_admin import credentials, firestore
import json, os

app = FastAPI()

# Firebase khởi tạo từ biến môi trường (dành cho Render)
firebase_key = os.environ.get("FIREBASE_SERVICE_ACCOUNT_KEY")
cred = credentials.Certificate(json.loads(firebase_key))
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Tải product data từ Firestore
def load_product_data():
    docs = db.collection("products").stream()
    products = []
    for doc in docs:
        item = doc.to_dict()
        item["id"] = doc.id
        products.append(item)
    return pd.DataFrame(products)

df = load_product_data()

# Preprocess
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_product_name"] = df["product_name"].fillna("").apply(clean_text)
df["clean_description"] = df["description"].fillna("").apply(clean_text)
df["clean_details"] = df["details"].fillna("").apply(clean_text)
df["full_text"] = df["clean_product_name"] + " " + df["clean_product_name"] + " " + df["clean_description"] + " " + df["clean_details"]
df["price_clean"] = df["price"].apply(lambda p: float(str(p).replace(",", "").strip()) if p else 0)
df["parent_product_name"] = df["product_name"].apply(lambda name: name.split("-")[0].strip() if "-" in name else name)

# Load embedding từ Firestore
embedding_docs = db.collection("product_embeddings").stream()
embedding_map = {}
for doc in embedding_docs:
    data = doc.to_dict()
    embedding_map[data["product_id"]] = np.array(data["embedding"])

product_embeddings = []
valid_ids = []
for idx, row in df.iterrows():
    pid = row["id"]
    if pid in embedding_map:
        product_embeddings.append(embedding_map[pid])
        valid_ids.append(pid)

# Chỉ giữ lại các sản phẩm có embedding
df = df[df["id"].isin(valid_ids)].reset_index(drop=True)
product_embeddings = np.stack(product_embeddings)

# Từ khóa gợi ý
unique_keywords = set()
for name in df["product_name"].dropna().unique():
    unique_keywords.add(clean_text(name))
    for word in name.lower().split():
        unique_keywords.add(word)

misspellings = {"fodd": "food", "catnipp": "catnip", "collor": "collar", "leesh": "leash"}

def correct_spelling(word, keyword_list):
    from difflib import get_close_matches
    word_lower = word.lower()
    if word_lower in misspellings:
        return misspellings[word_lower]
    if word_lower in keyword_list:
        return word_lower
    matches = get_close_matches(word_lower, keyword_list, n=1, cutoff=0.6)
    return matches[0] if matches else word_lower

def correct_full_query(query, keyword_list):
    return " ".join([correct_spelling(w, keyword_list) for w in query.lower().split()])

def autocomplete_prefix(input_text, keyword_list, limit=5):
    input_text = input_text.lower()
    suggestions = [kw for kw in keyword_list if input_text in kw]
    return sorted(set(suggestions))[:limit]

def autocomplete_with_correction(input_text):
    corrected = correct_spelling(input_text, list(unique_keywords))
    if corrected and corrected != input_text:
        fallback = autocomplete_prefix(corrected, unique_keywords)
        return {"type": "spell_corrected", "input": input_text, "suggestions": fallback, "correction": corrected}
    suggestions = autocomplete_prefix(input_text, unique_keywords)
    return {"type": "autocomplete" if suggestions else "no_match", "input": input_text, "suggestions": suggestions, "correction": None}

def parse_price_range(query):
    price_min = price_max = None
    m = re.search(r'price\s+under\s+([\d.,]+)', query, re.IGNORECASE)
    if m:
        price_max = float(re.sub(r"[.,]", "", m.group(1)))
        query = re.sub(r'price\s+under\s+[\d.,]+', '', query, flags=re.IGNORECASE).strip()
    return query.strip(), price_min, price_max

# API
class SearchRequest(BaseModel):
    query: str
    top_n: int = 10

@app.post("/search")
async def search_products(request: SearchRequest):
    query, price_min, price_max = parse_price_range(request.query)
    corrected_query = correct_full_query(query, unique_keywords)
    query_embedding = model.encode([clean_text(corrected_query)])
    scores = cosine_similarity(query_embedding, product_embeddings).flatten()
    df_search = df.copy()
    df_search["score"] = scores
    if price_min is not None:
        df_search = df_search[df_search["price_clean"] >= price_min]
    if price_max is not None:
        df_search = df_search[df_search["price_clean"] <= price_max]
    grouped = df_search.groupby("parent_product_name").agg({
        "score": "max",
        "price_clean": "min",
        "id": "first"
    }).reset_index().sort_values(by="score", ascending=False).head(request.top_n)
    grouped = grouped.rename(columns={"price_clean": "price"})
    return {
        "corrected_query": corrected_query if corrected_query != request.query else None,
        "results": grouped[["id", "parent_product_name", "price", "score"]].to_dict(orient="records")
    }

class AutocompleteRequest(BaseModel):
    input_text: str
    limit: int = 5

@app.post("/autocomplete")
async def autocomplete(request: AutocompleteRequest):
    return autocomplete_with_correction(request.input_text)
