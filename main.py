from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import re
from difflib import get_close_matches
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
import json
import os



# Khởi tạo app FastAPI
app = FastAPI()

# Khởi tạo Firebase

firebase_key = os.environ.get("FIREBASE_SERVICE_ACCOUNT_KEY")
cred = credentials.Certificate(json.loads(firebase_key))firebase_admin.initialize_app(cred)
db = firestore.client()

# Load model SBERT
model = SentenceTransformer('all-MiniLM-L6-v2')

# Lấy dữ liệu từ Firestore
def load_data_from_firestore():
    products_ref = db.collection("products")
    docs = products_ref.stream()
    data = []
    for doc in docs:
        product = doc.to_dict()
        product['id'] = doc.id
        data.append(product)
    return pd.DataFrame(data)

df = load_data_from_firestore()

# Tiền xử lý
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_product_name"] = df["product_name"].fillna("").apply(clean_text)
df["clean_description"] = df["description"].fillna("").apply(clean_text)
df["clean_details"] = df["details"].fillna("").apply(clean_text)
df["full_text"] = (df["clean_product_name"] + " " + df["clean_product_name"] + " " +
                   df["clean_description"] + " " + df["clean_details"])

def clean_price(price):
    if pd.isna(price):
        return 0
    try:
        return float(str(price).replace(",", "").strip())
    except:
        return 0
df["price_clean"] = df["price"].apply(clean_price)

def extract_parent_name(name):
    if pd.isna(name):
        return ""
    if '-' in name:
        return name.split('-')[0].strip()
    elif '(' in name:
        return name.split('(')[0].strip()
    else:
        return name.strip()
df["parent_product_name"] = df["product_name"].apply(extract_parent_name)

# Sinh embedding
product_texts = df["full_text"]
product_embeddings = model.encode(product_texts.tolist(), show_progress_bar=True)

# Tạo danh sách từ khóa autocomplete + sửa chính tả
unique_keywords = set()
for name in df["product_name"].dropna().unique():
    unique_keywords.add(clean_text(name))
    for word in name.lower().split():
        unique_keywords.add(word)
categories_and_brands = [
    'Accessories', 'Apparel & Costume', 'Bed', 'Blanket', 'Brushes & Combs', 'Carriers & Kennels',
    'Cat', 'Collar', 'Collar & Leash', 'Costume', 'Dental care', 'Deodorant tools', 'Dog', 'Dry food',
    'Feeders', 'Flea and Tick control', 'Food', 'Hammock', 'Leash', 'Nail care', 'Pillow', 'Set',
    'Shampoo & Conditioner', 'Small Animal', 'Supplements & Vitamins', 'Toys', 'Training', 'Treat',
    'Wet food', 'catnip', '3 Peaks', 'BAM!', 'Barkbutler', 'Basil', 'Chuckit!', 'Coachi', 'M-PETS',
    'Mitag', 'Noble', 'PAW', 'Papa Pawsome', 'Papaw Cartel', 'Pawgypets', 'Pawsome Couture',
    'Pedigree', 'Pets at Home', 'QPets', 'Squeeezys', 'TOPDOG', 'Trixie', 'dog food', 'cat food'
]
for kw in categories_and_brands:
    parts = kw.split()
    unique_keywords.add(clean_text(kw))
    for p in parts:
        ckw = clean_text(p)
        if ckw:
            unique_keywords.add(ckw)

# Sửa lỗi chính tả
misspellings = {
    "fodd": "food",
    "catnipp": "catnip",
    "collor": "collar",
    "leesh": "leash"
}

def correct_spelling(word, keyword_list):
    word_lower = word.lower()
    if word_lower in misspellings:
        return misspellings[word_lower]
    if word_lower in keyword_list:
        return word_lower
    matches = get_close_matches(word_lower, keyword_list, n=1, cutoff=0.6)
    return matches[0] if matches else word_lower

def correct_full_query(query, keyword_list):
    words = query.lower().split()
    corrected_words = [correct_spelling(w, keyword_list) for w in words]
    return " ".join(corrected_words)

# Gợi ý từ khóa
def autocomplete_prefix(input_text, keyword_list, limit=5):
    input_text = input_text.lower()
    suggestions = []
    suggestions.extend(kw for kw in keyword_list if kw.startswith(input_text))
    suggestions.extend(kw for kw in keyword_list if input_text in kw and not kw.startswith(input_text))
    return sorted(set(suggestions))[:limit]

def autocomplete_with_correction(input_text):
    input_text = input_text.lower()
    corrected = correct_spelling(input_text, list(unique_keywords))
    if corrected and corrected != input_text:
        fallback = autocomplete_prefix(corrected, unique_keywords)
        return {"type": "spell_corrected", "input": input_text, "suggestions": fallback, "correction": corrected}
    suggestions = autocomplete_prefix(input_text, unique_keywords)
    if suggestions:
        return {"type": "autocomplete", "input": input_text, "suggestions": suggestions, "correction": None}
    return {"type": "no_match", "input": input_text, "suggestions": [], "correction": None}

# Tách giá trong câu truy vấn
def parse_price_range(query):
    price_min = None
    price_max = None
    m = re.search(r'price\s+under\s+([\d.,]+)', query, re.IGNORECASE)
    if m:
        price_max = float(re.sub(r"[.,]", "", m.group(1)))
        query = re.sub(r'price\s+under\s+[\d.,]+', '', query, flags=re.IGNORECASE).strip()
        return query, price_min, price_max
    m = re.search(r'price\s+over\s+([\d.,]+)', query, re.IGNORECASE)
    if m:
        price_min = float(re.sub(r"[.,]", "", m.group(1)))
        query = re.sub(r'price\s+over\s+[\d.,]+', '', query, flags=re.IGNORECASE).strip()
        return query, price_min, price_max
    m = re.search(r'price\s+from\s+([\d.,]+)\s+to\s+([\d.,]+)', query, re.IGNORECASE)
    if m:
        price_min = float(re.sub(r"[.,]", "", m.group(1)))
        price_max = float(re.sub(r"[.,]", "", m.group(2)))
        query = re.sub(r'price\s+from\s+[\d.,]+\s+to\s+[\d.,]+', '', query, flags=re.IGNORECASE).strip()
        return query, price_min, price_max
    return query.strip(), price_min, price_max

# API tìm kiếm
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
    }).reset_index()
    grouped = grouped.sort_values(by="score", ascending=False).head(request.top_n)
    grouped = grouped.rename(columns={"price_clean": "price"})
    return {
        "corrected_query": corrected_query if corrected_query != request.query else None,
        "results": grouped[["id", "parent_product_name", "price", "score"]].to_dict(orient="records")
    }

# API autocomplete
class AutocompleteRequest(BaseModel):
    input_text: str
    limit: int = 5

@app.post("/autocomplete")
async def autocomplete(request: AutocompleteRequest):
    return autocomplete_with_correction(request.input_text)

# API gợi ý sản phẩm tương tự
class SimilarItemsRequest(BaseModel):
    product_id: str
    top_n: int = 5

@app.post("/similar_items")
async def get_similar_items(request: SimilarItemsRequest):
    if request.product_id not in df['id'].values:
        raise HTTPException(status_code=404, detail="Product ID not found")
    target_embedding = model.encode([df[df['id'] == request.product_id]["full_text"].values[0]])
    similarities = cosine_similarity(target_embedding, product_embeddings).flatten()
    df_copy = df.copy()
    df_copy["similarity"] = similarities
    df_copy = df_copy[df_copy["id"] != request.product_id]
    top_items = df_copy.sort_values(by="similarity", ascending=False).head(request.top_n)
    return top_items[["id", "product_name", "price"]].to_dict(orient="records")
