import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import re
from difflib import get_close_matches
from sentence_transformers import SentenceTransformer
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
import json
import asyncio
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pet Product Search API", version="1.0.0")

# Global variables
model = None
df = None
product_embeddings = None
unique_keywords = set()
db = None

def initialize_firebase():
    global db
    try:
        firebase_creds = os.getenv('FIREBASE_CREDENTIALS')
        if not firebase_creds:
            raise ValueError("FIREBASE_CREDENTIALS environment variable not set")
        cred_dict = json.loads(firebase_creds)
        cred = credentials.Certificate(cred_dict)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        logger.info("Firebase initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {str(e)}")
        return False

def initialize_model():
    global model
    try:
        logger.info("Loading SentenceTransformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

async def load_data_from_firestore():
    global df, product_embeddings, unique_keywords
    try:
        logger.info("Loading data from Firestore...")
        products_ref = db.collection("products")
        docs = products_ref.stream()
        data = []
        for doc in docs:
            product = doc.to_dict()
            product['id'] = doc.id
            data.append(product)
        if not data:
            logger.warning("No products found in Firestore")
            return False
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} products")
        preprocess_data()
        await load_embeddings_from_firestore()
        generate_keywords()
        logger.info("Data initialization completed successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        return False

async def load_embeddings_from_firestore():
    global product_embeddings
    try:
        logger.info("Loading embeddings from Firestore...")
        embeddings_ref = db.collection("product_embeddings")
        docs = embeddings_ref.stream()
        embeddings_dict = {}
        for doc in docs:
            data = doc.to_dict()
            embeddings_dict[data['product_id']] = np.array(data['embedding'])
        product_embeddings = np.array([
            embeddings_dict[product_id] for product_id in df['id']
        ])
        logger.info("Embeddings loaded from Firestore successfully")
    except Exception as e:
        logger.warning(f"Failed to load embeddings from Firestore: {str(e)}")

def preprocess_data():
    global df
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

def generate_keywords():
    global unique_keywords
    unique_keywords = set()
    for name in df["product_name"].dropna().unique():
        clean_name = str(name).lower().strip()
        unique_keywords.add(clean_name)
        for word in clean_name.split():
            if len(word) > 2:
                unique_keywords.add(word)
    for kw in ['Accessories', 'Apparel', 'Blanket', 'Brushes', 'Cat', 'Dog', 'Dry food',
               'Shampoo', 'Treat', 'Toys', 'Wet food', 'catnip', 'collar', 'leash']:
        clean_kw = kw.lower().strip()
        unique_keywords.add(clean_kw)
        for word in clean_kw.split():
            if len(word) > 2:
                unique_keywords.add(word)

misspellings = {"fodd": "food", "catnipp": "catnip", "collor": "collar", "leesh": "leash"}

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
    return " ".join([correct_spelling(w, keyword_list) for w in words])

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
    if suggestions:
        return {"type": "autocomplete", "input": input_text, "suggestions": suggestions, "correction": None}
    return {"type": "no_match", "input": input_text, "suggestions": [], "correction": None}

def parse_price_range(query):
    price_min = price_max = None
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

def search_sbert(query, top_n=10, price_min=None, price_max=None):
    if df is None or product_embeddings is None:
        raise HTTPException(status_code=503, detail="Service not ready. Data not loaded.")
    clean_query = re.sub(r"[^a-z0-9\s]", " ", query.lower()).strip()
    query_embedding = model.encode([clean_query])
    scores = np.dot(product_embeddings, query_embedding[0]) / (
        np.linalg.norm(product_embeddings, axis=1) * np.linalg.norm(query_embedding))
    df_search = df.copy()
    df_search["score"] = scores
    if price_min is not None:
        df_search = df_search[df_search["price_clean"] >= price_min]
    if price_max is not None:
        df_search = df_search[df_search["price_clean"] <= price_max]
    grouped = df_search.groupby("parent_product_name").agg({"score": "max", "price_clean": "min", "id": "first"}).reset_index()
    grouped = grouped.sort_values(by="score", ascending=False).head(top_n)
    grouped = grouped.rename(columns={"price_clean": "price"})
    return grouped[["id", "parent_product_name", "price", "score"]]

class SearchRequest(BaseModel):
    query: str
    top_n: int = 10

class AutocompleteRequest(BaseModel):
    input_text: str
    limit: int = 5

@app.on_event("startup")
async def startup_event():
    logger.info("Starting application initialization...")
    if not initialize_firebase():
        return
    if not initialize_model():
        return
    if not await load_data_from_firestore():
        return
    logger.info("Application ready.")

@app.get("/")
async def root():
    return {"message": "Pet Product Search API", "status": "running"}

@app.get("/health")
async def health_check():
    status = {
        "status": "healthy",
        "firebase_connected": db is not None,
        "model_loaded": model is not None,
        "data_loaded": df is not None,
        "embeddings_ready": product_embeddings is not None
    }
    if not all(status.values()):
        raise HTTPException(status_code=503, detail="Service not ready")
    return status

@app.post("/search")
async def search_products(request: SearchRequest):
    query, price_min, price_max = parse_price_range(request.query)
    corrected_query = correct_full_query(query, unique_keywords)
    results = search_sbert(corrected_query, request.top_n, price_min, price_max)
    return {
        "corrected_query": corrected_query if corrected_query != query else None,
        "results": results.to_dict(orient="records")
    }

@app.post("/autocomplete")
async def autocomplete(request: AutocompleteRequest):
    return autocomplete_with_correction(request.input_text)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
