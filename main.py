# -*- coding: utf-8 -*-
"""
Pet Product Search API - Optimized for Render deployment
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import re
from difflib import get_close_matches
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
import os
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Pet Product Search API", version="1.0.0")

# Add CORS middleware for Android app integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Android app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
db = None
model = None
df = None
product_embeddings = None
unique_keywords = set()

# Initialize Firebase
def init_firebase():
    global db
    try:
        # For Render, use environment variable for Firebase credentials
        firebase_key = os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY')
        if firebase_key:
            # Parse JSON from environment variable
            import json
            cred_dict = json.loads(firebase_key)
            cred = credentials.Certificate(cred_dict)
        else:
            # Fallback to service account file (for local development)
            cred = credentials.Certificate("firebase-key.json")
        
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("Firebase initialized successfully")
    except Exception as e:
        print(f"Firebase initialization error: {e}")
        raise e

# Initialize model
def init_model():
    global model
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("SentenceTransformer model loaded successfully")
    except Exception as e:
        print(f"Model initialization error: {e}")
        raise e

# Load product data from Firestore
def load_data_from_firestore():
    global df
    try:
        products_ref = db.collection("products")
        docs = products_ref.stream()
        data = []
        for doc in docs:
            product = doc.to_dict()
            product['id'] = doc.id
            data.append(product)
        
        if not data:
            raise ValueError("No products found in Firestore")
            
        df = pd.DataFrame(data)
        print(f"Loaded {len(df)} products from Firestore")
        return df
    except Exception as e:
        print(f"Error loading data from Firestore: {e}")
        raise e

# Load embeddings from Firestore
def load_embeddings_from_firestore():
    global product_embeddings
    try:
        embeddings_ref = db.collection("product_embeddings")
        docs = embeddings_ref.stream()
        embeddings_dict = {}
        
        for doc in docs:
            embedding_data = doc.to_dict()
            product_id = embedding_data.get('product_id')
            embedding = embedding_data.get('embedding')
            if product_id and embedding:
                embeddings_dict[product_id] = np.array(embedding)
        
        if not embeddings_dict:
            raise ValueError("No embeddings found in Firestore")
        
        # Reorder embeddings to match df order
        product_embeddings = []
        for _, row in df.iterrows():
            if row['id'] in embeddings_dict:
                product_embeddings.append(embeddings_dict[row['id']])
            else:
                # Generate embedding if missing
                full_text = clean_text(str(row.get('product_name', '')) + " " + 
                                     str(row.get('description', '')) + " " + 
                                     str(row.get('details', '')))
                embedding = model.encode([full_text])[0]
                product_embeddings.append(embedding)
        
        product_embeddings = np.array(product_embeddings)
        print(f"Loaded {len(product_embeddings)} embeddings")
        
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        # Generate embeddings if loading fails
        generate_embeddings()

# Generate embeddings if not available
def generate_embeddings():
    global product_embeddings
    try:
        print("Generating embeddings...")
        df["clean_product_name"] = df["product_name"].fillna("").apply(clean_text)
        df["clean_description"] = df["description"].fillna("").apply(clean_text)
        df["clean_details"] = df["details"].fillna("").apply(clean_text)
        df["full_text"] = (df["clean_product_name"] + " " + df["clean_product_name"] + " " +
                          df["clean_description"] + " " + df["clean_details"])
        
        product_texts = df["full_text"].tolist()
        product_embeddings = model.encode(product_texts, show_progress_bar=False)
        print("Embeddings generated successfully")
        
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        raise e

# Preprocessing functions
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_price(price):
    if pd.isna(price):
        return 0
    try:
        return float(str(price).replace(",", "").strip())
    except:
        return 0

def extract_parent_name(name):
    if pd.isna(name):
        return ""
    if '-' in name:
        return name.split('-')[0].strip()
    elif '(' in name:
        return name.split('(')[0].strip()
    else:
        return name.strip()

# Initialize keywords for autocomplete
def init_keywords():
    global unique_keywords
    try:
        for name in df["product_name"].dropna().unique():
            unique_keywords.add(clean_text(name))
            for word in name.lower().split():
                if word:
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
        
        print(f"Initialized {len(unique_keywords)} keywords")
        
    except Exception as e:
        print(f"Error initializing keywords: {e}")

# Spell correction
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

# Autocomplete functions
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

# Price parsing
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

# Search function
def search_sbert(query, top_n=10, price_min=None, price_max=None):
    try:
        clean_query = clean_text(query)
        query_embedding = model.encode([clean_query])
        scores = cosine_similarity(query_embedding, product_embeddings).flatten()
        
        df_search = df.copy()
        df_search["score"] = scores
        df_search["price_clean"] = df_search["price"].apply(clean_price)
        df_search["parent_product_name"] = df_search["product_name"].apply(extract_parent_name)
        
        if price_min is not None:
            df_search = df_search[df_search["price_clean"] >= price_min]
        if price_max is not None:
            df_search = df_search[df_search["price_clean"] <= price_max]
        
        if df_search.empty:
            return pd.DataFrame(columns=["id", "parent_product_name", "price", "score"])
        
        grouped = df_search.groupby("parent_product_name").agg({
            "score": "max",
            "price_clean": "min", 
            "id": "first"
        }).reset_index()
        
        grouped = grouped.sort_values(by="score", ascending=False).head(top_n)
        grouped = grouped.rename(columns={"price_clean": "price"})
        return grouped[["id", "parent_product_name", "price", "score"]]
        
    except Exception as e:
        print(f"Search error: {e}")
        return pd.DataFrame(columns=["id", "parent_product_name", "price", "score"])

# Similar items function
def find_similar_items(product_id, top_n=5):
    try:
        if product_id not in df['id'].values:
            return {"error": "Product ID not found"}
        
        product_idx = df[df['id'] == product_id].index[0]
        target_embedding = product_embeddings[product_idx].reshape(1, -1)
        
        similarities = cosine_similarity(target_embedding, product_embeddings).flatten()
        top_indices = np.argsort(similarities)[::-1][1:top_n+1]  # Exclude the product itself
        
        similar_products = df.iloc[top_indices][['id', 'product_name', 'price']].copy()
        similar_products['price'] = similar_products['price'].apply(clean_price)
        
        return similar_products.to_dict(orient='records')
        
    except Exception as e:
        print(f"Similar items error: {e}")
        return {"error": str(e)}

# Pydantic models
class SearchRequest(BaseModel):
    query: str
    top_n: int = 10

class AutocompleteRequest(BaseModel):
    input_text: str
    limit: int = 5

class SimilarItemsRequest(BaseModel):
    product_id: str
    top_n: int = 5

# API endpoints
@app.get("/")
async def root():
    return {"message": "Pet Product Search API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "products_loaded": len(df) if df is not None else 0}

@app.post("/search")
async def search_products(request: SearchRequest):
    try:
        if df is None or product_embeddings is None:
            raise HTTPException(status_code=503, detail="Service not ready")
        
        query, price_min, price_max = parse_price_range(request.query)
        corrected_query = correct_full_query(query, unique_keywords)
        results = search_sbert(corrected_query, request.top_n, price_min, price_max)
        
        return {
            "original_query": request.query,
            "corrected_query": corrected_query if corrected_query != query else None,
            "results": results.to_dict(orient="records"),
            "total_results": len(results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/autocomplete")
async def autocomplete(request: AutocompleteRequest):
    try:
        if not unique_keywords:
            raise HTTPException(status_code=503, detail="Keywords not loaded")
        
        result = autocomplete_with_correction(request.input_text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/similar_items")
async def get_similar_items(request: SimilarItemsRequest):
    try:
        if df is None or product_embeddings is None:
            raise HTTPException(status_code=503, detail="Service not ready")
        
        results = find_similar_items(request.product_id, request.top_n)
        if "error" in results:
            raise HTTPException(status_code=404, detail=results["error"])
        return {"results": results}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Startup event
@app.on_event("startup")
async def startup_event():
    try:
        print("Starting up application...")
        init_firebase()
        init_model()
        load_data_from_firestore()
        load_embeddings_from_firestore()
        init_keywords()
        print("Application startup completed successfully")
    except Exception as e:
        print(f"Startup error: {e}")
        raise e

# Main function for running the app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
