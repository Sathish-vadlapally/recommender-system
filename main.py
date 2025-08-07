from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import pickle
from lightfm import LightFM

# Initialize FastAPI app
app = FastAPI()

# CORS settings (you can adjust the origins if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and encoders
with open("lightfm_model.pkl", "rb") as f:
    model: LightFM = pickle.load(f)

with open("user_encoder.pkl", "rb") as f:
    user_encoder = pickle.load(f)

with open("item_encoder.pkl", "rb") as f:
    item_encoder = pickle.load(f)

# Reverse item_encoder to get product_id -> index and index -> product_id
item_decoder = {v: k for k, v in item_encoder.items()}
user_decoder = {v: k for k, v in user_encoder.items()}

# Load product metadata
products_df = pd.read_csv("products.csv")  # Must include columns: product_id, product_name


@app.get("/")
def root():
    return {"message": "LightFM Recommendation API is running"}


@app.get("/recommend/{user_id}")
def recommend_products(user_id: str, num_recommendations: int = 5):
    if user_id not in user_encoder:
        raise HTTPException(status_code=404, detail="User ID not found in training data")

    # Get internal user ID
    user_idx = user_encoder[user_id]

    # Prepare prediction for all items
    n_items = len(item_encoder)
    scores = model.predict(user_ids=user_idx, item_ids=np.arange(n_items))

    # Rank items
    top_items_idx = np.argsort(-scores)[:num_recommendations]

    # Decode to original product_ids
    recommended_product_ids = [item_decoder[i] for i in top_items_idx]

    # Get product names
    recommended_products = products_df[products_df["product_id"].isin(recommended_product_ids)]

    # Sort in same order as recommended_product_ids
    recommended_products["rank"] = recommended_products["product_id"].apply(lambda x: recommended_product_ids.index(x))
    recommended_products = recommended_products.sort_values("rank")

    return {
        "user_id": user_id,
        "recommendations": recommended_products[["product_id", "product_name"]].to_dict(orient="records")
    }
