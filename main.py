from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

app = FastAPI()

# Load LightFM model and encoders
with open("lightfm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("user_encoder.pkl", "rb") as f:
    user_encoder = pickle.load(f)

with open("item_encoder.pkl", "rb") as f:
    item_encoder = pickle.load(f)

# Load product metadata
products_df = pd.read_csv("products.csv")  # should have 'product_id' and 'product_name'

@app.get("/")
def read_root():
    return {"message": "✅ LightFM Recommender API is running."}

@app.get("/recommend")
def recommend(user_id: int, k: int = 5):
    # Handle user_id format
    try:
        user_id = str(user_id) if user_encoder.classes_.dtype.type is np.str_ else int(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user_id format.")

    if user_id not in user_encoder.classes_:
        raise HTTPException(status_code=404, detail="User not found in training data.")

    user_idx = user_encoder.transform([user_id])[0]
    item_indices = np.arange(len(item_encoder.classes_))

    scores = model.predict(user_ids=np.repeat(user_idx, len(item_indices)),
                           item_ids=item_indices)

    top_k = np.argsort(-scores)[:k]
    recommended_ids = item_encoder.inverse_transform(top_k)

    # Get product names from product IDs
    recommended_names = []
    for pid in recommended_ids:
        name = products_df.loc[products_df['product_id'] == pid, 'product_name']
        recommended_names.append(name.values[0] if not name.empty else "Unknown Product")

    return {
        "user_id": user_id,
        "recommended_product_ids": recommended_ids.tolist(),
        "recommended_product_names": recommended_names
    }
