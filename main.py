from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Load model and encoders
with open("lightfm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("user_encoder.pkl", "rb") as f:
    user_encoder = pickle.load(f)

with open("item_encoder.pkl", "rb") as f:
    item_encoder = pickle.load(f)

# Inverse mappings for decoding
item_decoder = {v: k for k, v in item_encoder.items()}

@app.get("/")
def read_root():
    return {"message": "LightFM Recommender API is running"}

@app.get("/recommend")
def recommend(user_id: int, k: int = 5):
    # Check if user_id exists
    if user_id not in user_encoder:
        raise HTTPException(status_code=404, detail="User not found in training data.")

    encoded_user = user_encoder[user_id]
    all_items = list(item_encoder.values())

    # Predict scores for all items for this user
    scores = model.predict(encoded_user, np.array(all_items))

    # Get top-k recommendations
    top_indices = np.argsort(scores)[::-1][:k]
    recommended_item_ids = [item_decoder[all_items[i]] for i in top_indices]

    return {"user_id": user_id, "recommended_items": recommended_item_ids}
