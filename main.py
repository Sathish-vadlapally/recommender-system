from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
from lightfm import LightFM

# Load model and encoders
with open("lightfm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("user_encoder.pkl", "rb") as f:
    user_encoder = pickle.load(f)

with open("item_encoder.pkl", "rb") as f:
    item_encoder = pickle.load(f)

# API setup
app = FastAPI()

# Input schema
class UserRequest(BaseModel):
    user_id: str  # or int, depending on your data

@app.post("/recommend")
def recommend_products(user_req: UserRequest):
    user_id = user_req.user_id

    # Check if user ID is known
    if user_id not in user_encoder.classes_:
        raise HTTPException(status_code=404, detail="User ID not found.")

    # Encode user ID to index
    user_idx = user_encoder.transform([user_id])[0]
    n_items = len(item_encoder.classes_)
    item_indices = np.arange(n_items)

    # Predict scores for all items
    scores = model.predict(user_ids=user_idx, item_ids=item_indices)

    # Get top-5 item indices
    top_indices = np.argsort(scores)[::-1][:5]
    recommended_items = item_encoder.inverse_transform(top_indices)

    return {"user_id": user_id, "recommended_products": recommended_items.tolist()}