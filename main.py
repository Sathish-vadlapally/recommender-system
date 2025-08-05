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

# Build item index to product_id decoder
item_decoder = {i: item_encoder.classes_[i] for i in range(len(item_encoder.classes_))}

@app.get("/")
def read_root():
    return {"message": "LightFM Recommender API is running"}

@app.get("/recommend")
def recommend(user_id: int, k: int = 5):
    # ✅ FIXED check here
    if user_id not in user_encoder.classes_:
        raise HTTPException(status_code=404, detail="User not found in training data.")

    encoded_user = user_encoder.transform([user_id])[0]

    n_items = len(item_encoder.classes_)
    scores = model.predict(user_ids=np.repeat(encoded_user, n_items),
                           item_ids=np.arange(n_items))

    top_k = np.argsort(-scores)[:k]
    recommended_items = [item_decoder[i] for i in top_k]

    return {"user_id": user_id, "recommended_items": recommended_items}
