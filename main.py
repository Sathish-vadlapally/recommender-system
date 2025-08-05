from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
from lightfm import LightFM
from scipy.sparse import csr_matrix

app = FastAPI()

# Load model and encoders
with open("lightfm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("user_encoder.pkl", "rb") as f:
    user_enc = pickle.load(f)

with open("item_encoder.pkl", "rb") as f:
    item_enc = pickle.load(f)

@app.get("/")
def root():
    return {"message": "LightFM Recommender is live!"}

@app.get("/recommend/{user_id}")
def recommend(user_id: str, k: int = 5):
    if user_id not in user_enc.classes_:
        raise HTTPException(status_code=404, detail="User not found")

    user_idx = user_enc.transform([user_id])[0]
    n_items = len(item_enc.classes_)

    scores = model.predict(user_ids=user_idx, item_ids=np.arange(n_items))
    top_items = np.argsort(scores)[::-1][:k]
    recommended_items = item_enc.inverse_transform(top_items)

    return {
        "user_id": user_id,
        "recommended_products": recommended_items.tolist()
    }
