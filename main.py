from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
from lightfm import LightFM

app = FastAPI(title="🛒 Recommender System API")

# Load model and encoders
with open("lightfm_model.pkl", "rb") as f:
    model: LightFM = pickle.load(f)

with open("user_encoder.pkl", "rb") as f:
    user_encoder = pickle.load(f)

with open("item_encoder.pkl", "rb") as f:
    item_encoder = pickle.load(f)

# Total number of items in training
n_items = len(item_encoder.classes_)

class UserRequest(BaseModel):
    user_id: int

@app.get("/")
def read_root():
    return {"message": "✅ Recommender API is up. Use POST /recommend with user_id."}

@app.post("/recommend")
def recommend(user_req: UserRequest):
    try:
        user_id = user_req.user_id

        # Convert user_id to internal encoding
        if user_id not in user_encoder.classes_:
            raise HTTPException(status_code=404, detail="404: User not found in training data.")

        user_idx = user_encoder.transform([user_id])[0]

        # Predict scores for all items
        scores = model.predict(user_ids=user_idx, item_ids=np.arange(n_items))

        # Top 5 recommended item indices
        top_items = np.argsort(-scores)[:5]

        # Decode item indices back to original product IDs
        recommended_products = item_encoder.inverse_transform(top_items)

        return {
            "user_id": user_id,
            "recommended_product_ids": recommended_products.tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
