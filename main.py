from fastapi import FastAPI, HTTPException
import numpy as np
import pickle
from lightfm import LightFM

app = FastAPI()

# Load model and encoders
try:
    with open("lightfm_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("user_encoder.pkl", "rb") as f:
        user_encoder = pickle.load(f)

    with open("item_encoder.pkl", "rb") as f:
        item_encoder = pickle.load(f)

    n_items = len(item_encoder.classes_)

except Exception as e:
    raise RuntimeError(f"Failed to load model or encoders: {e}")


@app.get("/")
def root():
    return {"message": "✅ Recommender API is running."}


@app.get("/recommend/{user_id}")
def recommend(user_id: str, k: int = 5):
    try:
        # Check if user exists in encoder
        if user_id not in user_encoder.classes_:
            raise HTTPException(status_code=404, detail="User not found in training data.")

        # Encode user
        user_idx = user_encoder.transform([user_id])[0]

        # Predict scores for all items
        scores = model.predict(user_ids=np.repeat(user_idx, n_items), item_ids=np.arange(n_items))

        # Get top-k recommendations
        top_k_item_indices = np.argsort(-scores)[:k]
        top_k_product_ids = item_encoder.inverse_transform(top_k_item_indices)

        return {
            "user_id": user_id,
            "recommended_products": top_k_product_ids.tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
