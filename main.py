from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
from lightfm import LightFM
from scipy.sparse import load_npz

app = FastAPI()

# Load the model and required files
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

interactions = load_npz("interactions.npz")

with open("user_id_map.pkl", "rb") as f:
    user_id_map = pickle.load(f)

with open("item_id_map.pkl", "rb") as f:
    item_id_map = pickle.load(f)

# Reverse item_id_map to get actual product IDs
reverse_item_id_map = {v: k for k, v in item_id_map.items()}


class RecommendationRequest(BaseModel):
    user_id: str
    num_recommendations: int = 5


@app.get("/")
def read_root():
    return {"message": "LightFM recommender system is running."}


@app.post("/recommend/")
def recommend(request: RecommendationRequest):
    user_id = request.user_id
    N = request.num_recommendations

    if user_id not in user_id_map:
        raise HTTPException(status_code=404, detail="User ID not found in dataset.")

    user_index = user_id_map[user_id]

    # Predict scores for all items
    scores = model.predict(user_ids=user_index, item_ids=np.arange(len(item_id_map)))

    # Filter out items the user has already interacted with
    user_interactions = interactions[user_index].toarray().flatten()
    scores[user_interactions > 0] = -np.inf

    # Get top N items
    top_indices = np.argsort(scores)[-N:][::-1]
    recommended_items = [reverse_item_id_map[i] for i in top_indices]

    return {"recommended_items": recommended_items}
