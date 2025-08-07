import pandas as pd
import numpy as np
import pickle
from fastapi import FastAPI, HTTPException
from lightfm import LightFM
import traceback

app = FastAPI()

# Load model and encoders
with open("lightfm_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("user_encoder.pkl", "rb") as f:
    user_encoder = pickle.load(f)

with open("item_encoder.pkl", "rb") as f:
    item_encoder = pickle.load(f)

# Load product metadata
products_df = pd.read_csv("products.csv")
products_df = products_df[["product_id", "product_name"]]
products_df.set_index("product_id", inplace=True)

# Total number of items
n_items = len(item_encoder.classes_)

# Health check
@app.get("/")
def root():
    return {"message": "API is running"}

# Recommendation endpoint
@app.get("/recommend")
def recommend(user_id: int, k: int = 5):
    try:
        user_index = user_encoder.transform([user_id])[0]
        scores = model.predict(user_ids=user_index, item_ids=np.arange(n_items))
        top_items = np.argsort(-scores)[:k]

        recommendations = []
        for i in top_items:
            product_id = item_encoder.inverse_transform([i])[0]
            if product_id in products_df.index:
                product_name = products_df.loc[product_id]["product_name"]
                recommendations.append({
                    "product_id": product_id,
                    "product_name": product_name
                })

        return {"user_id": user_id, "recommendations": recommendations}

    except ValueError:
        raise HTTPException(status_code=404, detail="User ID not found")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")
