import pandas as pd
import numpy as np
import pickle
from fastapi import FastAPI, HTTPException
from lightfm import LightFM
import traceback

app = FastAPI()

# Load model and encoders
try:
    with open("lightfm_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("user_encoder.pkl", "rb") as f:
        user_encoder = pickle.load(f)

    with open("item_encoder.pkl", "rb") as f:
        item_encoder = pickle.load(f)
except Exception as e:
    print("Error loading model or encoders:", e)
    traceback.print_exc()
    raise e

# Load product metadata
try:
    products_df = pd.read_csv("products.csv")
    products_df = products_df[["product_id", "product_name", "aisle", "department"]]
    products_df.set_index("product_id", inplace=True)
except Exception as e:
    print("Error loading products.csv:", e)
    traceback.print_exc()
    raise e

# Get total items
n_items = len(item_encoder.classes_)

# Health check route
@app.get("/")
def root():
    return {"message": "API is running"}

# Recommendation route
@app.get("/recommend")
def recommend(user_id: int, k: int = 5):
    try:
        # Check if user exists
        user_index = user_encoder.transform([user_id])[0]

        # Predict scores for all items
        scores = model.predict(user_ids=user_index, item_ids=np.arange(n_items))

        # Top K recommendations
        top_items = np.argsort(-scores)[:k]

        recommended = []
        for i in top_items:
            product_id = item_encoder.inverse_transform([i])[0]
            if product_id in products_df.index:
                product_info = products_df.loc[product_id].to_dict()
                product_info["product_id"] = product_id
                recommended.append(product_info)

        return {"user_id": user_id, "recommendations": recommended}

    except ValueError:
        raise HTTPException(status_code=404, detail="User ID not found")
    except Exception as e:
        print("Error during recommendation:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error: " + str(e))
