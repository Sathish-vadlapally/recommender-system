# 🛒 Recommender System (LightFM)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green)
![LightFM](https://img.shields.io/badge/LightFM-Recommender-orange)
![Deployment](https://img.shields.io/badge/Deployed-Render-blue)

A **personalized recommendation system** built using **LightFM**, combining collaborative & content-based filtering.  
The model provides product recommendations based on user preferences and historical purchase behavior.  
This project includes a **FastAPI backend** for serving recommendations and is fully deployable on **Render**.

---

## 🚀 Features
- Hybrid Recommendation Engine (**Collaborative + Content-based filtering**)
- Built using **LightFM** for high-quality predictions
- **FastAPI**-powered RESTful API
- Scalable and deployment-ready with **Render**
- Includes pre-trained model and encoders

---

## 📂 Project Structure
```
├── main.py               # FastAPI app serving recommendations
├── requirements.txt      # Python dependencies
├── .render.yaml          # Render deployment configuration
├── runtime.txt           # Python runtime version
├── setup.cfg             # LightFM OpenMP configuration
├── lightfm_model.pkl     # Trained LightFM model
├── user_encoder.pkl      # Label encoder for users
├── item_encoder.pkl      # Label encoder for items
├── products.csv          # Product metadata
└── .python-version       # Python version for deployment
```

---

## 🛠 Tech Stack
- **Language**: Python 3.11
- **Framework**: FastAPI, Uvicorn
- **Recommendation Engine**: LightFM
- **Libraries**: Pandas, NumPy, Scikit-learn
- **Deployment**: Render

---

## 🔌 API Endpoints

### **Root Endpoint**
```
GET /
```
Returns a welcome message.

### **Get Recommendations**
```
GET /recommend/{user_id}?num_items=5
```
Fetches top-N product recommendations for a specific user.

#### Example:
```
/recommend/123?num_items=5
```

#### Response:
```json
{
  "user_id": "123",
  "recommendations": [
    {"product_id": "567", "product_name": "Organic Apples"},
    {"product_id": "892", "product_name": "Whole Wheat Bread"}
  ]
}
```

---

## ⚙️ Installation & Setup

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/recommender-system.git
cd recommender-system
```

### **2. Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Run Locally**
```bash
uvicorn main:app --reload
```
API available at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🌐 Deployment (Render)
1. Push your code to GitHub  
2. Go to [Render](https://render.com/)  
3. Create a **New Web Service**  
4. Use `.render.yaml` and `runtime.txt` for seamless deployment

---

## 🚧 Future Improvements
- Add advanced **user & item embeddings**
- Improve ranking metrics (Precision@k, Recall@k, MAP)
- Support cold-start users/items
- Build an interactive **Streamlit/React frontend**

---

## 📌 Author
**Sathish Vadlapally**  
💻 Passionate about Machine Learning & Deploying Real-world Systems  
📧 Contact: *[sathishvadlapally@gmail.com]*

