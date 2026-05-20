# 🛡️ FraudShield — Credit Card Fraud Detection

A production-grade, end-to-end fraud detection web app built with **Streamlit**, **scikit-learn**, and **XGBoost**.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset
python generate_dataset.py

# 3. Train models
python train_model.py

# 4. Launch dashboard
streamlit run app.py
```

## 📋 Features
- ✅ Real-time single-transaction fraud prediction
- ✅ Ensemble model (Random Forest + XGBoost)
- ✅ SMOTE for class imbalance
- ✅ Batch CSV scoring with downloadable results
- ✅ Model analytics: confusion matrix, ROC-AUC, feature importance
- ✅ Interactive Plotly charts
- ✅ Streamlit Cloud deployment ready

## 📁 Pages
| Page | Description |
|------|-------------|
| Home | KPI metrics, dataset overview |
| Predict | Single transaction analyser |
| Analytics | Model performance & distributions |
| Batch Upload | Score thousands of transactions at once |
| About | Architecture, resume description, deployment guide |

## 🌐 Deploy to Streamlit Cloud
1. Push to GitHub
2. Go to share.streamlit.io → New app
3. Select repo, branch `main`, file `app.py`
4. Deploy!
