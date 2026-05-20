import streamlit as st

def show():
    st.markdown('<p class="hero-title">About FraudShield</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Architecture, Tech Stack & Resume Description</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏗️ Architecture")
        st.markdown("""
**Data Pipeline**
- Synthetic dataset (10,000 transactions, 3% fraud rate)
- Feature engineering: temporal, geographic, behavioural
- SMOTE oversampling to handle class imbalance

**ML Models**
| Model | Config |
|-------|--------|
| Random Forest | 200 trees, max_depth=12, balanced weights |
| XGBoost | 300 estimators, lr=0.05, scale_pos_weight |

**Ensemble Strategy**
- Soft voting (average probabilities)
- Threshold = 0.5 for fraud classification

**Features Used**
- `amount` — transaction value
- `hour`, `day_of_week` — temporal patterns
- `merchant_cat` — encoded category
- `distance_km` — geo anomaly signal
- `prev_txn_hrs` — velocity signal
- `v1`–`v5` — PCA-derived latent features
        """)

    with col2:
        st.subheader("🛠️ Tech Stack")
        st.markdown("""
| Layer | Technology |
|-------|------------|
| UI | Streamlit 1.32 |
| ML | scikit-learn, XGBoost |
| Sampling | imbalanced-learn (SMOTE) |
| Charts | Plotly |
| Serialisation | joblib (pkl) |
| Language | Python 3.10+ |
        """)

        st.subheader("📄 Resume Description")
        st.info("""
**Credit Card Fraud Detection System** · *Python, Streamlit, scikit-learn, XGBoost*

Built an end-to-end real-time fraud detection web application with an ensemble ML pipeline
(Random Forest + XGBoost), achieving ROC-AUC >0.97. Applied SMOTE to address a 97:3
class imbalance, implemented a Streamlit dashboard with single-transaction prediction and
CSV batch scoring, and packaged the solution for one-click deployment on Streamlit Cloud.
        """)

    st.markdown("---")
    st.subheader("📁 Project Structure")
    st.code("""
fraud_detection/
├── app.py                  # Streamlit entry point
├── train_model.py          # ML training pipeline
├── generate_dataset.py     # Synthetic data generator
├── requirements.txt        # Python dependencies
├── data/
│   └── creditcard.csv      # Dataset (generated)
├── models/
│   ├── random_forest.pkl   # Trained RF model
│   ├── xgboost.pkl         # Trained XGB model
│   ├── scaler.pkl          # StandardScaler
│   ├── label_encoder.pkl   # LabelEncoder
│   └── metrics.json        # Evaluation metrics
└── pages/
    ├── home.py             # Dashboard home
    ├── predict.py          # Single prediction
    ├── analytics.py        # Model analytics
    ├── batch.py            # Batch CSV scorer
    └── about.py            # This page
    """, language="")

    st.markdown("---")
    st.subheader("🚀 Deployment Guide (Streamlit Cloud)")
    st.markdown("""
1. Push this folder to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `main`, and file `app.py`
4. **Important**: Add a `packages.txt` with `libgomp1` for XGBoost on Linux
5. Click **Deploy** — done!

> **Tip**: Add a `setup.sh` to auto-run `generate_dataset.py` + `train_model.py`
> on first boot if no models exist, or commit your pre-trained `.pkl` files.
    """)
