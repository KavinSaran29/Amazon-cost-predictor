# ============================================================
# SMART PRODUCT PRICING CHALLENGE - STEP 4
# Model Training & Prediction Export
# ============================================================

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from lightgbm import LGBMRegressor

# ============================================================
# PATH SETUP
# ============================================================
REPO_ROOT = Path(__file__).resolve().parent
FEATURE_DIR = REPO_ROOT / "features"
DATA_DIR = REPO_ROOT / "student_resource" / "dataset"

TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV  = DATA_DIR / "test.csv"

# Feature paths
TFIDF_TRAIN = FEATURE_DIR / "tfidf_train.npy"
TFIDF_TEST  = FEATURE_DIR / "tfidf_test.npy"
BERT_TRAIN  = FEATURE_DIR / "bert_train_embeddings.npy"
BERT_TEST   = FEATURE_DIR / "bert_test_embeddings.npy"
IMG_TRAIN   = FEATURE_DIR / "image_train_embeddings.npy"
IMG_TEST    = FEATURE_DIR / "image_test_embeddings.npy"

# ============================================================
# LOAD DATA
# ============================================================
print("📂 Loading features & data...")

train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

y = train_df["price"].values  # target variable

# Ensure feature directory exists
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

def ensure_tfidf(feature_dir, data_dir):
    train_path = feature_dir / "tfidf_train.npy"
    test_path = feature_dir / "tfidf_test.npy"
    if train_path.exists() and test_path.exists():
        return np.load(train_path), np.load(test_path)

    # Try processed CSVs first
    processed_train = data_dir / "processed_train_text.csv"
    processed_test = data_dir / "processed_test_text.csv"
    from sklearn.feature_extraction.text import TfidfVectorizer

    if processed_train.exists() and processed_test.exists():
        print(f"Regenerating TF-IDF from {processed_train} and {processed_test}...")
        df_tr = pd.read_csv(processed_train)
        df_te = pd.read_csv(processed_test)
        tr_texts = df_tr.get("clean_text") if "clean_text" in df_tr.columns else df_tr.get("catalog_content", pd.Series([""] * len(df_tr)))
        te_texts = df_te.get("clean_text") if "clean_text" in df_te.columns else df_te.get("catalog_content", pd.Series([""] * len(df_te)))
        vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        Xtr = vec.fit_transform(tr_texts.fillna(""))
        Xte = vec.transform(te_texts.fillna(""))
        np.save(train_path, Xtr.toarray())
        np.save(test_path, Xte.toarray())
        return Xtr.toarray(), Xte.toarray()

    # Fallback to raw CSVs
    if TRAIN_CSV.exists() and TEST_CSV.exists():
        print("Regenerating TF-IDF from raw CSVs...")
        df_tr = pd.read_csv(TRAIN_CSV)
        df_te = pd.read_csv(TEST_CSV)
        tr_texts = df_tr.get("catalog_content", pd.Series([""] * len(df_tr)))
        te_texts = df_te.get("catalog_content", pd.Series([""] * len(df_te)))
        vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        Xtr = vec.fit_transform(tr_texts.fillna(""))
        Xte = vec.transform(te_texts.fillna(""))
        np.save(train_path, Xtr.toarray())
        np.save(test_path, Xte.toarray())
        return Xtr.toarray(), Xte.toarray()

    raise FileNotFoundError("Cannot find TF-IDF files or CSVs to regenerate them")


tfidf_train, tfidf_test = ensure_tfidf(FEATURE_DIR, DATA_DIR)

def load_or_zeros(path, n_rows):
    p = Path(path)
    if p.exists():
        return np.load(p)
    return np.zeros((n_rows, 0), dtype=np.float32)

bert_train = load_or_zeros(BERT_TRAIN, tfidf_train.shape[0])
bert_test  = load_or_zeros(BERT_TEST, tfidf_test.shape[0])
img_train  = load_or_zeros(IMG_TRAIN, tfidf_train.shape[0])
img_test   = load_or_zeros(IMG_TEST, tfidf_test.shape[0])

print("✅ Loaded all feature arrays successfully.")
print(f"TF-IDF: {tfidf_train.shape} | BERT: {bert_train.shape} | IMG: {img_train.shape}")

# ============================================================
# FEATURE FUSION
# ============================================================
print("🧩 Combining all feature sets...")
X_train = np.hstack([tfidf_train, bert_train, img_train])
X_test  = np.hstack([tfidf_test, bert_test, img_test])
print(f"✅ Combined feature shapes -> Train: {X_train.shape}, Test: {X_test.shape}")

# Optional scaling (important if feature magnitudes differ a lot)
print("⚙️  Scaling features...")
scaler = StandardScaler(with_mean=False)  # sparse compatible
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================
print("🧪 Splitting data for validation...")
X_tr, X_val, y_tr, y_val = train_test_split(X_train_scaled, y, test_size=0.1, random_state=42)

# ============================================================
# TRAIN LIGHTGBM MODEL
# ============================================================
print("🚀 Training LightGBM model...")

model = LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=100)
    ]
)

# ============================================================
# VALIDATION PERFORMANCE
# ============================================================
y_pred_val = model.predict(X_val)
try:
    # newer sklearn supports squared=False
    rmse = mean_squared_error(y_val, y_pred_val, squared=False)
except TypeError:
    # fallback for older sklearn versions
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

print(f"📊 Validation RMSE: {rmse:.4f}")

# ============================================================
# PREDICT TEST DATA
# ============================================================
print("🧠 Generating predictions on test set...")
test_preds = model.predict(X_test_scaled)

# ============================================================
# EXPORT SUBMISSION
# ============================================================
output = pd.DataFrame({
    "sample_id": test_df["sample_id"],
    "predicted_price": test_preds
})

output_path = REPO_ROOT / "submission.csv"
output.to_csv(output_path, index=False)

print(f"✅ Predictions saved successfully -> {output_path}")
