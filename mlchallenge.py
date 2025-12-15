# ============================================================
# SMART PRODUCT PRICING CHALLENGE
# Step 1 + Step 2 : Data Loading, Image Download & Text Preprocessing
# ============================================================

import os
import re
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from functools import partial
import multiprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor
# --- NEW IMPORTS FOR STEP 3 ---
import torch
from PIL import Image
# Compatibility shim: some torch builds expose a private API name
# `_register_pytree_node` while libraries (e.g. transformers) expect
# `register_pytree_node`. Ensure the expected name exists to avoid
# AttributeError at import time inside containers with mismatched deps.
try:
    pytree = torch.utils._pytree
    if not hasattr(pytree, "register_pytree_node") and hasattr(pytree, "_register_pytree_node"):
        pytree.register_pytree_node = pytree._register_pytree_node
except Exception:
    # If anything goes wrong, we'll let the normal import raise an error later
    pass

from transformers import AutoTokenizer, AutoModel
import timm
from torchvision import transforms

# Set a consistent batch size for processing
BATCH_SIZE = 32

tqdm.pandas()

# ============================================================
# PATH SETUP (robust for both host and container)
# ============================================================
# Prefer a dataset path relative to this script (works when repository
# is mounted into the container at /app or run locally from repo root).
REPO_ROOT = Path(__file__).resolve().parent
candidate1 = REPO_ROOT / "student_resource" / "dataset"
# If running on host in Windows dev env, candidate2 points to the original C: path
candidate2 = Path("C:/dev/ML CHALLENGE/student_resource/dataset")

if candidate1.exists():
    BASE_DIR = candidate1
    print(f"Using dataset directory: {BASE_DIR}")
elif candidate2.exists():
    BASE_DIR = candidate2
    print(f"Using dataset directory: {BASE_DIR}")
else:
    # Last resort: keep candidate1 as BASE_DIR (may be mounted into container)
    BASE_DIR = candidate1
    print(f"Warning: dataset directory not found locally; will attempt to use {BASE_DIR} (container or mounted path)")

# Optional debug: list mounted files inside container when diagnosing missing data
if os.environ.get("DEBUG_MOUNT") == "1":
    try:
        print("--- DEBUG_MOUNT: listing REPO_ROOT contents ---")
        for p in sorted(REPO_ROOT.iterdir()):
            print(p)
    except Exception as e:
        print("DEBUG_MOUNT: failed to list REPO_ROOT:", e)
    try:
        print("--- DEBUG_MOUNT: listing student_resource directory ---")
        sr = REPO_ROOT / "student_resource"
        for p in sorted(sr.iterdir()):
            print(p)
    except Exception as e:
        print("DEBUG_MOUNT: failed to list student_resource:", e)

TRAIN_PATH = BASE_DIR / "train.csv"
TEST_PATH  = BASE_DIR / "test.csv"
# Fallback sample files included with repo
SAMPLE_TRAIN = BASE_DIR / "sample_test.csv"
SAMPLE_TEST = BASE_DIR / "sample_test_out.csv"
IMAGE_DIR  = REPO_ROOT / "dataset" / "images"

os.makedirs(IMAGE_DIR, exist_ok=True)
def download_image(image_link, savefolder):
    """Download single image using urllib"""
    if not isinstance(image_link, str):
        return
    filename = Path(image_link).name
    image_save_path = os.path.join(savefolder, filename)
    if os.path.exists(image_save_path):
        return
    try:
        resp = requests.get(image_link, stream=True, timeout=10)
        resp.raise_for_status()
        with open(image_save_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)
    except Exception as ex:
        print(f"⚠️ Warning: Could not download {image_link}\n{ex}")
    return

def download_images(image_links, download_folder):
    """Parallel image downloader"""
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    download_image_partial = partial(download_image, savefolder=download_folder)
    # Use a thread pool on Windows to avoid process spawn/import overhead
    max_workers = min(32, (os.cpu_count() or 1) * 4)
    max_workers = min(max_workers, len(image_links) or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(tqdm(pool.map(download_image_partial, image_links),
                  total=len(image_links), desc="📸 Downloading images"))

def clean_text(text):
    """Basic text cleanup"""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)                # remove html
    text = re.sub(r"[^a-z0-9\s\.,\-×x%/]", " ", text) # keep essentials
    text = re.sub(r"\s+", " ", text).strip()
    return text
def extract_quantity(text):
    """Extract numeric pack size information"""
    text = text.lower()
    patterns = [
        r"pack\s*of\s*(\d+)",
        r"(\d+)\s*pcs",
        r"(\d+)\s*x\s*\d*",
        r"(\d+)\s*count",
        r"(\d+)\s*pieces?",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return int(m.group(1))
    return 1  # default if not found

# ------------------------------------------------------------
# Extract simple brand heuristic
# ------------------------------------------------------------
def extract_brand(text):
    tokens = text.split()
    return tokens[0] if tokens else "unknown"

# ============================================================
# STEP 3: FEATURE ENGINEERING HELPER FUNCTIONS (NEW CODE)
# ============================================================

def get_bert_embeddings(texts, model_name, device):
    """Generate BERT embeddings for a list of texts using the CLS token."""
    print(f"Initializing tokenizer and model for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    
    all_embeddings = []
    
    print("Generating BERT embeddings...")
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="🧠 Getting Text Embeddings"):
            batch_texts = texts[i:i+BATCH_SIZE]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            outputs = model(**inputs)
            # Use the [CLS] token's embedding as the sentence representation
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)
            
    return np.vstack(all_embeddings)


def get_image_embeddings(image_links, model, transform, device, image_base_dir):
    """Generate image embeddings for a list of image links."""
    all_embeddings = []

    for link in tqdm(image_links, desc="👁️ Getting Image Embeddings"):
        try:
            if not isinstance(link, str):
                 raise FileNotFoundError("Invalid image link")
            
            filename = Path(link).name
            image_path = image_base_dir / filename
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found at {image_path}")

            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                embedding = model(image_tensor).cpu().numpy().flatten()
            all_embeddings.append(embedding)

        except Exception as e:
            # If an image is missing or broken, append a zero vector of the correct shape
            all_embeddings.append(np.zeros(model.num_features))
            
    return np.vstack(all_embeddings)

# ------------------------------------------------------------
# Add numeric text stats
# ------------------------------------------------------------
# NOTE: Dataframe feature extraction happens inside main() after loading the CSVs

# ------------------------------------------------------------
# TF-IDF feature extraction (baseline text vectors)
# ------------------------------------------------------------
def main():
    # ============================================================
    # STEP 1: LOAD DATA
    # ============================================================
    print("📂 Loading dataset...")
    # Use fallback sample files if primary train/test are missing
    if not TRAIN_PATH.exists():
        print(f"⚠️ Train file not found at {TRAIN_PATH}. Falling back to {SAMPLE_TRAIN}")
        if SAMPLE_TRAIN.exists():
            TRAIN_USE = SAMPLE_TRAIN
        else:
            raise FileNotFoundError(f"Neither {TRAIN_PATH} nor {SAMPLE_TRAIN} were found")
    else:
        TRAIN_USE = TRAIN_PATH

    if not TEST_PATH.exists():
        print(f"⚠️ Test file not found at {TEST_PATH}. Falling back to {SAMPLE_TEST}")
        if SAMPLE_TEST.exists():
            TEST_USE = SAMPLE_TEST
        else:
            raise FileNotFoundError(f"Neither {TEST_PATH} nor {SAMPLE_TEST} were found")
    else:
        TEST_USE = TEST_PATH

    train_df = pd.read_csv(TRAIN_USE)
    test_df  = pd.read_csv(TEST_USE)

    print(f"✅ Train samples: {len(train_df)}")
    print(f"✅ Test samples : {len(test_df)}")
    print("📑 Columns:", train_df.columns.tolist())

    # Ensure expected columns exist
    if "catalog_content" not in train_df.columns:
        print("⚠️ Warning: 'catalog_content' not found in train CSV — creating empty column")
        train_df["catalog_content"] = ""
    if "catalog_content" not in test_df.columns:
        print("⚠️ Warning: 'catalog_content' not found in test CSV — creating empty column")
        test_df["catalog_content"] = ""

    # Uncomment below to download images (try small subset first)
    #download_images(train_df["image_link"].tolist()[:1000], IMAGE_DIR)

    # ============================================================
    # STEP 2: TEXT PREPROCESSING
    # ============================================================
    train_df["clean_text"] = train_df["catalog_content"].progress_apply(clean_text)
    test_df["clean_text"]  = test_df["catalog_content"].progress_apply(clean_text)

    # Extract item pack quantity (IPQ)
    train_df["item_pack_qty"] = train_df["clean_text"].progress_apply(extract_quantity)
    test_df["item_pack_qty"]  = test_df["clean_text"].progress_apply(extract_quantity)

    # Simple brand heuristic
    train_df["brand"] = train_df["clean_text"].progress_apply(extract_brand)
    test_df["brand"]  = test_df["clean_text"].progress_apply(extract_brand)

    # Add numeric text stats
    train_df["word_count"] = train_df["clean_text"].apply(lambda x: len(x.split()))
    test_df["word_count"]  = test_df["clean_text"].apply(lambda x: len(x.split()))
    train_df["char_count"] = train_df["clean_text"].apply(len)
    test_df["char_count"]  = test_df["clean_text"].apply(len)

    # TF-IDF feature extraction (baseline text vectors)
    print("⚙️  Generating TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    tfidf_train = vectorizer.fit_transform(train_df["clean_text"])
    tfidf_test  = vectorizer.transform(test_df["clean_text"])

    print("✅ TF-IDF shapes -> Train:", tfidf_train.shape, " Test:", tfidf_test.shape)

    # SAVE INTERMEDIATE FILES
    train_df.to_csv("processed_train_text.csv", index=False)
    test_df.to_csv("processed_test_text.csv", index=False)

    print("\n✅ Step 1 + 2 complete — cleaned text saved & ready for model training.")

# ============================================================
    # STEP 3: FEATURE ENGINEERING (Embeddings)
    # ============================================================
    # Respect SKIP_STEP3 environment variable for quick runs/tests
    print("\n⚙️  Starting Step 3: Generating or Loading Embeddings...")

    # Set device for PyTorch and create features directory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs("features", exist_ok=True)

    # --- 🧠 Text Feature Engineering (BERT) ---
    BERT_MODEL = 'distilbert-base-uncased'
    bert_train_path = Path("features/bert_train_embeddings.npy")
    bert_test_path = Path("features/bert_test_embeddings.npy")

    if bert_train_path.exists() and bert_test_path.exists():
        print("✅ Found existing BERT embeddings. Loading from file...")
        bert_train_embeddings = np.load(bert_train_path)
        bert_test_embeddings = np.load(bert_test_path)
    else:
        print("⚠️ BERT embeddings not found. Generating them now...")
        bert_train_embeddings = get_bert_embeddings(train_df["clean_text"].tolist(), BERT_MODEL, device)
        bert_test_embeddings = get_bert_embeddings(test_df["clean_text"].tolist(), BERT_MODEL, device)
        np.save(bert_train_path, bert_train_embeddings)
        np.save(bert_test_path, bert_test_embeddings)
        print("✅ Saved new BERT embeddings.")

    print(f"BERT embeddings loaded. Train shape: {bert_train_embeddings.shape}, Test shape: {bert_test_embeddings.shape}")

    # --- 👁️ Image Feature Engineering (EfficientNet) ---
    IMAGE_MODEL_NAME = 'efficientnet_b3'
    image_train_path = Path("features/image_train_embeddings.npy")
    image_test_path = Path("features/image_test_embeddings.npy")
    
    if image_train_path.exists() and image_test_path.exists():
        print("✅ Found existing Image embeddings. Loading from file...")
        image_train_embeddings = np.load(image_train_path)
        image_test_embeddings = np.load(image_test_path)
    else:
        print("⚠️ Image embeddings not found. Generating them now...")
        image_model = timm.create_model(IMAGE_MODEL_NAME, pretrained=True, num_classes=0).to(device)
        image_model.eval()
        data_config = timm.data.resolve_data_config(image_model.default_cfg)
        image_transform = timm.data.create_transform(**data_config, is_training=False)
        
        image_train_embeddings = get_image_embeddings(train_df["image_link"].tolist(), image_model, image_transform, device, Path(IMAGE_DIR))
        image_test_embeddings = get_image_embeddings(test_df["image_link"].tolist(), image_model, image_transform, device, Path(IMAGE_DIR))
        
        np.save(image_train_path, image_train_embeddings)
        np.save(image_test_path, image_test_embeddings)
        print("✅ Saved new Image embeddings.")

    print(f"Image embeddings loaded. Train shape: {image_train_embeddings.shape}, Test shape: {image_test_embeddings.shape}")

    print("\n✅ Step 3 complete — All features are ready!")


if __name__ == "__main__":
    # On Windows, protect the entrypoint for multiprocessing
    try:
        multiprocessing.freeze_support()
    except Exception:
        pass
    main()
