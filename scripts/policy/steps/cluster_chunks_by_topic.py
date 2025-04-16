import json
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import hdbscan

# --- Config ---
INPUT_JSONL = os.path.expanduser("~/data-pipeline/corpus/tower_hamlets/f985ddd124245046f2e5c67c656bba2e_TH_Local_Plan_2024_UPDATE_(ONLINE-HIGH_RES)_flat.jsonl")
OUTPUT_DIR = "clustered_chunks_hdbscan"
MODEL_NAME = "all-MiniLM-L6-v2"
MIN_CLUSTER_SIZE = 10  # Bigger number = broader clusters

# --- Setup ---
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]
print(f"Loaded {len(chunks)} chunks.")

# --- Embeddings ---
print(f"Embedding with {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
texts = [c["text"] for c in chunks]
embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)

# --- HDBSCAN Clustering ---
print("Clustering with HDBSCAN...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, metric='euclidean')
labels = clusterer.fit_predict(embeddings)

print(f"Found {len(set(labels)) - (1 if -1 in labels else 0)} clusters. Noise: {np.sum(labels == -1)} chunks.")

# --- Group chunks by cluster label ---
from collections import defaultdict
cluster_map = defaultdict(list)
for chunk, label in zip(chunks, labels):
    if label != -1:  # skip noise
        cluster_map[label].append(chunk)

# --- Save to disk ---
for label, group in cluster_map.items():
    with open(f"{OUTPUT_DIR}/cluster_{label:02d}.json", "w", encoding="utf-8") as f:
        json.dump(group, f, indent=2, ensure_ascii=False)

print(f"Saved {len(cluster_map)} cluster files to {OUTPUT_DIR}")
