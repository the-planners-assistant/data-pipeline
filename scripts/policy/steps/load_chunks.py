import json
from typing import List, Dict
from pathlib import Path
from scripts.policy.utils.utils import classify_chunk_type  # Import from utils.py


def load_chunks_with_ontology(jsonl_path: str, ontology_path: str) -> List[Dict]:
    chunks = []

    with open(ontology_path, 'r', encoding='utf-8') as f:
        ontology_map = json.load(f)

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunk = json.loads(line)
            chunk_id = chunk.get("id")
            path = ontology_map.get(chunk_id, "Unknown")
            chunk["ontology_path"] = path
            chunk["chunkType"] = classify_chunk_type(path)
            chunks.append(chunk)

    return chunks
