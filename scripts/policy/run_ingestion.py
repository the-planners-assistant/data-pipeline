import os
from collections import Counter
from steps.load_chunks import load_chunks_with_ontology

chunks = load_chunks_with_ontology(
    jsonl_path=os.path.expanduser("~/data-pipeline/corpus/the_london_plan_2021_flat.jsonl"),
    ontology_path=os.path.expanduser("~/data-pipeline/ontologies/the_london_plan_2021.json")
)

counts = Counter(chunk["chunkType"] for chunk in chunks)

print("📊 Chunk Type Summary:")
for label, count in counts.items():
    print(f"{label:20} {count}")

    print("\n🔎 Unmapped Chunks:")
for chunk in chunks:
    if chunk["chunkType"] == "Other":
        print(f"\n🧩 Chunk ID: {chunk['id']}")
        print(f"Title: {chunk.get('title', '')}")
        print(f"Ontology Path: {chunk.get('ontology_path')}")
        print(f"Text Preview: {chunk.get('text', '')[:200]}...")
