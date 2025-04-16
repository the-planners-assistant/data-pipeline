import re
from typing import Dict


def extract_policy_id(title: str) -> str | None:
    """
    Attempts to extract a policy ID like H1, GG2, D3.1 etc. from a title.
    """
    match = re.match(r"(Policy\s+)?([A-Z]{1,3}\d{1,2}(\.\d{1,2})?)", title.strip(), re.IGNORECASE)
    if match:
        return match.group(2).upper()
    return None


def infer_theme_from_path(path: str) -> str:
    """
    Extracts the high-level theme based on the ontology_path.
    e.g. 'Chapter4_Housing/Policy_H1...' → 'Housing'
    """
    if not path:
        return "Unknown"

    chapter_match = re.match(r"Chapter\d+_([A-Za-z]+)", path)
    if chapter_match:
        return chapter_match.group(1).replace("_", " ").title()
    elif path.startswith("Appendices/Annex3_Glossary"):
        return "Glossary"
    elif "GreenInfrastructure" in path:
        return "Green Infrastructure"
    return "Other"


def infer_supporting_of(chunk: Dict) -> str | None:
    """
    For supporting text chunks, try to infer the related policy ID from the ontology_path.
    e.g., "Chapter1_GoodGrowth/SupportingText_GG1" → "GG1"
    """
    path = chunk.get("ontology_path", "")
    if "SupportingText_" in path:
        match = re.search(r"SupportingText_([A-Z]{1,3}\d{1,2}(\.\d{1,2})?)", path)
        if match:
            return match.group(1).upper()
    return None


def enrich_chunk_metadata(chunk: Dict) -> Dict:
    """
    Adds simple, static metadata to the chunk in-place.
    """
    title = chunk.get("title", "")
    path = chunk.get("ontology_path", "")

    chunk["policy_id"] = extract_policy_id(title) if chunk["chunkType"] == "PolicyStatement" else None
    chunk["themes"] = [infer_theme_from_path(path)]
    chunk["supporting_of"] = infer_supporting_of(chunk) if chunk["chunkType"] == "SupportingText" else None
    chunk["docId"] = "the_london_plan_2021"  # or load from context

    return chunk
