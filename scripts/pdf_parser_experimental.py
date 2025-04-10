import os
import re
import json
import uuid
import concurrent.futures
import argparse
from pathlib import Path

import pymupdf4llm  # For converting PDFs to Markdown
import fitz         # PyMuPDF for image extraction
from langchain.text_splitter import RecursiveCharacterTextSplitter

#############################
# TEXT CLEANING FUNCTION    #
#############################

def clean_chunk_text(text: str) -> str:
    """
    Clean the extracted markdown text:
      - Remove bold formatting markers.
      - Remove markdown header symbols.
      - Remove noise like "to table of contents" and horizontal lines.
      - Remove stray control characters.
      - Collapse soft line breaks and excessive whitespace.
    """
    # Remove bold markers e.g., **text** → text
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    # Remove markdown header markers (e.g., '#' at line starts)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    # Remove unwanted navigation text like "to table of contents"
    text = re.sub(r"(?i)to table of contents", "", text)
    # Remove horizontal rules (repeated dashes)
    text = re.sub(r"[-]{3,}", "", text)
    # Remove stray backspace characters
    text = text.replace("\b", "")
    # Collapse single newlines (that are not paragraph breaks) to spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Normalize multiple spaces to a single space
    text = re.sub(r"\s{2,}", " ", text)
    # Remove stray arrow markers (e.g., )
    text = re.sub(r"", "", text)
    # Collapse multiple newlines to at most two
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

#############################################
# NESTED PARSING OF MARKDOWN (Up to 6 levels) #
#############################################

def parse_markdown_to_nested(md_text: str, doc_id: str) -> dict:
    """
    Parse the markdown text into a nested JSON structure using only markdown heading markers.
    This parser examines lines starting with '#' (up to six layers) and ignores lines that begin with a digit,
    ensuring that numbered headings are skipped. Each heading level is determined by the number of '#' characters.
    Returns a dictionary of the form:
      {
        "doc_id": "<doc_id>",
        "sections": [
          {
            "heading": "Section Title",
            "content": "The content under this heading",
            "tokens": <word_count>,
            "subsections": [ { ... nested sections ... } ]
          },
          ...
        ]
      }
    """
    lines = md_text.splitlines()
    headings = []
    md_heading_pattern = re.compile(r"^(#{1,6})\s+(.*)")

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        md_match = md_heading_pattern.match(line)
        if md_match:
            heading_text = md_match.group(2).strip()
            # Skip numbered headings if desired
            if not heading_text[0].isdigit():
                level = len(md_match.group(1))
                headings.append({
                    "line": i,
                    "level": level,
                    "heading": heading_text,
                    "subsections": []
                })

    # If no headings were detected, return the entire text as one section.
    if not headings:
        return {"doc_id": doc_id, "sections": [{
            "heading": "",
            "content": clean_chunk_text(md_text),
            "tokens": len(md_text.split()),
            "subsections": []
        }]}

    # Assign content to each heading based on the lines between headings.
    for idx, head in enumerate(headings):
        start_line = head["line"] + 1  # content starts after the heading line
        end_line = headings[idx + 1]["line"] if idx + 1 < len(headings) else len(lines)
        content = "\n".join(lines[start_line:end_line]).strip()
        head["content"] = clean_chunk_text(content)
        head["tokens"] = len(content.split())

    # Nest headings using a simple stack: every heading becomes a child of the previous heading of lower level.
    nested = []
    stack = []
    for head in headings:
        while stack and head["level"] <= stack[-1]["level"]:
            stack.pop()
        if stack:
            stack[-1]["subsections"].append(head)
        else:
            nested.append(head)
        stack.append(head)

    def simplify_heading(h):
        return {
            "heading": h["heading"],
            "content": h.get("content", ""),
            "tokens": h.get("tokens", 0),
            "subsections": [simplify_heading(child) for child in h.get("subsections", [])]
        }

    simplified_nested = [simplify_heading(section) for section in nested]
    return {"doc_id": doc_id, "sections": simplified_nested}

##########################################
# FLATTENING NESTED STRUCTURE FOR Qdrant #
##########################################

def flatten_nested_tree(nested_structure: dict, parent_path: list = None) -> list:
    """
    Recursively flatten the nested JSON structure into a flat list of policy records.
    Each record represents an individual policy (or section) with metadata including:
      - "id": unique identifier,
      - "title": the heading,
      - "content": text of the section,
      - "level": depth in the hierarchy,
      - "path": list of headings from the root to this node,
      - "tokens": word count,
      - "source": document identifier.
    """
    if parent_path is None:
        parent_path = []
    flat_records = []
    for section in nested_structure.get("sections", []):
        current_path = parent_path + [section["heading"]]
        record = {
            "id": str(uuid.uuid4()),
            "title": section["heading"],
            "content": section.get("content", ""),
            "level": len(current_path),
            "path": current_path,
            "tokens": section.get("tokens", 0),
            "source": nested_structure.get("doc_id", "")
        }
        flat_records.append(record)
        # Recurse into any subsections.
        if section.get("subsections"):
            flat_records.extend(flatten_nested_tree({"sections": section["subsections"], "doc_id": nested_structure.get("doc_id", "")}, current_path))
    return flat_records

###############################################
# FIXED SIZE CHUNKING (using LangChain Splitter) #
###############################################

def chunk_markdown_fixed(text: str, doc_id: str) -> list:
    """
    Fixed-size chunking using RecursiveCharacterTextSplitter from langchain.
    Splits the full document text into overlapping chunks (default: 1000 characters with 200 overlap).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    raw_chunks = splitter.split_text(text)
    chunks = []
    for idx, chunk in enumerate(raw_chunks):
        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "heading": "",  # Fixed splitting does not preserve heading info.
            "text": clean_chunk_text(chunk),
            "tokens": len(chunk.split()),
            "source": doc_id
        })
    return chunks

def chunk_hybrid(md_text: str, doc_id: str, strategy: str = "hybrid") -> list:
    """
    Hybrid chunking method:
      - "semantic": use the markdown header-based parser to create a flat per-policy list.
      - "fixed": use fixed-size chunking via the RecursiveCharacterTextSplitter.
      - "hybrid": if the markdown contains at least three "## " headers, use semantic; otherwise use fixed.
    Returns a flat list (each entry representing an individual policy/section).
    """
    if strategy == "semantic":
        # Use our semantic parser to get nested structure, then flatten.
        nested = parse_markdown_to_nested(md_text, doc_id)
        return flatten_nested_tree(nested)
    elif strategy == "fixed":
        return chunk_markdown_fixed(md_text, doc_id)
    elif strategy == "hybrid":
        header_count = md_text.count("## ")
        if header_count >= 3:
            print(f"[INFO] {doc_id}: Detected {header_count} headers → Using semantic chunking")
            nested = parse_markdown_to_nested(md_text, doc_id)
            return flatten_nested_tree(nested)
        else:
            print(f"[INFO] {doc_id}: Detected only {header_count} headers → Using fixed-size chunking")
            return chunk_markdown_fixed(md_text, doc_id)
    else:
        print(f"[WARN] Unknown strategy '{strategy}', defaulting to hybrid")
        return chunk_hybrid(md_text, doc_id, strategy="hybrid")

#####################
# IMAGE EXTRACTION  #
#####################

def extract_images(pdf_path: Path, images_dir: Path) -> list:
    """
    Extract images from the PDF using PyMuPDF.
    Saves each image as a PNG (or its native extension if PNG fails) and returns a manifest
    of extracted image metadata.
    """
    doc = fitz.open(str(pdf_path))
    images_manifest = []
    images_dir.mkdir(parents=True, exist_ok=True)
    image_count = 0
    for page_num, page in enumerate(doc, start=1):
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list, start=1):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n > 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            image_count += 1
            filename = f"page_{page_num}_img_{img_index:03}.png"
            out_path = images_dir / filename
            try:
                pix.save(str(out_path))
            except Exception as e:
                print(f"[WARN] {e}. Falling back for page {page_num}, image {img_index}.")
                image_dict = doc.extract_image(xref)
                ext = image_dict.get("ext", "png")
                filename = f"page_{page_num}_img_{img_index:03}.{ext}"
                out_path = images_dir / filename
                with out_path.open("wb") as img_file:
                    img_file.write(image_dict.get("image"))
            pix = None  # free memory
            images_manifest.append({
                "page_number": page_num,
                "image_file": filename,
                "xref": xref,
                "image_index": img_index
            })
    doc.close()
    print(f"[INFO] Extracted {image_count} images from {pdf_path.name}")
    return images_manifest

def save_images_manifest(manifest, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved images manifest with {len(manifest)} entries to {output_path}")

##########################################
# MAIN PIPELINE FUNCTIONS AND EXECUTION  #
##########################################

def process_pdf(pdf_path: str, strategy: str = "hybrid") -> tuple:
    """
    Process a single PDF:
      1. Extract Markdown using pymupdf4llm.
      2. Generate two representations:
         a. A flat, per-policy list (suitable for Qdrant).
         b. A nested structure preserving hierarchy (suitable for Neo4j).
    Returns a tuple (flat_chunks, nested_structure).
    """
    print(f"[INFO] Extracting markdown from {pdf_path} ...")
    md_text = pymupdf4llm.to_markdown(str(pdf_path))
    if not md_text:
        print(f"[WARN] No markdown text extracted for {pdf_path}")
        return ([], {})
    # For flat output, use the hybrid strategy which flattens nested structure.
    flat_chunks = chunk_hybrid(md_text, Path(pdf_path).stem, strategy=strategy)
    # For nested output, use the full nested parser.
    nested_structure = parse_markdown_to_nested(md_text, Path(pdf_path).stem)
    return (flat_chunks, nested_structure)

def process_and_write(pdf_path: str, output_folder: str, strategy: str) -> None:
    """
    Process a single PDF and produce:
      - A flat JSONL file for Qdrant, named: <doc_id>_flat.jsonl.
      - A nested JSON file for Neo4j, named: <doc_id>_nested.json.
      - An images manifest and extracted images saved in a subdirectory <doc_id>.
    """
    doc_id = Path(pdf_path).stem
    flat_chunks, nested_structure = process_pdf(pdf_path, strategy=strategy)
    if not flat_chunks:
        print(f"[WARN] No text chunks produced for {pdf_path}")
        return
    # Save flat JSONL output
    flat_output_file = Path(output_folder) / f"{doc_id}_flat.jsonl"
    with flat_output_file.open("w", encoding="utf-8") as f:
        for record in flat_chunks:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[INFO] Flat JSONL saved to {flat_output_file}")
    # Save nested JSON output
    nested_output_file = Path(output_folder) / f"{doc_id}_nested.json"
    with nested_output_file.open("w", encoding="utf-8") as f:
        json.dump(nested_structure, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Nested JSON saved to {nested_output_file}")
    # Process images
    images_dir = Path(output_folder) / doc_id
    manifest = extract_images(Path(pdf_path), images_dir)
    manifest_json = images_dir / "images_manifest.json"
    save_images_manifest(manifest, manifest_json)

def main_parallel(input_folder: str, output_folder: str, strategy: str, max_workers: int = 2) -> None:
    pdf_paths = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_paths.append(os.path.join(root, file))
    print(f"[INFO] Found {len(pdf_paths)} PDF(s) in '{input_folder}'")
    os.makedirs(output_folder, exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_and_write, pdf, output_folder, strategy): pdf for pdf in pdf_paths}
        for future in concurrent.futures.as_completed(futures):
            pdf = futures[future]
            try:
                future.result()
                print(f"[DONE] Processed {pdf}")
            except Exception as exc:
                print(f"[ERROR] Processing {pdf} raised an exception: {exc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Hybrid PDF Extractor for Local Plans: "
                     "Generates flat JSONL (for Qdrant indexing) and nested JSON (for Neo4j knowledge graphs) outputs, "
                     "along with image extraction.")
    )
    parser.add_argument("input_folder", help="Folder containing PDF files.")
    parser.add_argument("output_folder", help="Folder to save output JSON/JSONL files and images.")
    parser.add_argument("--strategy", choices=["semantic", "fixed", "hybrid"], default="hybrid",
                        help="Chunking strategy to use (default: hybrid)")
    parser.add_argument("--workers", type=int, default=2, help="Number of parallel processes (default: 2)")
    args = parser.parse_args()
    
    main_parallel(args.input_folder, args.output_folder, args.strategy, args.workers)
