import re
import json
import argparse
from pathlib import Path
import pymupdf4llm
import fitz  # PyMuPDF

def clean_chunk_text(text: str) -> str:
    """
    Clean the extracted markdown text:
      - Remove bold formatting markers.
      - Remove markdown header markers.
      - Remove navigation noise like "to table of contents" and horizontal lines.
      - Remove stray control characters.
      - Collapse soft line breaks and excessive whitespace.
    """
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # Remove bold markers
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)  # Remove header symbols
    text = re.sub(r"(?i)to table of contents", "", text)  # Remove navigation text
    text = re.sub(r"[-]{3,}", "", text)  # Remove horizontal rules
    text = text.replace("\b", "")  # Remove stray backspace characters
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # Collapse soft line breaks to a space
    text = re.sub(r"\s{2,}", " ", text)  # Normalize multiple spaces
    text = re.sub(r"", "", text)  # Remove stray arrow markers
    text = re.sub(r"\n{3,}", "\n\n", text)  # Collapse excessive newlines to max 2
    return text.strip()

def parse_markdown_to_nested(md_text: str, doc_id: str) -> dict:
    """
    Parse the markdown text into a nested JSON structure using only markdown-extracted headings
    (i.e. those marked with '#' symbols). Numbered headings (e.g. "1.1.2 ...") are ignored.
    
    The hierarchy is built based on the number of '#' characters.
    Returns a dictionary:
      { "doc_id": <doc_id>, "sections": [ { "heading": ..., "content": ..., "subsections": [...] }, ... ] }
    """
    lines = md_text.splitlines()
    headings = []

    md_heading_pattern = re.compile(r"^(#{1,6})\s+(.*)")
    
    # Only include headings that are found using markdown syntax,
    # and ignore any where the actual text starts with a digit.
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        md_match = md_heading_pattern.match(line)
        if md_match:
            heading_text = md_match.group(2).strip()
            # Only add if the heading text does not begin with a number.
            if not heading_text[0].isdigit():
                level = len(md_match.group(1))
                headings.append({
                    "line": i,
                    "level": level,
                    "heading": heading_text,
                    "subsections": []
                })
    
    # If no headings found, return entire text as one section.
    if not headings:
        return {"doc_id": doc_id, "sections": [{
            "heading": "",
            "content": clean_chunk_text(md_text),
            "subsections": [],
            "tokens": len(md_text.split())
        }]}

    # Assign content to each heading.
    for idx, head in enumerate(headings):
        start_line = head["line"] + 1  # start content after heading
        end_line = headings[idx+1]["line"] if idx+1 < len(headings) else len(lines)
        content = "\n".join(lines[start_line:end_line]).strip()
        head["content"] = clean_chunk_text(content)
    
    # Nest headings using a simple stack based on header level.
    nested_sections = []
    stack = []
    for head in headings:
        while stack and head["level"] <= stack[-1]["level"]:
            stack.pop()
        if stack:
            stack[-1]["subsections"].append(head)
        else:
            nested_sections.append(head)
        stack.append(head)

    # Remove the 'line' key and prepare final output.
    def simplify_heading(h):
        return {
            "heading": h["heading"],
            "content": h.get("content", ""),
            "tokens": len(h.get("content", "").split()),
            "subsections": [simplify_heading(child) for child in h.get("subsections", [])]
        }
    
    simplified = [simplify_heading(s) for s in nested_sections]
    return {"doc_id": doc_id, "sections": simplified}

def save_nested_chunks_to_json(nested_chunks, output_json: Path):
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(nested_chunks, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved nested text structure to {output_json}")

def extract_images(pdf_path: Path, images_dir: Path) -> list:
    """
    Extract images from the PDF using PyMuPDF and attempt to save as PNG.
    On failure due to colorspace issues, falls back to raw extraction.
    Returns a manifest of images with metadata.
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
                print(f"Warning: {e}. Using fallback extraction for page {page_num}, image {img_index}.")
                image_dict = doc.extract_image(xref)
                ext = image_dict.get("ext", "png")
                filename = f"page_{page_num}_img_{img_index:03}.{ext}"
                out_path = images_dir / filename
                with out_path.open("wb") as img_file:
                    img_file.write(image_dict.get("image"))
            pix = None
            images_manifest.append({
                "page_number": page_num,
                "image_file": filename,
                "xref": xref,
                "image_index": img_index
            })

    doc.close()
    print(f"✅ Extracted {image_count} images from {pdf_path.name}")
    return images_manifest

def save_images_manifest(manifest, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved images manifest with {len(manifest)} entries to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract nested Markdown sections (using only markdown-extracted headings) and images from a PDF."
    )
    parser.add_argument("pdf", type=str, help="Path to the input PDF.")
    parser.add_argument("output_folder", type=str, help="Output folder for JSON and images.")
    
    args = parser.parse_args()
    pdf_path = Path(args.pdf)
    output_folder = Path(args.output_folder)
    
    # Use the PDF filename (without extension) as doc_id.
    doc_id = pdf_path.stem
    output_json = output_folder / f"{doc_id}.json"
    
    # PART 1: Extract Markdown and create nested text structure.
    print(f"Extracting markdown from {pdf_path} ...")
    md_text = pymupdf4llm.to_markdown(str(pdf_path))
    nested_structure = parse_markdown_to_nested(md_text, doc_id)
    save_nested_chunks_to_json(nested_structure, output_json)
    
    # PART 2: Extract images.
    images_dir = output_folder / doc_id
    manifest = extract_images(pdf_path, images_dir)
    manifest_json = images_dir / "images_manifest.json"
    save_images_manifest(manifest, manifest_json)

if __name__ == "__main__":
    main()
