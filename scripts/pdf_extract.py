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
      - Remove noise like "to table of contents" and horizontal lines.
      - Remove stray control characters.
      - Collapse soft line breaks and excessive whitespace.
    """
    # Remove bold markers: **text** → text
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    # Remove markdown header markers (e.g. '#' characters at the beginning of lines)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    # Remove unwanted text like "to table of contents" (case-insensitive)
    text = re.sub(r"(?i)to table of contents", "", text)
    # Remove horizontal rules (sequences of dashes)
    text = re.sub(r"[-]{3,}", "", text)
    # Remove stray '\b' characters
    text = text.replace("\b", "")
    # Replace soft line breaks (single newline not part of a paragraph break) with a space
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Normalize multiple spaces into one
    text = re.sub(r"\s{2,}", " ", text)
    # Remove any stray arrow or marker artifacts (e.g. )
    text = re.sub(r"", "", text)
    # Collapse multiple newlines into a maximum of two
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_markdown(md_text: str, doc_id: str) -> list:
    """
    Splits the markdown text into chunks based on level-2 headings (## ).
    Cleans up layout noise like soft line breaks, excessive spacing, and PDF artifacts.
    """
    sections = re.split(r"\n(?=## )", md_text.strip())
    chunks = []

    for i, section in enumerate(sections):
        lines = section.strip().split("\n", 1)
        heading = lines[0].replace("##", "").strip() if lines else f"Section {i+1}"
        body = lines[1].strip() if len(lines) > 1 else ""

        # --- Layout cleanup ---
        body = re.sub(r"\n{2,}", "\n\n", body)  # keep paragraph breaks
        body = re.sub(r"(?<!\n)\n(?!\n)", " ", body)  # collapse soft line breaks
        body = re.sub(r"\s{2,}", " ", body)  # remove extra spaces
        body = body.replace("\b", "")  # remove leftover \b artifacts
        body = re.sub(r"^•\s*", "- ", body, flags=re.MULTILINE)  # normalize bullets

        # Apply further cleaning from our helper function.
        body = clean_chunk_text(body)

        chunk = {
            "chunk_id": f"{doc_id}_chunk_{i+1:03}",
            "heading": heading,
            "text": body,
            "source": doc_id,
            "tokens": len(body.split())
        }
        chunks.append(chunk)

    return chunks

def save_chunks_to_json(chunks, output_json: Path):
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {len(chunks)} text chunks to {output_json}")

def extract_images(pdf_path: Path, images_dir: Path) -> list:
    """
    Extracts all images from the PDF using PyMuPDF.
    Attempts to save each image as PNG. If saving fails due to unsupported colorspace,
    falls back to using doc.extract_image(xref) and saves the raw image using its native extension.
    Returns a manifest of extracted images with metadata.
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

            # If the image uses more than 4 components (e.g., CMYK), convert it to RGB.
            if pix.n > 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)

            image_count += 1
            filename = f"page_{page_num}_img_{img_index:03}.png"
            out_path = images_dir / filename
            try:
                pix.save(str(out_path))
            except Exception as e:
                print(f"Warning: {e}. Falling back to doc.extract_image for page {page_num}, image {img_index}.")
                # Fallback: Extract raw image data and determine its native extension
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
    print(f"✅ Extracted {image_count} images from {pdf_path.name}")
    return images_manifest

def save_images_manifest(manifest, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved images manifest with {len(manifest)} entries to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract Markdown chunks and images from a PDF."
    )
    parser.add_argument("pdf", type=str, help="Path to the input PDF.")
    parser.add_argument("output_folder", type=str, help="Output folder for JSON and images.")
    
    args = parser.parse_args()
    pdf_path = Path(args.pdf)
    output_folder = Path(args.output_folder)
    
    # Use the PDF filename (without extension) as the doc_id.
    doc_id = pdf_path.stem
    # Output JSON is saved as <doc_id>.json in the output folder.
    output_json = output_folder / f"{doc_id}.json"
    
    # PART 1: Extract Markdown and create text chunks.
    print(f"Extracting markdown from {pdf_path} ...")
    md_text = pymupdf4llm.to_markdown(str(pdf_path))
    chunks = chunk_markdown(md_text, doc_id=doc_id)
    save_chunks_to_json(chunks, output_json)
    
    # PART 2: Extract images.
    # Create an images directory inside the output folder with the same name as the document.
    images_dir = output_folder / doc_id
    manifest = extract_images(pdf_path, images_dir)
    manifest_json = images_dir / "images_manifest.json"
    save_images_manifest(manifest, manifest_json)

if __name__ == "__main__":
    main()
