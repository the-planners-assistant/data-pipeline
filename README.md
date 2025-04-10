# data-pipeline

- NPPF: December 2024 - no useful images
- London Plan: 2021 - lots of vector graphics, should be replicated in London datasets
- City of London:

## London Boroughs

### Inner London
- Camden: 2017, v1 draft at consultation stage as of 04/25
- Greenwich
- Hackney
- Hammersmith and Fulham
- Islington
- Kensington and Chelsea
- Lambeth
- Lewisham
- Southwark: 2022, complete
- Tower Hamlets: 2020, draft examination in progress as of 04/25
- Wandsworth
- Westminster

### Outer London
- Barking and Dagenham
- Barnet
- Bexley
- Brent
- Bromley
- Croydon
- Ealing
- Enfield
- Haringey: 2017, new plan on horizon
- Harrow
- Havering: 2021, complete
- Hillingdon
- Hounslow
- Kingston upon Thames
- Merton
- Newham
- Redbridge
- Richmond upon Thames: 2018, draft examination in progress as of 04/25
- Sutton
- Waltham Forest


# pdf_extract Documentation

## Overview

The **pdf_extract** script is a command-line tool designed to process planning documents (or any PDFs) by converting them into two primary outputs:

1. **Structured Text Chunks (JSON):**  
   - Converts the PDF to Markdown using `pymupdf4llm.to_markdown()`.
   - Splits the Markdown into chunks based on level‑2 headings (lines that begin with "`##`").
   - Cleans the extracted text by removing excessive Markdown formatting, artifacts (such as stray control characters or repeated navigation markers), and extraneous whitespace.
   - Saves the resulting JSON file with the same name as the PDF (e.g., `input.pdf` produces `input.json`).

2. **Extracted Images:**  
   - Uses PyMuPDF to extract all embedded images from the PDF.
   - Converts images to PNG format (including conversion from CMYK to RGB when required).
   - Saves each image in a new subdirectory (named after the PDF without the extension) within the specified output folder.
   - Creates an `images_manifest.json` file in that subdirectory containing metadata (page number, image filename, xref, and image index) for every extracted image.

---

## Requirements

- **Python 3.6 or higher**
- **Dependencies:**
  - `pymupdf4llm`
  - `pymupdf`

It is recommended that you use a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows
pip install pymupdf4llm pymupdf
```

---

## Script Usage

The script takes two required arguments from the command line:  
1. The path to the input PDF file.  
2. The output folder where the JSON and image files will be saved.

### Command-Line Example
```bash
python pdf_extract.py path/to/input.pdf path/to/output_folder
```

### What Happens:
- The script uses the **PDF’s filename** (without extension) as the document identifier (`doc_id`).
- **Text Extraction:**
  - The PDF is converted into Markdown via `pymupdf4llm.to_markdown()`.
  - The Markdown is split into chunks based on level‑2 headings.
  - Each chunk is cleaned using a set of regular expressions that strip bold markers, header symbols, and navigation artifacts (like “to table of contents” lines).
  - The cleaned chunks are output as a JSON file named `<doc_id>.json` in the specified output folder.
  
- **Image Extraction:**
  - All images embedded in the PDF are extracted using PyMuPDF.
  - Each image is saved as a PNG file.
  - Images are stored in a subdirectory named after the PDF (without extension) within the output folder.
  - An `images_manifest.json` file is generated in the images subdirectory to record details for each image.

---

## Output Structure

Assume the input PDF is named **`input.pdf`** and the output folder is **`output_folder`**. The resulting structure will be as follows:

```
output_folder/
├── input.json               # JSON file with structured text chunks
└── input/                   # Subdirectory for images (named after the PDF)
    ├── page_1_img_001.png
    ├── page_1_img_002.png
    ├── ...                # Additional extracted images
    └── images_manifest.json  # JSON manifest for images metadata
```

---

## Script Details

### 1. Text Chunking and Cleaning
- **Function `chunk_markdown(md_text: str, doc_id: str) -> list`:**  
  Splits the Markdown text into chunks based on level‑2 headings. For each chunk, it removes artifacts such as soft line breaks, excessive spaces, bold markers (`**`), header markers, stray control characters (like `\b`), and irrelevant text (e.g., navigation links like "to table of contents").  
  The chunk is stored with:
  - `chunk_id`: Unique identifier based on the document ID.
  - `heading`: Cleaned heading text.
  - `text`: The cleaned and formatted body text.
  - `source`: The document ID.
  - `tokens`: A crude count of words in the body.

- **Function `clean_chunk_text(text: str) -> str`:**  
  This helper function applies several regular expressions to:
  - Remove **bold markers**.
  - Strip out markdown header symbols.
  - Remove unnecessary navigation artifacts (e.g., "to table of contents").
  - Collapse extra whitespace and newlines.

### 2. Image Extraction
- **Function `extract_images(pdf_path: Path, images_dir: Path) -> list`:**  
  Opens the PDF with PyMuPDF, iterates through each page, and extracts every embedded image. It converts images (if needed) from CMYK to RGB and saves them as PNG files.
- **Manifest Generation:**  
  Collects metadata such as page number, image filename, xref, and image index for each image into an `images_manifest.json` file.

### 3. Saving Output Files
- **Functions `save_chunks_to_json(chunks, output_json: Path)` and `save_images_manifest(manifest, output_path: Path)`:**  
  Ensure that the parent directories exist, then save the JSON data with proper indentation.

---

## Additional Considerations

- **Cleaning Further:**  
  Depending on the quality of the PDF, you may need to add or adjust cleaning rules in the `clean_chunk_text` function to handle any additional layout artifacts.
  
- **Scalability:**  
  This script works well for medium-sized PDFs. For giant documents (e.g., the London Plan), consider processing in batches (page-by-page processing) or running the script on a cloud platform with more memory.


  Below is a more **creative, AI-first approach** to auto-generating your prompt templating DSL, harnessing the **LLM’s emergent domain understanding** rather than purely rule-based or code-based transformations. The concept is to let the LLM itself *architect* and *instantiate* the DSL based on **high-level domain specifications**, examples, and constraints—so you can flexibly scale to new reasoning tasks without manually writing out every prompt template.


# Meta-prompting DSL

## 1. Big Picture Overview

1. **High-Level Domain Specs**  
   You maintain a **lightweight, domain-level specification** (in YAML or JSON) describing each planning or policy task:  
   - Input data needed (e.g., site constraints, viability data, policy references)  
   - Desired outputs (e.g., classification, narrative explanation, numeric result)  
   - Domain constraints (e.g., the policy must refer to NPPF paragraphs, must handle local objections, etc.)  

2. **AI-Driven “Meta-Prompt”**  
   You feed these specs into a *special LLM prompt* (the “meta-prompt”) that instructs the LLM on how to:  
   - Interpret the domain specs  
   - Generate a DSL or “prompt blueprint” that captures all placeholders, instructions, format details, etc.  
   - Provide an example prompt instance for demonstration  

3. **LLM-Generated DSL Artifacts**  
   The LLM *emergently designs* the DSL for a new reasoning chain. It outputs:  
   - A **YAML or JSON file** describing the prompt structure (variables, instructions, placeholders).  
   - Optionally, **example input-output pairs** to show usage.  

4. **Self-Review or Iterative Improvement**  
   You can then feed the DSL back into the LLM or a second pass to **validate**, refine, or unify it with existing templates.  

The idea is: rather than you or your dev team writing DSL templates from scratch, you let the LLM’s domain understanding create them—using your minimal domain specs plus examples, constraints, and an overarching meta-prompt.

---

## 2. Example Workflow

### Step A: Provide High-Level Domain Spec

Imagine you have a new “S106 Negotiation Reasoning” module. You define minimal domain parameters, something like:

```yaml
task_name: s106_negotiation
description: A chain for suggesting S106 contributions based on local viability, policy, and precedent.
input_schema:
  site_context:
    type: string
    description: "Key site constraints, location, existing consents, viability context"
  policy_references:
    type: list
    items:
      type: string
    description: "Relevant policies or SPD references impacting S106"
  relevant_precedents:
    type: list
    items:
      type: string
    description: "Similar S106 cases or appeal decisions to cross-check"
  financials:
    type: object
    properties:
      build_cost: number
      land_value: number
      ...
desired_outputs:
  - s106_recommendation: string
  - justification_text: string
domain_constraints:
  - "Always check if the policy references align with NPPF requirements"
  - "Cross-verify financial viability with typical local developer margins"
  - "Use standard policy thresholds for S106 contributions if found"
```

### Step B: Feed into a Meta-Prompt

You might store a “meta-prompt” that instructs the LLM:

> "You are the **DSL generator**. You receive a domain task specification describing the inputs, desired outputs, and domain constraints for a new reasoning chain.  
> Please produce:  
> 1) A re-usable prompt *template* (in YAML) that includes placeholders for each input field.  
> 2) Clear instructions for how the LLM should transform these inputs into the desired outputs.  
> 3) A short example usage scenario.  
> 4) A reflection on how the domain constraints map to instructions in the prompt.  
>
> Return the final DSL file as valid YAML with an inline example usage."

### Step C: LLM Emergent DSL Generation

An LLM (like GPT-4) reads your domain spec and meta-prompt. It *writes its own DSL* in structured YAML, e.g.:

```yaml
description: "Template for the S106 Negotiation Reasoning Chain"
variables:
  - name: site_context
    description: "Site constraints, location, existing consents"
  - name: policy_references
    description: "List of relevant local or SPD policies"
  - name: relevant_precedents
    description: "List of S106 or appeal precedents to consider"
  - name: financials
    description: "Project financial data (cost, land value, etc.)"
prompt_template: |
  You are an expert UK planning and viability officer. Your task is to propose an S106 package.

  Site context: {{ site_context }}
  Policy references: {{ policy_references }}
  Precedents: {{ relevant_precedents }}
  Financials: {{ financials }}

  Constraints:
  1) Always check alignment with NPPF viability guidance.
  2) Reflect typical local developer margins.
  3) For each policy, see if thresholds are triggered.

  Provide:
  - s106_recommendation: Summarize the recommended contributions
  - justification_text: A concise explanation referencing policy alignment and viability details

example_usage:
  input:
    site_context: "Mixed-use development of 50 flats, allocated site..."
    policy_references:
      - "LB Camden SPD on Affordable Housing"
      - "London Plan Policy H4"
    relevant_precedents:
      - "Appeal APP/X5210/W/20/323 from Camden in 2020"
    financials:
      build_cost: 12000000
      land_value: 4000000
  output:
    s106_recommendation: "Recommend 35% affordable housing + £150k towards local infrastructure"
    justification_text: "Consistent with SPD thresholds, minimal conflict with viability margin..."
```

Notice how the LLM applies domain constraints (NPPF viability, local thresholds) *within* the prompt.

### Step D: Validate & Iterate

- You review the generated YAML.  
- If you see issues or want more detail (e.g., “We forgot to mention a phasing plan for S106!”), you *edit the domain spec* or *augment the meta-prompt*, and re-run.  
- Over time, you build a library of these automatically generated DSL templates.

---

## 3. Key Benefits

1. **Scaling**  
   If tomorrow you add a new “Green Belt Exceptional Circumstances” chain, you just define a short domain spec, run the meta-prompt, and voilà—**LLM creates a DSL** so you aren’t writing each prompt from scratch.

2. **Consistency**  
   The same meta-prompt engine ensures consistent structure and language across all generated prompts, while letting each chain’s domain spec add custom constraints or instructions.

3. **Domain-Intelligent**  
   Because the LLM has inherent knowledge of UK planning law, viability nuances, policy constraints, etc., it can embed that understanding directly into the DSL’s instructions—yielding more correct prompts than a purely code-based approach that lacks domain nuance.

4. **Flexibility**  
   You can later unify these DSLs, or layer them with advanced chain-of-thought logic, without rewriting everything. The LLM can handle big refactors if you feed it the entire library and request a *“DSL Consolidation pass.”*

---

## 4. Next-Level Ideas

- **Self-Verification**: Have the LLM itself do a QA pass, checking if the final DSL meets the domain constraints.  
- **Chain-of-DSL**: For very complex tasks (e.g., “Simultaneous Green Belt + Heritage + S106 + design viability”), the meta-prompt could orchestrate multiple DSL templates into a single pipeline.  
- **LLM Functions**: If you’re using an LLM that supports function calling (like OpenAI’s function calling feature), you can define a standard function signature that the DSL must match—again letting the LLM generate function stubs.  
- **Prompt Composition**: Let the LLM reference *existing DSL templates* to see how similar tasks were structured, so it can build on them or ensure consistent variable naming.

---

### TL;DR

**Harness the LLM’s emergent domain smarts** by:
1. Writing a **minimal domain specification** for each new reasoning chain.  
2. Feeding that into a **meta-prompt** instructing the LLM to auto-generate a custom DSL template (YAML/JSON).  
3. **Iterate and refine** using the same or a secondary QA pass.  

That way, you can spin up sophisticated, domain-accurate prompt templates *faster* and *more consistently* than manual DSL writing, letting your system grow seamlessly across the entire planning domain.