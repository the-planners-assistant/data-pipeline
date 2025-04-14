Okay, here is the updated `README.md` reflecting the changes to the GeoJSON loader script (loading into a single table) and keeping the documentation for the PDF processor accurate.

```markdown
# UK Planning Data Processing Tools

This repository contains Python scripts designed for processing different types of data relevant to UK planning and policy analysis.

**Scripts Included:**

1.  **GeoJSON PostGIS Loader:** Downloads specified GeoJSON datasets from `planning.data.gov.uk` and loads them into a **single, unified** PostGIS table (`spatial_constraints`), adding a column to identify the original constraint type.
2.  **PDF Policy Document Processor:** Processes PDF policy documents (e.g., local plans) to extract text, structure, and images for RAG and knowledge graph applications.

---

## 1. GeoJSON PostGIS Loader (Unified Table Output)

### Description

This Python script downloads specified geospatial datasets in GeoJSON format from the UK Planning Data platform (`files.planning.data.gov.uk`). It then uses the standard `ogr2ogr` command-line utility (part of GDAL) to load **all** datasets into a **single, unified** PostGIS table, typically named `spatial_constraints`. A `constraint_type` column is added during loading to identify the source dataset for each feature.

This unified table structure is designed to simplify querying across different constraint types based on location.

### Features

* Downloads GeoJSON datasets based on a configurable list of slugs[cite: 1].
* Loads all downloaded features into a **single** target PostGIS table (e.g., `spatial_constraints`).
* Adds a `constraint_type` column to the target table, populating it with the dataset slug (e.g., 'conservation_area', 'listed_building').
* Uses the robust `ogr2ogr` tool with `-append` and `-sql` flags for efficient loading and data transformation.
* Includes an optional step (controlled by `OVERWRITE_TABLE` config) to automatically drop and recreate the target table with the correct schema and indexes before loading.
* Handles temporary file management during the download process.
* Configurable database connection details via `.env` file or environment variables.
* Securely prompts for the database password if not provided otherwise.
* Provides progress messages and a final summary report.
* Basic error handling for downloads (including 404 Not Found) and `ogr2ogr`/database operations.

### Prerequisites

Before running this script, ensure you have the following installed and configured:

1.  **Python 3:** Version 3.6 or higher recommended.
2.  **Python Libraries:** Install using pip (ideally from `requirements.txt`):
    ```bash
    pip install requests python-dotenv psycopg2-binary
    ```
    * `requests`: For downloading files.
    * `python-dotenv`: For loading configuration from `.env` file.
    * `psycopg2-binary` (or `psycopg2`): Required *by the script* for the initial table drop/create step when `OVERWRITE_TABLE` is true.
3.  **GDAL (Geospatial Data Abstraction Library):** The `ogr2ogr` command-line tool must be installed and accessible in your system's PATH.
    * **Linux (Debian/Ubuntu):** `sudo apt update && sudo apt install gdal-bin`
    * **macOS (using Homebrew):** `brew install gdal`
    * **Windows:** Installation via [OSGeo4W](https://trac.osgeo.org/osgeo4w/), Conda Forge (`conda install -c conda-forge gdal`), or [GISInternals](https://www.gisinternals.com/release.php) is recommended. Ensure the `bin` directory is in your PATH.
    * **Verification:** Run `ogr2ogr --version` in your terminal.
4.  **PostgreSQL Server with PostGIS:** A running PostgreSQL database instance with the PostGIS extension enabled. See the **"Setting Up PostGIS"** section below for detailed instructions.

### Installation

1.  Save the script code (e.g., `load_planning_data.py`).
2.  Ensure all prerequisites are met and libraries installed (e.g., `pip install -r requirements.txt` if using the provided file).
3.  Create a `.env` file in the same directory as the script for configuration (see below).
4.  (Optional) Use a Python virtual environment.

### Configuration (`.env` file)

It is **highly recommended** to configure this script using a `.env` file placed in the same directory as the script (`load_planning_data.py`).

Create a file named `.env` with the following format:

```dotenv
# .env file - Example for GeoJSON Loader

# PostgreSQL Connection Details (Required)
PGHOST=localhost
PGPORT=5432
PGDATABASE=planning_data
PGUSER=planner
PGPASSWORD=your_database_password_here

# Optional: Target Schema (defaults to public if not set in config.py)
PGSCHEMA=public

# Optional: ogr2ogr/Loader Settings
TARGET_SRS=4326          # Target EPSG Code (e.g., 4326 for WGS84, 27700 for BNG)
GEOMETRY_NAME=geom       # Name for the geometry column in the target table
OVERWRITE_TABLE=True     # Set to True to drop/recreate the target table on run, False to just append
```

**Configuration Precedence:**

1.  System Environment Variables
2.  `.env` File Variables
3.  Script Defaults (defined in `load_planning_data.py`)
4.  Password Prompt (for `PGPASSWORD` only, as a last resort)

The `DATASET_SLUGS` list is configured directly within the Python script.

### Usage

1.  Navigate to the script directory in your terminal.
2.  Ensure your `.env` file is present and correctly configured.
3.  Run the script:
    ```bash
    python load_planning_data.py
    ```
4.  If `PGPASSWORD` was not set, enter the database password when prompted.
5.  Monitor the output. If `OVERWRITE_TABLE` was `True`, it will first drop/create the `spatial_constraints` table. Then, it will download each GeoJSON and append its features to the table.

### Output

* Creates (if `OVERWRITE_TABLE=True`) or appends data to a **single table** named `spatial_constraints` (within the configured `PGSCHEMA`).
* The table structure includes:
    * `gid`: Serial Primary Key.
    * `constraint_type`: Text (populated with the dataset slug, e.g., 'conservation_area').
    * `original_source_id`: Text (attempts to store an ID from the source GeoJSON).
    * `properties`: JSONB (contains the original feature properties).
    * `geom`: Geometry (using the name from `GEOMETRY_NAME` and SRID from `TARGET_SRS`).
* Spatial (GIST) and attribute indexes are created on the table for performance.

### Important Notes

* **`ogr2ogr` & `psycopg2`:** The script relies on `ogr2ogr` for loading and `psycopg2` for initial table management. Ensure both are installed and accessible.
* **`OVERWRITE_TABLE = True`:** This setting will **permanently delete** the *entire* existing `spatial_constraints` table before starting the load process. Use `False` if you want to add new datasets or re-run without losing existing data in that table.
* **Data Consistency:** Ensure the source GeoJSON files have reasonably consistent structures if you rely heavily on specific fields within the `properties` JSONB column later.
* **Large Datasets:** Some source files can be large; allow sufficient time and disk space for downloads and loading.
* **Database Permissions:** The `PGUSER` needs privileges to connect, use the schema, create/drop tables (if overwriting), and insert/select data.

---

## Setting Up PostGIS (for GeoJSON Loader)

These instructions explain how to prepare a PostgreSQL database with the PostGIS extension, making it ready for the `load_planning_data.py` script. Uses `psql` command-line client.

### Prerequisites

* **PostgreSQL Installed:** A working PostgreSQL server installation (version 12+ recommended).
* **`psql` Access:** Command-line access to `psql`, typically connecting as a superuser (e.g., `postgres`) initially.

### Steps

1.  **Connect as Superuser:**
    ```bash
    psql -U postgres -h localhost
    ```
    (Enter password if prompted).

2.  **Create Database (Recommended):**
    (Match `PGDATABASE` in `.env`).
    ```sql
    CREATE DATABASE planning_data;
    ```

3.  **Connect to New Database:**
    ```sql
    \c planning_data
    ```

4.  **Enable PostGIS Extension:**
    ```sql
    CREATE EXTENSION postgis;
    ```

5.  **Create Dedicated User (Recommended):**
    (Match `PGUSER` and `PGPASSWORD` in `.env`).
    ```sql
    CREATE USER planner WITH PASSWORD 'your_database_password_here';
    ```

6.  **Grant Necessary Privileges:**
    (Match `PGDATABASE`, `planner`, `public`/`PGSCHEMA`).
    ```sql
    -- Allow connection
    GRANT CONNECT ON DATABASE planning_data TO planner;

    -- Allow usage and creation in schema
    GRANT USAGE, CREATE ON SCHEMA public TO planner;

    -- Grant data manipulation permissions on existing & future tables
    GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO planner;
    ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO planner;
    ```

7.  **Verify (Optional):**
    Exit (`\q`), reconnect as `planner`, check PostGIS version (`SELECT PostGIS_Version();`).

---

## 2. PDF Policy Document Processor

### Description

This script processes PDF documents, typically local plans or policy documents, to extract text, structure, and images suitable for various downstream tasks like Retrieval-Augmented Generation (RAG) or building knowledge graphs.

### Overview

The script performs the following pipeline for each input PDF:

1.  **PDF to Markdown Conversion:** Uses the `pymupdf4llm` library.
2.  **Text Cleaning:** Removes formatting artifacts and noise.
3.  **Content Chunking:** Splits text using `semantic`, `fixed`, or `hybrid` strategies.
4.  **Image Extraction:** Extracts images using `PyMuPDF`.
5.  **Output Generation:** Produces flat JSONL, nested JSON, extracted images, and an image manifest.
6.  **Parallel Processing:** Processes multiple PDFs concurrently.

### Features

* Converts PDFs to clean Markdown text.
* Provides multiple chunking strategies (semantic, fixed-size, hybrid).
* Parses Markdown headers to create a nested document structure.
* Flattens nested structures into records suitable for vector database ingestion.
* Extracts images from PDFs and saves them with a manifest file.
* Processes multiple PDFs in parallel for efficiency.
* Configurable via command-line arguments.

### Dependencies

Requires Python libraries: `pymupdf4llm`, `PyMuPDF`, `langchain`. Install using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Installation

1.  Save the script code (e.g., `process_pdfs.py`).
2.  Ensure Python 3.x is installed.
3.  (Recommended) Use a Python virtual environment.
4.  Install dependencies: `pip install -r requirements.txt`.

### Output File Structure

For each input PDF (e.g., `mydocument.pdf`), generates in the output folder:

```
output_folder/
├── mydocument_flat.jsonl      # Flat chunks for vector DBs
├── mydocument_nested.json     # Nested structure for graph DBs/hierarchy
└── mydocument/                # Subdirectory for images
    ├── page_1_img_001.png   # Extracted image file
    │   ...
    └── images_manifest.json # Metadata about extracted images
```

### Configuration and Usage

Use command-line arguments:

```bash
python process_pdfs.py <input_folder> <output_folder> [--strategy {semantic,fixed,hybrid}] [--workers N]
```

* `input_folder`: **Required.** Folder containing source PDFs.
* `output_folder`: **Required.** Folder for output files/directories.
* `--strategy`: **Optional.** Chunking method (`semantic`, `fixed`, `hybrid`). Default: `hybrid`.
* `--workers`: **Optional.** Number of parallel processes. Default: `2`.

### Chunking Strategies Explained

* **`semantic`**: Uses Markdown headers (`#`-`######`) for structure-aware chunks. Skips numbered headings. Best for well-structured docs.
* **`fixed`**: Ignores structure, uses fixed-size overlapping chunks (~1000 chars). Good for unstructured docs.
* **`hybrid` (Default)**: Chooses `semantic` if >= 3 `##` headers exist, else `fixed`.

### Output Formats Explained

1.  **Flat JSONL (`<doc_id>_flat.jsonl`)**
    * One JSON object per line/chunk. For vector DBs/RAG.
    * Fields: `id`, `title`, `content`, `level`, `path`, `tokens`, `source`.

2.  **Nested JSON (`<doc_id>_nested.json`)**
    * Single JSON object representing document hierarchy. For graph DBs/structure analysis.
    * Structure: `{ "doc_id": "...", "sections": [ { "heading": "...", "content": "...", "tokens": ..., "subsections": [...] } ] }`

3.  **Images Directory (`<doc_id>/`) & Manifest**
    * Contains extracted image files.
    * `images_manifest.json`: Lists metadata for each image (`page_number`, `image_file`, `xref`, `image_index`).

### Function Descriptions

*(Refer to previous detailed documentation for descriptions of functions like `clean_chunk_text`, `parse_markdown_to_nested`, `extract_images`, etc.)*

### Notes and Troubleshooting

* Output quality depends heavily on input PDF quality/structure.
* Ensure all dependencies from `requirements.txt` are installed.
* High worker count or large PDFs can consume significant RAM.
* Experiment with `--strategy` for best results.

---

## License

AFFERO GPLV3