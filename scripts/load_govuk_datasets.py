#!/usr/bin/env python3
import os
import sys
import requests
import subprocess
import tempfile
import getpass
import psycopg2 # For schema management & final update/indexing
import json
from pathlib import Path
from dotenv import load_dotenv

# --- Load Environment Variables ---
dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path, override=False)
print(f"INFO:     Attempting to load environment variables from: {dotenv_path.resolve()}", file=sys.stderr)

# --- Configuration ---
PG_HOST = os.getenv("PGHOST", "localhost")
PG_PORT = os.getenv("PGPORT", "5432")
PG_DATABASE = os.getenv("PGDATABASE", "planning_data")
PG_USER = os.getenv("PGUSER", "planner")
PG_SCHEMA = os.getenv("PGSCHEMA", "public")
PG_PASSWORD_ENV = os.getenv("PGPASSWORD")

TARGET_SRS_CODE_STR = os.getenv("TARGET_SRS", "4326")
GEOMETRY_NAME = os.getenv("GEOMETRY_NAME", "geom")
TARGET_TABLE_NAME = "spatial_constraints"
overwrite_str = os.getenv("OVERWRITE_TABLE", "True").lower()
OVERWRITE_TABLE = overwrite_str in ['true', '1', 't', 'y', 'yes']

# List of potential keys in source properties to check for the original ID
# These should match columns we are now creating explicitly below
POTENTIAL_ID_KEYS = ['OBJECTID', 'id', 'ID', 'identifier', 'reference', 'entity']

BASE_DOWNLOAD_URL = "https://files.planning.data.gov.uk/dataset/"
DATASET_SLUGS = [ # Keep the user's specified list
    "agricultural-land-classification", "air-quality-management-area", "ancient-woodland",
    "archaeological-priority-area", "area-of-outstanding-natural-beauty", "article-4-direction-area",
    "asset-of-community-value", "battlefield", "brownfield-land",
    "brownfield-site", "building-preservation-notice", "built-up-area",
    "central-activities-zone", "certificate-of-immunity", "parish", "conservation-area",
    "design-code-area", "educational-establishment",
    "flood-storage-area", "green-belt", "heritage-at-risk", "heritage-coast",
    "park-and-garden", "infrastructure-project", "listed-building",
    "listed-building-outline", "local-authority-district", "local-nature-reserve",
    "local-plan-boundary", "local-planning-authority", "local-resilience-forum-boundary",
    "locally-listed-building", "national-nature-reserve", "national-park",
    "nature-improvement-area", "protected-wreck-site", "transport-access-node",
    "ramsar", "region", "scheduled-monument", "site-of-special-scientific-interest",
    "special-area-of-conservation", "special-protection-area",
    "tree", "tree-preservation-zone", "ward", "world-heritage-site",
    "world-heritage-site-buffer-zone",
]
# --- End Configuration ---

def create_or_clear_target_table(pg_conn_details: dict, schema: str, table: str, srid_code_str: str):
    """
    Drops (if exists) and Creates the target table with explicitly defined common columns.
    """
    conn = None
    cur = None
    quoted_schema = f'"{schema}"'
    quoted_table = f'"{table}"'
    full_table_name = f"{quoted_schema}.{quoted_table}"
    quoted_geom_col_name = f'"{GEOMETRY_NAME}"'
    index_name_geom = f"{table}_geom_idx"
    index_name_type = f"{table}_type_idx"
    index_name_orig_id = f"{table}_orig_id_idx"

    try:
        numeric_srid = srid_code_str.split(':')[-1]
        int(numeric_srid)
    except (ValueError, AttributeError):
        print(f"ERROR: Invalid SRID format '{srid_code_str}'. Using 4326 as fallback.", file=sys.stderr)
        numeric_srid = "4326"

    # *** MODIFIED: Define expected property columns explicitly ***
    # Add other common columns found in the datasets as needed, mostly using TEXT.
    create_sql = f"""
    CREATE TABLE {full_table_name} (
        gid SERIAL PRIMARY KEY,
        constraint_type TEXT,         -- Populated by ogr2ogr -sql
        original_source_id TEXT,    -- Populated later via SQL UPDATE
        {quoted_geom_col_name} geometry(Geometry, {numeric_srid}), -- Geometry column
        -- Explicitly define common property columns (use TEXT for flexibility)
        name TEXT,
        dataset TEXT,
        reference TEXT,
        entity TEXT,
        prefix TEXT,
        typology TEXT,
        "start-date" TEXT, -- Quote names with hyphens
        "end-date" TEXT,
        "entry-date" TEXT, -- Use TEXT, can be parsed/cast later if needed
        "organisation-entity" TEXT,
        notes TEXT,
        "documentation-url" TEXT,
        -- Add other frequently occurring property columns here as TEXT
        point TEXT -- Example if 'point' is a property key sometimes
        -- properties JSONB -- Removed: Using individual columns instead
    );
    """
    # Define indexes needed
    index_sql = f"CREATE INDEX IF NOT EXISTS \"{index_name_geom}\" ON {full_table_name} USING GIST ({quoted_geom_col_name});"
    index_type_sql = f"CREATE INDEX IF NOT EXISTS \"{index_name_type}\" ON {full_table_name} (constraint_type);"
    index_orig_id_sql = f"CREATE INDEX IF NOT EXISTS \"{index_name_orig_id}\" ON {full_table_name} (original_source_id);"

    print(f"Attempting to recreate table {full_table_name} (pre-defined schema) with SRID {numeric_srid}...")
    try:
        conn = psycopg2.connect(**pg_conn_details)
        conn.autocommit = False
        with conn.cursor() as cur:
            print(f"Dropping table {full_table_name} if it exists...")
            cur.execute(f"DROP TABLE IF EXISTS {full_table_name} CASCADE;")
            print(f"Creating table {full_table_name} with pre-defined columns...")
            cur.execute(create_sql)
            print(f"Creating spatial index on {quoted_geom_col_name}...")
            cur.execute(index_sql)
            print("Creating index on constraint_type...")
            cur.execute(index_type_sql)
            print("Creating index on original_source_id...")
            cur.execute(index_orig_id_sql)
            # Add other indexes on common query columns (e.g., name) if desired
            # cur.execute(f"CREATE INDEX IF NOT EXISTS \"{table}_name_idx\" ON {full_table_name} (name);")

        conn.commit()
        print(f"Table {full_table_name} created and indexed.")
        return True
    except psycopg2.Error as e:
        print(f"DATABASE ERROR during table creation: {e}", file=sys.stderr)
        if conn: conn.rollback()
        return False
    except Exception as e:
        print(f"UNEXPECTED ERROR during table creation: {e}", file=sys.stderr)
        if conn: conn.rollback()
        return False
    finally:
        if conn: conn.close()

def download_file(url, target_path):
    """Downloads a file from a URL to a target path."""
    # (Implementation remains the same)
    print(f"Downloading: {url}...")
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(target_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Successfully downloaded to {target_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}", file=sys.stderr)
        if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 404:
             print(f"Note: Dataset likely not available (404). Skipping.", file=sys.stderr)
        return False
    except Exception as e:
        print(f"An unexpected error occurred during download: {e}", file=sys.stderr)
        return False

def load_geojson_to_postgis(
    geojson_path: str,
    pg_conn_str_ogr: str,
    target_schema: str,
    target_table: str,
    constraint_slug: str
    ):
    """
    Loads GeoJSON using ogr2ogr -append and -sql SELECT *.
    Relies on target table having pre-defined columns for properties.
    """
    nln_target_table_arg = target_table # Use table name only for -nln
    full_target_table_log = f"{target_schema}.{target_table}"
    quoted_geom_col_name = f'"{GEOMETRY_NAME}"'
    source_layer_name_sql = f'"{constraint_slug}"'

    print(f"Loading layer '{constraint_slug}' from {geojson_path} into PostGIS table {full_target_table_log}...")

    # *** Use -sql SELECT * but WITHOUT -addfields ***
    # ogr2ogr will map fields from SELECT * to existing columns by name (case-insensitive usually)
    # and ignore source fields that don't have a matching column in the target table.
    sql_statement = (
        f"SELECT *, '{constraint_slug}' AS constraint_type "
        f"FROM {source_layer_name_sql}"
    )

    ogr2ogr_cmd = [
        "ogr2ogr",
        "-f", "PostgreSQL",
        f"PG:{pg_conn_str_ogr}",
        str(geojson_path),
        "-nln", nln_target_table_arg,
        "-sql", sql_statement,          # Select all source props + add constraint type
        # "-addfields",                 # REMOVED: Columns pre-defined in target table
        "-nlt", "PROMOTE_TO_MULTI",
        # NB: Use the literal GEOMETRY_NAME w/o extra quotes
        "-lco", f"GEOMETRY_NAME={GEOMETRY_NAME}",
        "-lco", "PRECISION=NO",         # Still useful to avoid potential width checks
        "-append",
        "--config", "PG_USE_COPY", "YES",
        "-skipfailures",              # ADD BACK: Ignore features that fail (e.g., schema mismatch)
        "-progress",
    ]
    # *** END CHANGE ***

    if TARGET_SRS_CODE_STR and TARGET_SRS_CODE_STR != "4326":
        ogr2ogr_cmd.extend(["-t_srs", f"EPSG:{TARGET_SRS_CODE_STR}"])

    print(f"Executing ogr2ogr (summary): ogr2ogr -append -sql ... -nln {nln_target_table_arg} ... {geojson_path}")

    try:
        result = subprocess.run(
            ogr2ogr_cmd, check=True, capture_output=True, text=True,
            encoding='utf-8', errors='replace'
        )
        print(f"Successfully appended data for {constraint_slug} (to pre-defined columns).")
        if result.stderr:
             non_progress_stderr = '\n'.join(line for line in result.stderr.splitlines() if not line.startswith('0...10...'))
             if non_progress_stderr.strip():
                print("ogr2ogr Warnings/Info:\n", non_progress_stderr, file=sys.stderr)
        return True
    except FileNotFoundError:
        print("Error: 'ogr2ogr' command not found. Is GDAL installed and in your PATH?", file=sys.stderr)
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error loading {constraint_slug} using ogr2ogr.", file=sys.stderr)
        print(f"Command failed with exit code {e.returncode}", file=sys.stderr)
        stdout = e.stdout if e.stdout else ''
        stderr = e.stderr if e.stderr else ''
        print("ogr2ogr stdout:\n", stdout, file=sys.stderr)
        print("--- ogr2ogr stderr START ---", file=sys.stderr)
        print(stderr, file=sys.stderr)
        print("--- ogr2ogr stderr END ---", file=sys.stderr)
        return False
    except Exception as e:
        print(f"An unexpected error occurred during ogr2ogr execution ({type(e).__name__}): {e}", file=sys.stderr)
        return False

def populate_original_source_id(pg_conn_details: dict, schema: str, table: str):
    """
    Populates the original_source_id column using COALESCE on the now individual
    property columns created during the load. Uses psycopg2.
    """
    conn = None
    cur = None
    quoted_schema = f'"{schema}"'
    quoted_table = f'"{table}"'
    full_table_name = f"{quoted_schema}.{quoted_table}"

    print(f"\n--- Populating original_source_id in {full_table_name} ---")
    try:
        conn = psycopg2.connect(**pg_conn_details)
        conn.autocommit = False # Use transaction
        with conn.cursor() as cur:
            # 1. Get actual column names from the table
            print("Getting current table columns to find potential ID columns...")
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
            """, (schema, table))
            actual_columns = [row[0] for row in cur.fetchall()]
            print(f"Found columns: {actual_columns}")

            # 2. Build COALESCE arguments from existing columns matching potential keys
            coalesce_cols = []
            for key in POTENTIAL_ID_KEYS:
                # Find the actual column name matching the key (case-insensitive check)
                match = next((col for col in actual_columns if col.lower() == key.lower()), None)
                if match:
                    coalesce_cols.append(f'"{match}"::text') # Quote column name, cast to text

            if coalesce_cols:
                coalesce_args_sql = ", ".join(coalesce_cols)
                update_orig_id_sql = f"""
                    UPDATE {full_table_name}
                    SET original_source_id = COALESCE({coalesce_args_sql})
                    WHERE original_source_id IS NULL;
                """
                print(f"Executing UPDATE for original_source_id using columns: {coalesce_cols}...")
                # print(f"DEBUG SQL (Orig ID Update): {update_orig_id_sql}")
                cur.execute(update_orig_id_sql)
                print(f"UPDATE statement affected {cur.rowcount} rows for original_source_id.")
            else:
                print("No matching ID columns found in table from POTENTIAL_ID_KEYS. Skipping original_source_id update.")

        conn.commit()
        print(f"Successfully populated 'original_source_id' where possible.")
        return True
    except psycopg2.Error as e:
        print(f"DATABASE ERROR during original_source_id update: {e}", file=sys.stderr)
        if conn: conn.rollback()
        return False
    except Exception as e:
        print(f"UNEXPECTED ERROR during original_source_id update: {e}", file=sys.stderr)
        if conn: conn.rollback()
        return False
    finally:
        if conn: conn.close()

# ------------------------------ NEW INSPECTION FUNCTIONS ------------------------------

def get_existing_table_columns(pg_conn_details: dict, schema: str, table: str):
    """
    Fetch and return the list of existing columns in the specified Postgres table.
    This helps us compare actual DB columns with the JSON property names.
    """
    try:
        with psycopg2.connect(**pg_conn_details) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                """, (schema, table))
                return [row[0] for row in cur.fetchall()]
    except Exception as e:
        print(f"Error fetching existing column names: {e}", file=sys.stderr)
        return []

def inspect_geojson_fields(geojson_path, table_columns):
    """
    Inspect a GeoJSON file's properties and report any missing fields
    compared to the predefined PostGIS table schema.
    """
    try:
        with open(geojson_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if "features" not in data or not data["features"]:
                print(f"⚠️  No features found in {geojson_path.name}")
                return

            # Just inspect the first feature's keys as a sample
            sample_props = data["features"][0].get("properties", {})
            source_keys = set(sample_props.keys())

            # Compare ignoring case
            table_keys = set(col.lower() for col in table_columns)

            missing_in_table = [k for k in source_keys if k.lower() not in table_keys]
            present_in_table = [k for k in source_keys if k.lower() in table_keys]

            print(f"🔍 Inspecting: {geojson_path.name}")
            print(f"  ➕ Found {len(source_keys)} property keys in sample feature.")
            print(f"  ✅ Matched columns: {present_in_table}")
            if missing_in_table:
                print(f"  ⚠️  Unmatched fields (not in DB schema): {missing_in_table}")
            else:
                print("  🎉 All fields present in DB schema.")
    except Exception as e:
        print(f"Error inspecting {geojson_path}: {e}", file=sys.stderr)

# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Unified GeoJSON Download and PostGIS Load (Pre-defined Columns) ---")
    print(f"Target DB: {PG_DATABASE} on {PG_HOST}:{PG_PORT}, User: {PG_USER}, Schema: {PG_SCHEMA}")
    print(f"Target Table: {TARGET_TABLE_NAME}, Overwrite: {OVERWRITE_TABLE}")
    print(f"Target SRID: {TARGET_SRS_CODE_STR}")

    PG_PASSWORD = PG_PASSWORD_ENV
    if not PG_PASSWORD:
        print(f"Enter PostGIS password for user '{PG_USER}':")
        try: 
            PG_PASSWORD = getpass.getpass()
        except Exception as e: 
            sys.exit(f"Error reading password: {e}")
    if not PG_PASSWORD: 
        sys.exit("Error: Database password not provided.")

    pg_connection_details_psycopg = {
        "dbname": PG_DATABASE, "user": PG_USER, "password": PG_PASSWORD,
        "host": PG_HOST, "port": PG_PORT
    }
    pg_connection_string_ogr = (
        f"dbname='{PG_DATABASE}' user='{PG_USER}' password='{PG_PASSWORD}' "
        f"host='{PG_HOST}' port='{PG_PORT}'"
    )

    # 1. Create or Clear Target Table (with pre-defined columns)
    if OVERWRITE_TABLE:
        if not create_or_clear_target_table(pg_connection_details_psycopg, PG_SCHEMA, TARGET_TABLE_NAME, TARGET_SRS_CODE_STR):
            sys.exit("Exiting due to error during table creation.")
    else:
        print("Overwrite set to False. Appending data.")
        print("WARNING: Assumes target table exists with appropriate pre-defined columns.")

    # 🆕 2. Fetch existing table columns to compare vs. GeoJSON property names
    table_columns = get_existing_table_columns(pg_connection_details_psycopg, PG_SCHEMA, TARGET_TABLE_NAME)
    print(f"\n--- 🔍 SCHEMA FIELD CHECK ---")
    print(f"Existing DB columns in {PG_SCHEMA}.{TARGET_TABLE_NAME}: {table_columns}")

    # 3. Process each dataset slug
    success_count = 0
    fail_count = 0
    skip_count = 0
    some_data_loaded_successfully = False

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"\nUsing temporary directory for downloads: {temp_dir}")
        temp_dir_path = Path(temp_dir)

        for slug in DATASET_SLUGS:
            print(f"\n--- Processing dataset: {slug} ---")
            file_name = f"{slug}.geojson"
            download_url = f"{BASE_DOWNLOAD_URL}{file_name}"
            temp_geojson_path = temp_dir_path / file_name

            if download_file(download_url, temp_geojson_path):
                # 🔍 Inspect first, to see if all fields match
                inspect_geojson_fields(temp_geojson_path, table_columns)

                # Then attempt loading
                if load_geojson_to_postgis(
                    geojson_path=temp_geojson_path,
                    pg_conn_str_ogr=pg_connection_string_ogr,
                    target_schema=PG_SCHEMA,
                    target_table=TARGET_TABLE_NAME,
                    constraint_slug=slug
                ):
                    success_count += 1
                    some_data_loaded_successfully = True
                else:
                    fail_count += 1
                    print(f"Failed to load {slug}.")
            else:
                print(f"Skipping loading for {slug} due to download failure.")
                skip_count += 1

    # 4. Post-process data (if anything loaded or if appending)
    if some_data_loaded_successfully or not OVERWRITE_TABLE:
        if not populate_original_source_id(pg_connection_details_psycopg, PG_SCHEMA, TARGET_TABLE_NAME):
             print("WARNING: Post-processing (IDs) step failed. Check database errors.", file=sys.stderr)
    elif OVERWRITE_TABLE and not some_data_loaded_successfully:
         print("\nOverwrite was True, but no data was successfully loaded. Skipping post-processing.")

    # 5. Final Summary
    print("\n--- Processing Summary ---")
    print(f"Target Table: {PG_SCHEMA}.{TARGET_TABLE_NAME}")
    print(f"Successfully downloaded & loaded: {success_count}")
    print(f"Failed during ogr2ogr load:      {fail_count}")
    print(f"Skipped (download failed/404):   {skip_count}")
    print(f"Total datasets attempted:        {len(DATASET_SLUGS)}")
    print("----------------------------")

    if fail_count > 0:
        print(f"WARNING: {fail_count} dataset(s) failed during the ogr2ogr loading process. Check logs.", file=sys.stderr)
        sys.exit(1)
    else:
        print("Script finished successfully.")
        sys.exit(0)
