import json
import os
import uuid
import re
import regex # Using the third-party regex library for more features if needed
from dotenv import load_dotenv
from tqdm.auto import tqdm # Progress bars
from collections import Counter, defaultdict
import logging
import time

# --- Database Clients ---
from neo4j import GraphDatabase, ManagedTransaction, basic_auth
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, PayloadSchemaType, Filter, FieldCondition, MatchValue # Explicit imports

# --- NLP Libraries ---
from sentence_transformers import SentenceTransformer
import spacy
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
# Download necessary NLTK data if not already present
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    print("Downloading NLTK data (stopwords, punkt, averaged_perceptron_tagger)...")
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

from nltk.corpus import stopwords as nltk_stopwords


# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_URL = os.getenv("QDRANT_URL") # For Qdrant Cloud
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") # For Qdrant Cloud/Auth

# --- File Paths ---
# Adapt this list to include all documents you want to process
DOCUMENTS_TO_PROCESS = [
    {
        "doc_id": "the_london_plan_2021",
        "nested_json_path": os.getenv("NESTED_JSON_PATH", "./the_london_plan_2021_nested.json"),
        "ontology_map_path": os.getenv("ONTOLOGY_MAP_PATH", "./ontology.json"),
        "title": "The London Plan 2021",
        "type": "Spatial Development Strategy",
        "versionDate": "2021-03-01" # Example date
    },
    # Add other documents here, e.g.:
    # {
    #     "doc_id": "borough_x_local_plan_2023",
    #     "nested_json_path": "./borough_x_local_plan_2023_nested.json", # You'd need to create these nested JSONs
    #     "ontology_map_path": "./borough_x_ontology.json", # And ontology maps
    #     "title": "Borough X Local Plan",
    #     "type": "Local Plan",
    #     "versionDate": "2023-01-15"
    # },
]
DISCOVERED_CONCEPTS_OUTPUT_PATH = "./discovered_concepts_for_review.json"
MANUALLY_REVIEWED_CONCEPTS_PATH = "./manually_reviewed_concepts.json" # Path to load reviewed concepts from
SYNONYM_MAP_PATH = "./synonyms.json" # Path for manual synonym map

# --- Qdrant Config ---
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "planning_doc_chunks")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", 768))

# --- NLP Config ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", 'sentence-transformers/all-mpnet-base-v2')
SPACY_MODEL = os.getenv("SPACY_MODEL", 'en_core_web_lg')
# Stopwords for concept extraction
PLANNING_STOPWORDS = set(nltk_stopwords.words('english')).union({
    'policy', 'development', 'proposal', 'plan', 'borough', 'london', 'mayor', 'gla', 'tfl',
    'shall', 'must', 'should', 'may', 'use', 'area', 'site', 'need', 'provide', 'ensure', 'support',
    'part', 'chapter', 'figure', 'table', 'annex', 'paragraph', 'section', 'also', 'within',
    'including', 'regard', 'example', 'etc', 'guidance', 'strategy', 'accordance', 'requirement',
    'consideration', 'approach', 'local', 'impact', 'level', 'term', 'role', 'appropriate',
    'relevant', 'potential', 'significant', 'likely', 'possible', 'existing', 'new', 'set', 'out',
    'per', 'cent', 'fig', 'ref', 'where', 'number', 'order', 'defined', 'term', 'ha', 'sqm',
    'summary', 'key', 'page', 'contents', 'list', 'introduction', 'glossary', 'index', 'data',
    'note', 'source', 'copyright', 'contributor', 'credit', 'map', 'diagram', 'illustration',
    'www', 'http', 'https'
    # Add more domain-specific stopwords
})
# POS tags to keep for n-grams/keywords (adjust based on testing)
VALID_POS_PATTERNS = {
    ('NOUN',), ('PROPN',),
    ('ADJ', 'NOUN'), ('ADJ', 'PROPN'),
    ('NOUN', 'NOUN'), ('NOUN', 'PROPN'),
    ('PROPN', 'NOUN'), ('PROPN', 'PROPN'),
    ('ADJ', 'NOUN', 'NOUN'), ('ADJ', 'PROPN', 'NOUN'), ('ADJ', 'NOUN', 'PROPN'),
    ('NOUN', 'NOUN', 'NOUN'), ('PROPN', 'NOUN', 'NOUN'), ('NOUN', 'PROPN', 'NOUN'), ('NOUN', 'NOUN', 'PROPN'),
    ('PROPN', 'PROPN', 'NOUN'), ('PROPN', 'NOUN', 'PROPN'), ('NOUN', 'PROPN', 'PROPN'),
    ('PROPN', 'PROPN', 'PROPN'),
    # Allow common patterns like 'agent of change' (NOUN, ADP, NOUN)
    ('NOUN', 'ADP', 'NOUN'), ('PROPN', 'ADP', 'NOUN'), ('NOUN', 'ADP', 'PROPN'),
    ('NOUN', 'CCONJ', 'NOUN'), # e.g., 'research and development'
    ('ADJ', 'CCONJ', 'ADJ', 'NOUN') # e.g., 'black, asian and minority ethnic' - might need refining
}
# Map spaCy NER labels to your desired Concept types and Neo4j Labels
NER_LABEL_MAP = {
    "ORG": {"type": "Organisation", "label": "OrganisationConcept"},
    "GPE": {"type": "SpatialArea", "label": "SpatialConcept"}, # Geo-political entity
    "LOC": {"type": "SpatialArea", "label": "SpatialConcept"}, # Location
    "FAC": {"type": "Infrastructure", "label": "InfrastructureConcept"}, # Facility
    "PRODUCT": {"type": "Topic", "label": "TopicConcept"}, # Can be relevant sometimes
    "LAW": {"type": "Topic", "label": "TopicConcept"},
    "PERSON": None, # Usually ignore persons for policy analysis
    "NORP": None, # Nationalities or religious or political groups - ignore?
    "EVENT": None, # Ignore events?
    "WORK_OF_ART": {"type": "Culture", "label": "CultureConcept"},
    "LANGUAGE": None,
    "DATE": None,
    "TIME": None,
    "PERCENT": {"type": "Metric", "label": "MetricConcept"}, # Could capture percentages
    "MONEY": None,
    "QUANTITY": {"type": "Metric", "label": "MetricConcept"},# Could capture quantities
    "ORDINAL": None,
    "CARDINAL": None,
}

# Batch sizes for uploads
QDRANT_BATCH_SIZE = 100
NEO4J_BATCH_SIZE = 500
CONCEPT_LINKING_BATCH_SIZE = 2000 # Linking concepts can be memory intensive

# Concept Discovery Config
MIN_TERM_FREQ = 3 # Minimum times a discovered concept must appear overall
MIN_KEYBERT_SCORE = 0.3 # Minimum relevance score for KeyBERT terms
MAX_KEYBERT_WORDS = 3 # Max words in KeyBERT phrases
NUM_KEYBERT_KEYWORDS = 500 # Number of candidates from KeyBERT
NUM_TFIDF_KEYWORDS = 500 # Number of candidates from TFIDF
SUBSUMPTION_COUNT_MULTIPLIER = 5 # How many times more frequent must shorter term be to keep it?

# --- Initialize Clients and Models ---
logger.info("Initializing clients and models...")
try:
    neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=basic_auth(NEO4J_USER, NEO4J_PASSWORD))
    neo4j_driver.verify_connectivity()
    logger.info("Neo4j connection successful.")
except Exception as e:
    logger.error(f"Neo4j connection failed: {e}", exc_info=True)
    exit()

try:
    if QDRANT_URL:
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=True)
    else:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, prefer_grpc=True)
    qdrant_client.get_collections() # Basic check
    logger.info("Qdrant connection successful.")
except Exception as e:
    logger.error(f"Qdrant connection failed: {e}", exc_info=True)
    exit()

logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
logger.info(f"Loading spaCy model: {SPACY_MODEL}")
try:
    # Disable components not needed for normalization/POS/NER to speed up
    nlp = spacy.load(SPACY_MODEL, disable=['parser', 'attribute_ruler'])
except OSError:
    logger.warning(f"Spacy model '{SPACY_MODEL}' not found. Downloading...")
    spacy.cli.download(SPACY_MODEL)
    nlp = spacy.load(SPACY_MODEL, disable=['parser', 'attribute_ruler'])

kw_model = KeyBERT(model=embedding_model)
logger.info("Initialization complete.")


# --- Helper Functions ---
def load_json(filepath):
    """Loads a JSON file."""
    logger.info(f"Loading JSON from {filepath}...")
    if not os.path.exists(filepath):
        logger.error(f"File not found at {filepath}")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON file {filepath}: {e}")
        return None

def load_synonyms(filepath):
    """Loads a JSON file containing synonyms mapping, returns an empty dictionary if file not found or error occurs."""
    logger.info(f"Loading synonyms from {filepath}...")
    if not os.path.exists(filepath):
        logger.warning(f"Synonyms file not found at {filepath}, using empty map.")
        return {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading synonyms file {filepath}: {e}")
        return {}

def get_policy_id(title):
    """Attempts to extract a policy ID like GG1, SD1, H4, T6.1 etc."""
    # Regex to find patterns like "Policy [A-Z]{1,3}\d{1,2}(\.\d{1,2})?" or "GG\d" at the start
    match = regex.match(r"^(Policy\s+([A-Z]{1,3}\d{1,2}(\.\d{1,2})?)|(GG\d+)|([A-Z]{1,3}\d{1,2}(\.\d{1,2})?))\b", title, regex.IGNORECASE)
    if match:
        if match.group(2): p_id = match.group(2)
        elif match.group(4): p_id = match.group(4)
        elif match.group(5): p_id = match.group(5)
        else: return None
        return p_id.replace(" ", "").upper() # Clean up
    return None

def flatten_nested_json(node, doc_id, source_ontology_map, parent_uuid=None, path_list=None, sequence=0):
    """Recursively flattens nested JSON, adds UUIDs, docId, and sequence."""
    flat_list = []
    chunk_uuid = str(uuid.uuid4())
    # Attempt to get original ID from 'id' field, fallback to UUID
    source_id = node.get('id', chunk_uuid)
    # Use source_id (original ID if present, else the new UUID) to lookup ontology
    content_type = source_ontology_map.get(source_id, "Unknown")

    current_path = path_list + [node.get('heading', 'Untitled')] if path_list else [node.get('heading', 'Untitled')]

    flat_node = {
        "chunkId": chunk_uuid,
        "sourceChunkId": source_id if source_id != chunk_uuid else None, # Store original ID only if it existed
        "docId": doc_id,
        "title": node.get('heading', ''),
        "text": node.get('content', ''),
        "level": len(current_path),
        "path": current_path,
        "tokens": node.get('tokens', 0),
        "contentType": content_type,
        "parentChunkId": parent_uuid,
        "sequenceInParent": sequence, # Store sequence
        "policyId": get_policy_id(node.get('heading', '')) if content_type.startswith("Policy") else None
    }
    flat_list.append(flat_node)

    if 'subsections' in node and node['subsections']:
        for i, sub_node in enumerate(node['subsections']):
            flat_list.extend(flatten_nested_json(sub_node, doc_id, source_ontology_map, parent_uuid=chunk_uuid, path_list=current_path, sequence=i)) # Pass sequence i

    return flat_list

def ensure_qdrant_collection(client, collection_name, vector_size):
    """Creates Qdrant collection if it doesn't exist."""
    try:
        client.get_collection(collection_name=collection_name)
        logger.info(f"Qdrant collection '{collection_name}' already exists.")
    except Exception as e:
        # Check for specific 'Not Found' conditions based on client behavior
        not_found_conditions = [
            "not found",                    # General gRPC/REST not found message
            "status_code=404",              # REST API status code
            "StatusCode.NOT_FOUND"          # Potential gRPC status code string representation
        ]
        if any(cond in str(e).lower() for cond in not_found_conditions):
            logger.info(f"Creating Qdrant collection '{collection_name}'...")
            try:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                    optimizers_config=models.OptimizersConfigDiff(memmap_threshold=20000),
                    hnsw_config=models.HnswConfigDiff(on_disk=True, m=16, ef_construct=100)
                )
                # Create payload indexes immediately after creation
                try:
                    client.create_payload_index(collection_name=collection_name, field_name="docId", field_schema=PayloadSchemaType.KEYWORD)
                    client.create_payload_index(collection_name=collection_name, field_name="contentType", field_schema=PayloadSchemaType.KEYWORD)
                    client.create_payload_index(collection_name=collection_name, field_name="policyId", field_schema=PayloadSchemaType.KEYWORD)
                    logger.info(f"Collection '{collection_name}' created with payload indexes.")
                except Exception as idx_e:
                    logger.error(f"Failed to create payload indexes for {collection_name}: {idx_e}", exc_info=True)
            except Exception as create_e:
                 logger.error(f"Failed to create Qdrant collection {collection_name}: {create_e}", exc_info=True)
                 raise create_e # Reraise creation error
        else:
            logger.error(f"Unexpected error checking Qdrant collection {collection_name}: {e}", exc_info=True)
            raise e # Reraise unexpected error

def ingest_to_qdrant(client, collection_name, chunks_data):
    """Ingests chunks into Qdrant with embeddings."""
    logger.info(f"Starting ingestion into Qdrant collection '{collection_name}'...")
    points_to_upsert = []
    texts_to_embed = []
    chunk_id_map = {} # Map original text index to chunkId

    for i, chunk in enumerate(chunks_data):
        # Only embed chunks with meaningful text
        text_content = chunk.get('text', '').strip()
        if text_content:
            texts_to_embed.append(text_content)
            chunk_id_map[len(texts_to_embed) - 1] = chunk['chunkId']

    if not texts_to_embed:
        logger.warning("No text found in chunks to embed.")
        return

    logger.info(f"Generating embeddings for {len(texts_to_embed)} non-empty chunks...")
    embeddings = embedding_model.encode(texts_to_embed, show_progress_bar=True, batch_size=32) # Adjust batch_size based on VRAM

    logger.info("Preparing points for Qdrant...")
    for i, embedding in enumerate(embeddings):
        chunk_id = chunk_id_map.get(i)
        if chunk_id is None:
            logger.warning(f"Could not find chunkId for embedding index {i}. Skipping.")
            continue

        # Find the original chunk data using the chunk_id
        chunk = next((c for c in chunks_data if c['chunkId'] == chunk_id), None)
        if chunk:
            payload = {
                "chunkId": chunk["chunkId"], # Redundant? ID is the point ID
                "docId": chunk["docId"],
                "contentType": chunk["contentType"],
                "title": chunk["title"],
                "level": chunk["level"],
                "text": chunk["text"] # Store text for retrieval context
            }
            # Only add policyId to payload if it exists and is not None
            if chunk.get("policyId"):
                payload["policyId"] = chunk["policyId"]

            points_to_upsert.append(PointStruct( # Use explicit import
                id=chunk["chunkId"], # Use the UUID as the point ID
                vector=embedding.tolist(),
                payload=payload
            ))
        else:
             logger.warning(f"Could not find chunk data for chunkId {chunk_id} during point preparation.")

    if not points_to_upsert:
         logger.warning("No points generated for Qdrant upsert.")
         return

    logger.info(f"Upserting {len(points_to_upsert)} points to Qdrant in batches of {QDRANT_BATCH_SIZE}...")
    try:
        client.upsert(
            collection_name=collection_name,
            points=points_to_upsert,
            wait=True, # Ensure operation is finished before proceeding
            batch_size=QDRANT_BATCH_SIZE # Let the client handle batching
        )
        # Verify count after upsert
        count_result = client.count(collection_name=collection_name, exact=True)
        logger.info(f"Qdrant ingestion complete. Collection count: {count_result.count}")

    except Exception as e:
        logger.error(f"Error during Qdrant upsert: {e}", exc_info=True)


# --- Neo4j Cypher Functions ---

def clear_neo4j_database(tx: ManagedTransaction):
    """Clears all nodes and relationships from the Neo4j database."""
    logger.warning("CLEARING Neo4j database...")
    tx.run("MATCH (n) DETACH DELETE n")
    logger.info("Neo4j database cleared.")

def create_neo4j_constraints(tx: ManagedTransaction):
    """Creates unique constraints for key node properties."""
    logger.info("Creating Neo4j constraints...")
    try:
        tx.run("CREATE CONSTRAINT unique_docId IF NOT EXISTS FOR (d:Document) REQUIRE d.docId IS UNIQUE")
        tx.run("CREATE CONSTRAINT unique_chunkId IF NOT EXISTS FOR (dc:DocumentChunk) REQUIRE dc.chunkId IS UNIQUE")
        # Use OPTIONAL MATCH for Concept constraint creation if using Neo4j < 5.x
        # For Neo4j 5.x+, REQUIRE ... IS NODE KEY is preferred but simpler UNIQUE is fine here.
        tx.run("CREATE CONSTRAINT unique_conceptName IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE")
        tx.run("CREATE CONSTRAINT unique_glossaryTerm IF NOT EXISTS FOR (gt:GlossaryTerm) REQUIRE gt.term IS UNIQUE")
        logger.info("Neo4j constraints created or verified.")
    except Exception as e:
        logger.error(f"Failed to create constraints: {e}", exc_info=True)

def load_neo4j_document(tx: ManagedTransaction, doc_data):
    """Loads a single Document node."""
    logger.debug(f"Loading Document node: {doc_data['doc_id']}")
    query = """
    MERGE (d:Document {docId: $doc_id})
    SET d.title = $title,
        d.type = $type,
        d.versionDate = date($versionDate) // Ensure versionDate is YYYY-MM-DD string
    """
    try:
        tx.run(query, doc_id=doc_data['doc_id'], title=doc_data.get('title', doc_data['doc_id']), type=doc_data.get('type', 'Unknown'), versionDate=doc_data.get('versionDate'))
    except Exception as e:
        logger.error(f"Failed to load document {doc_data['doc_id']}: {e}", exc_info=True)

def load_neo4j_chunks_batch(tx: ManagedTransaction, chunks_batch):
    """Loads a batch of DocumentChunk nodes and applies labels."""
    # Using map projection for cleaner property setting
    # Using APOC for dynamic labels is preferred if available
    apoc_query = """
    UNWIND $chunks AS chunk
    MERGE (dc:DocumentChunk {chunkId: chunk.chunkId})
    // Set properties using map projection, excluding chunkId to avoid self-setting
    SET dc += apoc.map.removeKeys(chunk, ['chunkId'])
    // Apply Policy/Objective labels based on contentType/policyId
    WITH dc, chunk
    CALL apoc.create.addLabels( dc, CASE WHEN chunk.policyId IS NOT NULL THEN ['Policy'] ELSE [] END ) YIELD node AS policyNode
    WITH policyNode AS dc, chunk // Pass the node through
    CALL apoc.create.addLabels( dc, CASE WHEN chunk.policyId IS NOT NULL AND chunk.policyId STARTS WITH 'GG' THEN ['Objective'] ELSE [] END ) YIELD node
    RETURN count(*)
    """
    # Fallback query without APOC
    fallback_query = """
    UNWIND $chunks AS chunk
    MERGE (dc:DocumentChunk {chunkId: chunk.chunkId})
    SET dc.sourceChunkId = chunk.sourceChunkId,
        dc.docId = chunk.docId,
        dc.title = chunk.title,
        dc.text = chunk.text, // Consider excluding large text if causing issues
        dc.level = chunk.level,
        dc.path = chunk.path,
        dc.contentType = chunk.contentType,
        dc.sequenceInParent = chunk.sequenceInParent,
        dc.policyId = chunk.policyId
    FOREACH(ignoreMe IN CASE WHEN chunk.policyId IS NOT NULL THEN [1] ELSE [] END | SET dc:Policy)
    FOREACH(ignoreMe IN CASE WHEN chunk.policyId IS NOT NULL AND chunk.policyId STARTS WITH 'GG' THEN [1] ELSE [] END | SET dc:Objective)
    """
    try:
        # Attempt to use APOC query first
        tx.run(apoc_query, chunks=chunks_batch)
    except Exception as e:
         # Check if the error is due to APOC procedure not found
         if "unknown procedure" in str(e).lower() and "apoc.create.addlabels" in str(e).lower():
              logger.warning("APOC procedure apoc.create.addLabels not found. Falling back to basic SET labels.")
              try:
                  tx.run(fallback_query, chunks=chunks_batch)
              except Exception as fallback_e:
                  logger.error(f"Error executing fallback chunk loading query: {fallback_e}", exc_info=True)
                  # Optionally re-raise or handle more gracefully
                  raise fallback_e
         elif "apoc.map.removekeys" in str(e).lower():
             logger.warning("APOC procedure apoc.map.removeKeys not found. Falling back to explicit SET properties.")
             # Re-run fallback query which explicitly sets properties
             try:
                 tx.run(fallback_query, chunks=chunks_batch)
             except Exception as fallback_e:
                 logger.error(f"Error executing fallback chunk loading query (after removeKeys failure): {fallback_e}", exc_info=True)
                 raise fallback_e
         else:
              # Log other unexpected errors
              logger.error(f"Unexpected error loading chunk batch: {e}", exc_info=True)
              raise e # Re-raise other errors

def create_neo4j_hierarchy_batch(tx: ManagedTransaction, hierarchy_batch):
    """Creates PART_OF and HAS_CHUNK relationships."""
    query_part_of = """
    UNWIND $hierarchy_links AS link
    MATCH (child:DocumentChunk {chunkId: link.childId})
    MATCH (parent:DocumentChunk {chunkId: link.parentId})
    MERGE (child)-[r:PART_OF]->(parent)
    SET r.order = link.order // Use sequenceInParent for ordering
    """
    query_has_chunk = """
    UNWIND $document_links AS link
    MATCH (doc:Document {docId: link.docId})
    MATCH (chunk:DocumentChunk {chunkId: link.chunkId})
    MERGE (doc)-[:HAS_CHUNK]->(chunk)
    """
    try:
        # Process links where parent is another chunk
        part_of_links = [h for h in hierarchy_batch if h.get('parentId')]
        if part_of_links:
            tx.run(query_part_of, hierarchy_links=part_of_links)

        # Process links where parent is the document
        doc_links = [h for h in hierarchy_batch if not h.get('parentId')]
        if doc_links:
            tx.run(query_has_chunk, document_links=doc_links)
    except Exception as e:
        logger.error(f"Error creating hierarchy links: {e}", exc_info=True)

def create_neo4j_supports_rels_batch(tx: ManagedTransaction, supports_batch):
    """Creates SUPPORTS relationships between text chunks and policies."""
    query = """
    UNWIND $supports_links AS link
    MATCH (text:DocumentChunk {chunkId: link.textId})
    MATCH (policy:Policy {chunkId: link.policyId}) // Assumes policy node exists and has Policy label
    MERGE (text)-[:SUPPORTS]->(policy)
    """
    try:
        tx.run(query, supports_links=supports_batch)
    except Exception as e:
        logger.error(f"Error creating SUPPORTS links: {e}", exc_info=True)

def extract_and_link_policy_refs(tx: ManagedTransaction, chunk_id, chunk_text, current_doc_id):
    """Finds and links policy references within text."""
    # Example patterns - NEEDS ROBUST TESTING AND EXPANSION
    policy_pattern = r'(Policy\s+([A-Z]{1,3}\d{1,2}(\.\d{1,2})?))|(NPPF\s+Paragraph\s+(\d+))'
    try:
        matches = re.finditer(policy_pattern, chunk_text, re.IGNORECASE | re.DOTALL)
        links_created = 0
        for match in matches:
            context_snippet = chunk_text[max(0, match.start()-30):min(len(chunk_text), match.end()+30)].replace('\n',' ') # Context window
            target_policy_id = None
            target_doc_id = current_doc_id # Default to same document

            if match.group(1): # Policy XXX pattern
                target_policy_id = match.group(2).replace(" ", "").upper()
            elif match.group(4): # NPPF Paragraph YYY pattern
                target_policy_id = f"NPPF_Para{match.group(5)}"
                target_doc_id = "nppf_2023" # Example - need mapping for different docs

            if target_policy_id:
                # Avoid self-references based on policyId
                result = tx.run("MATCH (dc:DocumentChunk {chunkId: $chunk_id}) RETURN dc.policyId as policyId", chunk_id=chunk_id).single()
                current_chunk_policy_id = result['policyId'] if result else None

                if target_policy_id != current_chunk_policy_id:
                    # Check if target exists before creating relationship
                    target_exists_result = tx.run("""
                        MATCH (target:Policy)
                        WHERE target.policyId = $target_p_id AND target.docId = $target_d_id
                        RETURN target.chunkId AS targetChunkId LIMIT 1
                    """, target_p_id=target_policy_id, target_d_id=target_doc_id).single()

                    if target_exists_result:
                        # Create relationship if target found
                        tx.run("""
                            MATCH (source:DocumentChunk {chunkId: $source_id})
                            MATCH (target:Policy {chunkId: $targetChunkId}) // Match by chunkId now
                            MERGE (source)-[r:REFERENCES_POLICY]->(target)
                            SET r.context = $context // Store matched text as context
                        """, source_id=chunk_id, targetChunkId=target_exists_result['targetChunkId'], context=context_snippet)
                        links_created += 1
                    # else:
                    #     logger.debug(f"Policy reference target not found: ID={target_policy_id}, Doc={target_doc_id} (referenced in chunk {chunk_id})")
        # if links_created > 0: logger.debug(f"Created {links_created} policy ref links for chunk {chunk_id[:8]}")
    except Exception as e:
        logger.error(f"Error linking policy refs for chunk {chunk_id}: {e}", exc_info=True)


def normalize_term(term, nlp):
    """Normalize a term using spaCy lemmatization and returns lowercased lemmas."""
    doc = nlp(term)
    return " ".join([token.lemma_.lower() for token in doc if token.is_alpha])

# Added get_canonical_form function to compute the canonical form of a term.
def get_canonical_form(term, nlp_processor, acronym_map, synonym_map):
    """
    Computes the canonical form of a term by lowercasing, lemmatizing,
    and mapping via synonym_map or acronym_map if available.
    """
    # Normalize the term using spaCy lemmatization
    normalized = " ".join([token.lemma_.lower() for token in nlp_processor(term) if token.is_alpha])
    # Map using synonym_map if available
    if synonym_map and normalized in synonym_map:
        normalized = synonym_map[normalized]
    # Map using acronym_map if available
    if acronym_map and normalized in acronym_map:
        normalized = acronym_map[normalized]
    return normalized

def detect_acronyms(texts):
    """
    Detects acronyms from a list of texts using a simple regex and returns a dictionary mapping each acronym to itself.
    """
    acronyms = {}
    pattern = r'\b[A-Z]{2,}\b'
    for text in texts:
        for match in re.findall(pattern, text):
            acronyms[match] = match
    return acronyms

def get_pos_tags(text, nlp):
    """
    Returns a list of part-of-speech tags for the given text using the provided spaCy model.
    """
    doc = nlp(text)
    return [token.pos_ for token in doc]

def is_valid_pos_pattern(pos_tags, valid_patterns):
    """
    Returns True if the tuple of pos_tags matches one of the valid patterns.
    """
    return tuple(pos_tags) in valid_patterns

def process_glossary(tx: ManagedTransaction, glossary_chunks):
    """Parses glossary chunks and creates GlossaryTerm nodes."""
    # Needs robust parsing based on actual glossary structure
    logger.info(f"Processing {len(glossary_chunks)} potential Glossary Chunks...")
    all_terms = set()
    for chunk in glossary_chunks:
        # Example parsing: assumes "**Term** Definition" or "Term\nDefinition"
        text = chunk['text'].strip()
        term = None
        definition = None

        # Try **Term** Definition format
        bold_match = re.match(r"^\*\*?([\w\s\-/'’]+)\*\*?\s*(.*)", text, re.DOTALL)
        if bold_match and len(bold_match.group(1).split()) < 10: # Limit title length
            term = bold_match.group(1).strip()
            definition = bold_match.group(2).strip()

        # Try Term\nDefinition format
        elif '\n' in text:
            parts = text.split('\n', 1)
            potential_term = parts[0].strip()
            # Heuristic: check if first line looks like a term heading
            if len(potential_term.split()) < 10 and not potential_term.endswith(('.', '?')):
                 term = potential_term
                 definition = parts[1].strip() if len(parts) > 1 else ""

        if term and definition and len(term) > 1:
            # Normalize the term
            norm_term = normalize_term(term, nlp)
            if norm_term and norm_term not in PLANNING_STOPWORDS:
                try:
                    # Merge term node, set definition, link chunk
                    tx.run("""
                        MATCH (dc:DocumentChunk {chunkId: $chunk_id})
                        MERGE (gt:GlossaryTerm {term: $norm_term}) // Use normalized term
                        ON CREATE SET gt.definition = $definition, gt.originalTerm = $original_term
                        ON MATCH SET gt.definition = $definition // Update definition if term exists
                        // Ensure relationship is created
                        MERGE (dc)-[:DEFINES]->(gt)
                    """, chunk_id=chunk['chunkId'], norm_term=norm_term, definition=definition, original_term=term)
                    all_terms.add(norm_term)
                except Exception as e:
                    logger.error(f"Error processing glossary term '{term}' from chunk {chunk['chunkId']}: {e}", exc_info=True)
        # else:
        #     logger.debug(f"Could not parse term/definition from glossary chunk {chunk['chunkId'][:8]}...")

    logger.info(f"Processed and normalized {len(all_terms)} unique glossary terms.")
    return list(all_terms) # Return list of normalized terms


def link_glossary_mentions(tx: ManagedTransaction, chunk_id, chunk_text, glossary_terms_list):
    """Links chunk text mentions to GlossaryTerm nodes using normalized terms."""
    if not chunk_text or not glossary_terms_list: return
    lower_text = chunk_text.lower()
    links_created = 0
    try:
        # Use pre-compiled patterns if possible (pass patterns instead of terms list)
        # For simplicity here, re-compiling per chunk batch (less efficient)
        for term in glossary_terms_list:
            # Use word boundaries (\b) for more precise matching
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, lower_text):
                res = tx.run("""
                     MATCH (dc:DocumentChunk {chunkId: $chunk_id})
                     MATCH (gt:GlossaryTerm {term: $term}) // Match normalized term in graph
                     MERGE (dc)-[:MENTIONS_TERM]->(gt)
                     RETURN count(*) AS c
                 """, chunk_id=chunk_id, term=term).single()
                if res and res['c'] > 0: links_created+=1
        # if links_created > 0: logger.debug(f"Created {links_created} glossary links for chunk {chunk_id[:8]}")
    except Exception as e:
        logger.error(f"Error linking glossary mentions for chunk {chunk_id}: {e}", exc_info=True)


# --- Concept Discovery Function (Updated) ---
def discover_concepts(all_chunks_data, nlp_processor, acronym_map, synonym_map):
    """Hybrid approach including normalization, POS, subsumption, and mapping."""
    logger.info("--- Starting Concept Discovery ---")
    discovered_raw = defaultdict(lambda: {"type": "Topic", "label": "TopicConcept", "count": 0, "sources": set()})
    all_text_corpus = [c['text'] for c in all_chunks_data if c.get('text')]
    full_doc_text_for_extraction = " ".join(all_text_corpus)

    # 1. Extract from Structure (Titles and Glossary)
    logger.info("Extracting from structure (Titles and Glossary)...")
    structured_terms = set()
    # ... (Get glossary terms - assumed available from a previous step or passed in) ...
    # Use glossary_terms_discovered if populated earlier, otherwise extract here if needed
    all_glossary_chunks = [c for c in all_chunks_data if c.get('contentType', '').endswith('Glossary/Definitions')]
    if all_glossary_chunks:
         with neo4j_driver.session(database="neo4j") as session:
             glossary_terms_discovered = set(session.execute_read(process_glossary, all_glossary_chunks))
             for term in glossary_terms_discovered:
                 if term not in PLANNING_STOPWORDS and len(term) > 1:
                     discovered_raw[term]['sources'].add("glossary")

    structural_title_patterns = r'^(chapter|annex|policy|figure|table|part)\s*\d*'
    for chunk in all_chunks_data:
        title = chunk['title'].strip(' *-:').strip()
        if 3 < len(title) < 100 and len(title.split()) <= 7:
            term = title.lower().strip()
            if term and term not in PLANNING_STOPWORDS and not re.match(structural_title_patterns, term, re.IGNORECASE):
                 discovered_raw[term]['sources'].add("title_candidate")

    # 2. Extract from NER
    logger.info("Running NER on text content...")
    ner_hit_counter = Counter()
    # Consider processing in batches if corpus is huge
    doc = nlp(full_doc_text_for_extraction[:1000000]) # Limit if needed
    for ent in tqdm(doc.ents, desc="NER Processing"):
        label_info = NER_LABEL_MAP.get(ent.label_)
        if label_info:
            term_text = ent.text.strip()
            if 2 < len(term_text) < 80 and term_text.lower() not in PLANNING_STOPWORDS and not term_text.isdigit():
                raw_term = term_text.lower()
                discovered_raw[raw_term]['sources'].add("ner")
                if discovered_raw[raw_term]['type'] == 'Topic': # Prioritize NER type
                    discovered_raw[raw_term]['type'] = label_info['type']
                    discovered_raw[raw_term]['label'] = label_info['label']
                ner_hit_counter[raw_term] += 1

    # 3. Extract from KeyBERT
    logger.info("Running KeyBERT on text content...")
    keybert_keywords = kw_model.extract_keywords(full_doc_text_for_extraction,
                                                keyphrase_ngram_range=(1, MAX_KEYBERT_WORDS),
                                                stop_words='english', use_mmr=True, diversity=0.7,
                                                top_n=NUM_KEYBERT_KEYWORDS)
    for kw, score in keybert_keywords:
        if score >= MIN_KEYBERT_SCORE:
            term = kw.lower().strip()
            if 2 < len(term) < 80 and term not in PLANNING_STOPWORDS:
                discovered_raw[term]['sources'].add("keybert")

    # 4. Normalize, Map Synonyms, Count & Filter
    logger.info("Normalizing terms, mapping synonyms, counting occurrences...")
    canonical_concepts = defaultdict(lambda: {"type": "Topic", "label": "TopicConcept", "count": 0, "sources": set(), "original_terms": set()})
    raw_to_canonical_map = {}
    term_patterns_linking = {} # For faster searching

    # Build map and compile patterns
    for raw_term in discovered_raw.keys():
        canonical = get_canonical_form(raw_term, nlp_processor, acronym_map, synonym_map)
        raw_to_canonical_map[raw_term] = canonical
        if canonical not in term_patterns_linking:
             try: # Handle potential regex errors for complex terms
                 term_patterns_linking[canonical] = re.compile(r'\b' + re.escape(canonical) + r'\b', re.IGNORECASE)
             except re.error as re_err:
                 logger.warning(f"Could not compile regex for term '{canonical}': {re_err}")


    # Count occurrences of CANONICAL terms
    canonical_term_counts = Counter()
    for chunk_text in tqdm(all_text_corpus, desc="Counting canonical terms"):
        if not chunk_text: continue
        lower_text = chunk_text.lower()
        for canonical_term, pattern in term_patterns_linking.items():
             if pattern and pattern.search(lower_text): # Check pattern exists and search
                 canonical_term_counts[canonical_term] += 1

    # Consolidate into canonical_concepts
    logger.info("Consolidating counts and sources...")
    for raw_term, data in discovered_raw.items():
        canonical_term = raw_to_canonical_map.get(raw_term)
        if not canonical_term: continue # Skip if mapping failed

        count = canonical_term_counts.get(canonical_term, 0)
        if count >= MIN_TERM_FREQ:
            canonical_concepts[canonical_term]['original_terms'].add(raw_term)
            canonical_concepts[canonical_term]['count'] = count
            canonical_concepts[canonical_term]['sources'].update(data['sources'])
            if data['type'] != 'Topic' and canonical_concepts[canonical_term]['type'] == 'Topic':
                canonical_concepts[canonical_term]['type'] = data['type']
                canonical_concepts[canonical_term]['label'] = data['label']


    # 5. Final Filtering (POS, Subsumption on CANONICAL terms)
    final_concepts_filtered = {}
    logger.info(f"Applying final filters (POS, Subsumption) to {len(canonical_concepts)} canonical concepts...")
    sorted_canonical_terms = sorted(canonical_concepts.keys(), key=len, reverse=True)
    terms_to_remove_subsumed = set()

    for term in sorted_canonical_terms:
        data = canonical_concepts[term]
        count = data['count'] # Already filtered by freq

        # POS Pattern Check
        try:
            pos_tags = get_pos_tags(term, nlp_processor)
            if not is_valid_pos_pattern(pos_tags, VALID_POS_PATTERNS):
                # logger.debug(f"Filtering term '{term}' due to POS pattern: {pos_tags}")
                continue
        except Exception as e:
             logger.warning(f"Could not get POS tags for '{term}': {e}")

        # Subsumption Check
        is_subsumed = False
        for longer_term in final_concepts_filtered:
            if term != longer_term and re.search(r'\b' + re.escape(term) + r'\b', longer_term):
                 if count < final_concepts_filtered[longer_term]['count'] * SUBSUMPTION_COUNT_MULTIPLIER:
                      terms_to_remove_subsumed.add(term)
                      is_subsumed = True; break

        if not is_subsumed:
             data['sources'] = list(data['sources'])
             data['original_terms'] = list(data['original_terms'])
             final_concepts_filtered[term] = data

    logger.info(f"Removing {len(terms_to_remove_subsumed)} subsumed canonical terms.")
    for term in terms_to_remove_subsumed:
         final_concepts_filtered.pop(term, None)

    # 6. Save and Load Reviewed
    logger.info(f"Discovered {len(final_concepts_filtered)} final concepts after filtering.")
    logger.info(f"Saving candidate concepts to {DISCOVERED_CONCEPTS_OUTPUT_PATH}")
    try:
        sorted_concepts = dict(sorted(final_concepts_filtered.items(), key=lambda item: item[1]['count'], reverse=True))
        with open(DISCOVERED_CONCEPTS_OUTPUT_PATH, "w", encoding='utf-8') as f:
            json.dump(sorted_concepts, f, indent=2, ensure_ascii=False)
        logger.info("Candidates saved. Please review/edit this file.")
        logger.info(f"If reviewed, save as '{MANUALLY_REVIEWED_CONCEPTS_PATH}' to use in linking.")
    except Exception as e:
        logger.error(f"Failed to save discovered concepts: {e}")

    concepts_for_linking = final_concepts_filtered # Default
    if os.path.exists(MANUALLY_REVIEWED_CONCEPTS_PATH):
         logger.warning(f"Found manually reviewed concepts file: {MANUALLY_REVIEWED_CONCEPTS_PATH}. USING THIS INSTEAD.")
         try:
             with open(MANUALLY_REVIEWED_CONCEPTS_PATH, "r", encoding='utf-8') as f:
                 concepts_for_linking = json.load(f)
             logger.info(f"Loaded {len(concepts_for_linking)} reviewed concepts.")
         except Exception as e:
             logger.error(f"Failed to load reviewed concepts file, using auto-discovered: {e}")
             concepts_for_linking = final_concepts_filtered
    else:
        logger.warning("No manually reviewed concepts file found. Proceeding with auto-discovered/filtered concepts.")

    return concepts_for_linking


def link_discovered_concepts_batch(tx: ManagedTransaction, concepts_to_link):
    """Links discovered concepts to chunks in batches, handling dynamic labels."""
    # concepts_to_link is list: [{"chunkId": cid, "conceptName": cname, "conceptLabel": clabel, "conceptType": ctype}, ...]
    logger.debug(f"Linking batch of {len(concepts_to_link)} concept references.")
    grouped_by_label = defaultdict(list)
    for item in concepts_to_link:
        # Basic sanitization for labels (Neo4j doesn't like spaces or special chars)
        # Ensure label starts with uppercase and contains only alphanumerics or underscore
        sanitized_label = re.sub(r'^[^A-Za-z_]+|[^A-Za-z0-9_]', '', item['conceptLabel'])
        if not sanitized_label or not sanitized_label[0].isalpha():
             sanitized_label = "Concept" # Default fallback label
        elif not sanitized_label[0].isupper():
             sanitized_label = sanitized_label[0].upper() + sanitized_label[1:]

        item['sanitizedLabel'] = sanitized_label # Add sanitized label to the item
        grouped_by_label[sanitized_label].append(item)

    for label, batch_for_label in grouped_by_label.items():
        # Use parameters for node properties and relationship properties
        label_safe_query = f"""
        UNWIND $concepts AS link_data
        MATCH (dc:DocumentChunk {{chunkId: link_data.chunkId}})
        // Merge concept using the sanitized label
        MERGE (c:Concept:`{label}` {{name: link_data.conceptName}})
        ON CREATE SET c.type = link_data.conceptType, c.discoveredFrom = link_data.docId // Add type and source doc on creation
        ON MATCH SET c.type = link_data.conceptType // Update type if needed
        // Merge relationship
        MERGE (dc)-[:REFERENCES_CONCEPT]->(c)
        """
        try:
            # Pass only necessary data for the batch
            batch_payload = [{"chunkId": item["chunkId"], "conceptName": item["conceptName"], "conceptType": item["conceptType"], "docId": item["docId"]} for item in batch_for_label]
            tx.run(label_safe_query, concepts=batch_payload)
        except Exception as e:
            logger.error(f"Error linking concepts with label '{label}': {e}", exc_info=True)


# --- Main Ingestion Logic ---
if __name__ == "__main__":
    main_start_time = time.time()
    all_processed_chunks = []
    all_docs_metadata = []

    # --- Phase 1: Loading and Flattening Documents ---
    logger.info("--- Phase 1: Loading and Flattening Documents ---")
    for doc_info in DOCUMENTS_TO_PROCESS:
        logger.info(f"Processing Document: {doc_info['doc_id']}")
        nested_data = load_json(doc_info['nested_json_path'])
        ontology_map = load_json(doc_info['ontology_map_path'])
        if not nested_data or not ontology_map:
            logger.warning(f"Skipping document {doc_info['doc_id']} due to loading errors.")
            continue

        doc_metadata = {
            "doc_id": doc_info['doc_id'], "title": doc_info.get('title', doc_info['doc_id']),
            "type": doc_info.get('type', 'Unknown'), "versionDate": doc_info.get('versionDate', '1970-01-01')
        }
        all_docs_metadata.append(doc_metadata)
        try:
            # Find the root content node correctly
            root_node = nested_data # Assume root is the dict itself if no 'sections'
            if 'sections' in nested_data and isinstance(nested_data['sections'], list) and len(nested_data['sections']) > 0:
                 root_node = nested_data['sections'][0]

            doc_chunks = []
            # Check if root_node itself should be a chunk or just contains subsections
            # Current flatten assumes subsections contain the main content chunks
            if 'subsections' in root_node and root_node['subsections']:
                for i, top_level_chunk_data in enumerate(root_node['subsections']):
                    doc_chunks.extend(flatten_nested_json(top_level_chunk_data, doc_metadata['doc_id'], ontology_map, sequence=i))
            else:
                 logger.warning(f"No 'subsections' found in root node for {doc_info['doc_id']}. Check JSON structure.")

            logger.info(f"Generated {len(doc_chunks)} chunks for {doc_info['doc_id']}.")
            all_processed_chunks.extend(doc_chunks)
        except Exception as e:
            logger.error(f"Error parsing nested structure for {doc_info['doc_id']}: {e}", exc_info=True)
            continue

    if not all_processed_chunks:
        logger.error("No chunks processed from any document. Exiting.")
        exit()
    logger.info(f"Total chunks processed across all documents: {len(all_processed_chunks)}")

    # --- Phase 2: Qdrant Ingestion ---
    logger.info("\n--- Phase 2: Ingesting to Qdrant ---")
    ensure_qdrant_collection(qdrant_client, QDRANT_COLLECTION_NAME, VECTOR_SIZE)
    ingest_to_qdrant(qdrant_client, QDRANT_COLLECTION_NAME, all_processed_chunks)

    # --- Phase 3: Concept Discovery & Synonym Mapping ---
    logger.info("\n--- Phase 3: Discovering Concepts & Mapping Synonyms ---")
    corpus_texts = [c['text'] for c in all_processed_chunks if c.get('text')]
    detected_acronyms = detect_acronyms(corpus_texts)
    manual_synonyms = load_synonyms(SYNONYM_MAP_PATH)
    concepts_for_linking = discover_concepts(all_processed_chunks, nlp, detected_acronyms, manual_synonyms)

    # --- Phase 4: Neo4j Ingestion ---
    logger.info("\n--- Phase 4: Ingesting to Neo4j ---")
    with neo4j_driver.session(database="neo4j") as session:
        # Optional: Clear DB
        # logger.warning("CLEARING Neo4j database enabled!")
        # session.execute_write(clear_neo4j_database)

        # Constraints
        session.execute_write(create_neo4j_constraints)

        # Load Documents
        logger.info("Loading Document nodes...")
        for meta in tqdm(all_docs_metadata, desc="Loading Documents"): session.execute_write(load_neo4j_document, meta)

        # Load Chunks
        logger.info(f"Loading {len(all_processed_chunks)} chunks...")
        for i in tqdm(range(0, len(all_processed_chunks), NEO4J_BATCH_SIZE), desc="Loading Chunks"): session.execute_write(load_neo4j_chunks_batch, all_processed_chunks[i:i + NEO4J_BATCH_SIZE])

        # Create Hierarchy
        logger.info("Creating hierarchy relationships...")
        hierarchy_links = [{"docId": c["docId"], "chunkId": c["chunkId"], "parentId": c.get("parentChunkId"), "order": c.get("sequenceInParent", 0)} for c in all_processed_chunks]
        for i in tqdm(range(0, len(hierarchy_links), NEO4J_BATCH_SIZE), desc="Linking Hierarchy"): session.execute_write(create_neo4j_hierarchy_batch, hierarchy_links[i:i + NEO4J_BATCH_SIZE])

        # Create Supports Relationships
        logger.info("Creating SUPPORTS relationships...")
        supports_links = []; chunk_map = {c['chunkId']: c for c in all_processed_chunks}
        for chunk in all_processed_chunks:
            if chunk.get("parentChunkId"):
                parent_chunk = chunk_map.get(chunk.get("parentChunkId"))
                if parent_chunk and parent_chunk.get("policyId") and chunk['contentType'].startswith("SupportingText"):
                     supports_links.append({"textId": chunk["chunkId"], "policyId": parent_chunk["chunkId"]})
        for i in tqdm(range(0, len(supports_links), NEO4J_BATCH_SIZE), desc="Linking SUPPORTS"): session.execute_write(create_neo4j_supports_rels_batch, supports_links[i:i + NEO4J_BATCH_SIZE])

        # --- Link Discovered Concepts ---
        logger.info("Linking concepts to chunks...")
        concepts_to_link_batch = []
        # Use the final concepts (potentially reviewed)
        term_patterns_linking = {term: re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
                                 for term in concepts_for_linking.keys()}

        for chunk in tqdm(all_processed_chunks, desc="Linking Concepts"):
            if chunk['text']:
                chunk_text_lower = chunk['text'].lower()
                for concept_name, pattern in term_patterns_linking.items():
                    if pattern and pattern.search(chunk_text_lower): # Ensure pattern compiled
                        concept_data = concepts_for_linking[concept_name]
                        concepts_to_link_batch.append({
                             "chunkId": chunk["chunkId"],
                             "conceptName": concept_name,
                             "conceptLabel": concept_data.get("label", "Concept"), # Use label from data
                             "conceptType": concept_data.get("type", "Topic"),   # Use type from data
                             "docId": chunk["docId"] # Pass docId for potential concept property
                        })
                        if len(concepts_to_link_batch) >= CONCEPT_LINKING_BATCH_SIZE:
                             session.execute_write(link_discovered_concepts_batch, concepts_to_link_batch)
                             concepts_to_link_batch = []
        # Process remaining
        if concepts_to_link_batch:
            session.execute_write(link_discovered_concepts_batch, concepts_to_link_batch)
        logger.info("Concept linking complete.")

        # --- Process Glossary and Policy References ---
        logger.info("Processing Glossary definitions and linking mentions/policy refs...")
        glossary_terms = []
        all_glossary_chunks_for_defs = [c for c in all_processed_chunks if c.get('contentType', '').endswith('Glossary/Definitions')]
        if all_glossary_chunks_for_defs:
             glossary_terms = session.execute_read(process_glossary, all_glossary_chunks_for_defs)

        linking_tasks = [{'chunkId': c['chunkId'], 'text': c['text'], 'docId': c['docId']} for c in all_processed_chunks if c.get('text')]
        batch_size = 1000
        for i in tqdm(range(0, len(linking_tasks), batch_size), desc="Linking Policy Refs & Glossary"):
            batch_tasks = linking_tasks[i : i + batch_size]
            def linking_tx_func(tx, tasks, terms):
                for task in tasks:
                    try:
                        extract_and_link_policy_refs(tx, task['chunkId'], task['text'], task['docId'])
                        if terms:
                            link_glossary_mentions(tx, task['chunkId'], task['text'], terms)
                    except Exception as task_e:
                        logger.error(f"Error in linking task for chunk {task['chunkId']}: {task_e}", exc_info=False) # Log error but continue batch
            try:
                session.execute_write(linking_tx_func, batch_tasks, glossary_terms)
            except Exception as batch_e:
                logger.error(f"Error processing linking batch {i // batch_size}: {batch_e}", exc_info=True)


    logger.info("Neo4j ingestion complete.")
    neo4j_driver.close()
    end_time = time.time()
    logger.info(f"Script finished in {end_time - main_start_time:.2f} seconds.")