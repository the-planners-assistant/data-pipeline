import json
import os

def count_nodes(node):
    """Recursively counts nodes in a nested structure."""
    # Count the current node itself
    count = 1
    # If the node has subsections, recursively count them
    if 'subsections' in node and isinstance(node['subsections'], list):
        for subsection in node['subsections']:
            count += count_nodes(subsection) # Add count of child and its descendants
    return count

# --- Configuration ---
NESTED_JSON_PATH = "corpus/the_london_plan_2021_nested.json" # Adjust path if needed

# --- Main Logic ---
total_nodes = 0
if not os.path.exists(NESTED_JSON_PATH):
    print(f"Error: File not found at {NESTED_JSON_PATH}")
else:
    try:
        with open(NESTED_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Assuming the structure is { "doc_id": ..., "sections": [ root_node ] }
        if 'sections' in data and isinstance(data['sections'], list) and len(data['sections']) > 0:
            # Start counting from the root node within the 'sections' array
            root_content_node = data['sections'][0]
            total_nodes = count_nodes(root_content_node)
            print(f"Successfully counted nodes starting from root: '{root_content_node.get('heading', 'N/A')}'.")
        else:
            print("Unexpected JSON structure: 'sections' key missing, not a list, or empty.")

        print(f"\nThe JSON file contains a total of {total_nodes} nodes (including the main document root and all subsections).")

    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {NESTED_JSON_PATH}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")