import json
import os

# --- Configuration ---
input_filename = 'f985ddd124245046f2e5c67c656bba2e_TH_Local_Plan_2024_UPDATE_(ONLINE-HIGH_RES)_flat.jsonl'
output_prefix = 'tower_hamlets_split'
output_filenames = {
    1: f"{output_prefix}_group1_intro.jsonl",
    2: f"{output_prefix}_group2_policies.jsonl",
    3: f"{output_prefix}_group3_sites_appendices.jsonl",
}

# Define the headings for each group (using sets for efficient lookup)
group1_headings = {
    "Mayor’s foreword",
    "Introduction",
    "1. What is the Local Plan?",
    "2. Why is a new Local Plan being produced?",
    "3. What are the statutory requirements in developing a Local Plan?",
    "4. What informs the development of the Local Plan?",
    "5. Consultation and engagement",
    "6. How to get involved",
    "7. Next steps",
    "Setting the Scene",
    "8. Setting the scene",
    "Vision and objectives",
    "9. Our vision for Tower Hamlets",
    "10. Key objectives and principles",
}

group2_headings = {
    "Policies",
    "11. Delivering the Local Plan",
    "12. Homes for the community",
    "13. Clean and green future",
    "14. People, places and spaces",
    "15. Inclusive economy and good growth",
    "16. Town centres",
    "17. Community infrastructure",
    "18. Biodiversity and open space",
    "19. Movement and connectivity",
    "20. Reuse, recycling and waste",
}

group3_headings = {
    "Site Allocations",
    "21. Introduction",
    "22. City Fringe",
    "23. Central",
    "24. Leaside",
    "25. Isle of Dogs and South Poplar",
    "Appendices",
    "Appendix 1: Glossary",
    "Appendix 2: Financial contribution calculation methodologies",
    "Appendix 3: Links to the Tower Hamlets Local Plan 2031 (Managing growth and sharing the benefits)",
    "Appendix 4: Key monitoring indicators",
    "Appendix 5: Noise",
    "Appendix 6: Air Quality",
    "Appendix 7: List of Policies by Strategic/non-Strategic",
}
# --- End Configuration ---

# --- Main Script Logic ---
def split_jsonl(input_file, output_map, headings_map):
    """
    Splits a JSONL file into multiple files based on headings.

    Args:
        input_file (str): Path to the input JSONL file.
        output_map (dict): Dictionary mapping group number to output filename.
        headings_map (dict): Dictionary mapping group number to a set of headings.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    output_files = {}
    writers = {}
    try:
        # Open output files
        for group_num, filename in output_map.items():
            output_files[group_num] = open(filename, 'w', encoding='utf-8')
            print(f"Opened {filename} for writing.")

        current_heading = None
        unmatched_headings = set()
        processed_lines = 0
        group_counts = {1: 0, 2: 0, 3: 0}
        unmatched_count = 0

        # Read input file and write to output files
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    chunk_data = json.loads(line)
                    processed_lines += 1
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON on line {line_num}: {line[:100]}...")
                    continue

                # Determine the effective heading for this chunk
                chunk_heading = chunk_data.get('heading', '').strip()

                # Update current_heading if a new heading is found in the chunk
                if chunk_heading:
                    current_heading = chunk_heading
                
                # Use the chunk's own heading if present, otherwise the last seen heading
                effective_heading = chunk_heading if chunk_heading else current_heading

                # Determine which group the chunk belongs to
                assigned_group = None
                if effective_heading in headings_map[1]:
                    assigned_group = 1
                elif effective_heading in headings_map[2]:
                    assigned_group = 2
                elif effective_heading in headings_map[3]:
                    assigned_group = 3
                elif effective_heading is None and chunk_heading == '': # Handle chunks at the very start before any heading
                     # Decide how to handle chunks before the first heading.
                     # Option 1: Assign to group 1 (assuming intro)
                     # assigned_group = 1
                     # Option 2: Skip/Log (current implementation implicitly skips)
                     # print(f"Warning: Skipping line {line_num} with no preceding heading.")
                     pass

                # Write to the corresponding output file
                if assigned_group:
                    output_files[assigned_group].write(line + '\n')
                    group_counts[assigned_group] += 1
                elif effective_heading: # Log headings that didn't match any group
                    unmatched_headings.add(effective_heading)
                    unmatched_count += 1
                else: # Chunk has no heading and no preceding heading seen yet
                     unmatched_count += 1


        print("\n--- Processing Summary ---")
        print(f"Total lines processed: {processed_lines}")
        for group, count in group_counts.items():
             print(f"Lines written to Group {group} ({output_map[group]}): {count}")
        if unmatched_count > 0:
             print(f"Lines not matched to any group: {unmatched_count}")
        if unmatched_headings:
            print("\nUnmatched Headings Found:")
            for h in sorted(list(unmatched_headings)):
                print(f" - '{h}'")

    except IOError as e:
        print(f"An error occurred during file operations: {e}")
    finally:
        # Close all output files
        for f in output_files.values():
            if f and not f.closed:
                f.close()
        print("\nAll output files closed.")

# --- Run the script ---
if __name__ == "__main__":
    all_headings_map = {
        1: group1_headings,
        2: group2_headings,
        3: group3_headings,
    }
    print(f"Starting split of '{input_filename}'...")
    split_jsonl(input_filename, output_filenames, all_headings_map)
    print("Script finished.")