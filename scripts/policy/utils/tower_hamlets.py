import json
import os

# --- Configuration ---
INPUT_FILENAME = 'f985ddd124245046f2e5c67c656bba2e_TH_Local_Plan_2024_UPDATE_(ONLINE-HIGH_RES)_flat.jsonl'
OUTPUT_FILENAME_1 = 'group1_intro_vision.jsonl'
OUTPUT_FILENAME_2 = 'group2_policies.jsonl'
OUTPUT_FILENAME_3 = 'group3_sites_appendices.jsonl'

# --- Define Heading Groups (using sets for efficient lookup) ---

GROUP1_HEADINGS = {
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
  # Include empty string to catch initial chunks like title page, ToC
  "",
}

GROUP2_HEADINGS = {
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

GROUP3_HEADINGS = {
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

# --- Main Script Logic ---

def split_jsonl_by_heading(input_file, out_file1, out_file2, out_file3):
    """
    Reads a JSONL file and splits its lines into three output files
    based on the 'heading' field.

    Args:
        input_file (str): Path to the input JSONL file.
        out_file1 (str): Path for the first output JSONL file.
        out_file2 (str): Path for the second output JSONL file.
        out_file3 (str): Path for the third output JSONL file.
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    current_group = 1  # Default to group 1 for initial chunks before a heading match
    line_count = 0
    group_counts = {1: 0, 2: 0, 3: 0}

    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(out_file1, 'w', encoding='utf-8') as outfile1, \
             open(out_file2, 'w', encoding='utf-8') as outfile2, \
             open(out_file3, 'w', encoding='utf-8') as outfile3:

            files = {1: outfile1, 2: outfile2, 3: outfile3}

            for line in infile:
                line = line.strip()
                line_count += 1
                if not line:
                    continue # Skip empty lines

                try:
                    data = json.loads(line)
                    # Get heading, default to empty string if missing, strip whitespace
                    heading = data.get('heading', '').strip()

                    # Determine the group *only if* the heading matches a group explicitly
                    # Otherwise, keep the current_group
                    if heading in GROUP1_HEADINGS:
                        current_group = 1
                    elif heading in GROUP2_HEADINGS:
                        current_group = 2
                    elif heading in GROUP3_HEADINGS:
                        current_group = 3
                    # Chunks with non-matching headings (sub-headings, empty ones after
                    # the first sections) will inherit the group from the previous chunk.

                    # Write the original JSON line to the correct output file
                    if current_group in files:
                        files[current_group].write(line + '\n')
                        group_counts[current_group] += 1
                    else:
                        # This should theoretically not happen with the default
                        print(f"Warning: Line {line_count} could not be assigned to a group. Heading: '{heading}'")

                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON on line {line_count}: {line[:100]}...")
                except Exception as e:
                     print(f"Warning: Error processing line {line_count}: {e} - Line: {line[:100]}...")


        print("\n--- Processing Complete ---")
        print(f"Total lines processed: {line_count}")
        print(f"Lines written to '{out_file1}': {group_counts[1]}")
        print(f"Lines written to '{out_file2}': {group_counts[2]}")
        print(f"Lines written to '{out_file3}': {group_counts[3]}")
        total_written = sum(group_counts.values())
        print(f"Total lines written: {total_written}")
        if line_count - infile.read().count('\n\n') > total_written: # Adjust for potential blank lines skipped
             print(f"Note: Some lines might have been skipped due to errors or being blank.")

    except IOError as e:
        print(f"Error opening or writing file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Run the script ---
if __name__ == "__main__":
    print(f"Starting script to split '{INPUT_FILENAME}'...")
    split_jsonl_by_heading(INPUT_FILENAME, OUTPUT_FILENAME_1, OUTPUT_FILENAME_2, OUTPUT_FILENAME_3)
    print("Script finished.")