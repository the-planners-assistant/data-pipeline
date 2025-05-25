import os
import json

BASE_DIR = "../../corpus"

# Handle subdirectories (LPA-specific)
for entry in os.listdir(BASE_DIR):
    entry_path = os.path.join(BASE_DIR, entry)

    # Handle LPA folders like hammersmith_and_fulham/
    if os.path.isdir(entry_path):
        policies_path = os.path.join(entry_path, "policies.json")
        page_map_path = os.path.join(entry_path, "page_numbers.json")
    else:
        # Handle top-level files like london_plan.json
        if not entry.endswith(".json") or "_page_numbers" in entry:
            continue

        base_name = entry[:-5]  # Strip ".json"
        policies_path = os.path.join(BASE_DIR, entry)
        page_map_path = os.path.join(BASE_DIR, f"{base_name}_page_numbers.json")

    if not (os.path.exists(policies_path) and os.path.exists(page_map_path)):
        print(f"⚠️  Skipping {policies_path} – missing page map or policy file")
        continue

    # Load data
    with open(policies_path) as f:
        policies = json.load(f)

    with open(page_map_path) as f:
        page_map = json.load(f)

    # Merge page numbers
    missing_id_count = 0
    for policy in policies:
        policy_id = policy.get("policy_id")
        if not policy_id:
            print(f"⚠️  Missing policy_id in entry:\n{json.dumps(policy, indent=2)}")
            missing_id_count += 1
            continue
        policy["page_number"] = page_map.get(policy_id)

    # Overwrite
    with open(policies_path, "w") as f:
        json.dump(policies, f, indent=2)

    print(f"✅ Enriched {entry} with page numbers")
    if missing_id_count > 0:
        print(f"⚠️  {missing_id_count} entries in {entry} were missing policy_id")
