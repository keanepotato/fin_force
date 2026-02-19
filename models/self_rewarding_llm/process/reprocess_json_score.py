import json
import os
import re

input_path = "/home/finaxai/llm_scenario_modelling/models/self_rewarding_llm/oxen_implementation/data/N0/generated/scores.jsonl"
output_path = "/home/finaxai/llm_scenario_modelling/models/self_rewarding_llm/oxen_implementation/data/N0/generated/parsed_scores.jsonl"

valid_objects = []
with open(input_path, "r", encoding='utf-8') as file:
    content = file.read()
    # Try to find all JSON-like objects in the file
    # Look for patterns that start with { and end with }
    json_pattern = r'(\{.*?\})(?=\s*\{|\s*$)'
    potential_jsons = re.findall(json_pattern, content, re.DOTALL)
    
    for i, potential_json in enumerate(potential_jsons):
        try:
            json_obj = json.loads(potential_json)
            valid_objects.append(json_obj)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON object {i+1}: {e}")
            print(f"Problematic content: {potential_json[:100]}...")

print(f"Successfully parsed {len(valid_objects)} JSON objects")

# Write the valid objects back to a proper JSONL file
with open(output_path, "w", encoding='utf-8') as outfile:
    for obj in valid_objects:
        # Write each object as a separate line in JSONL format
        outfile.write(json.dumps(obj) + "\n")

print(f"Fixed JSONL data saved to {output_path}")