import json


# Replace with your file paths
input_file = "/home/finaxai/llm_scenario_modelling/models/self_rewarding_llm/oxen_implementation/data/D0/generated/generated_ift.jsonl"
output_file = "/home/finaxai/llm_scenario_modelling/models/self_rewarding_llm/oxen_implementation/data/D0/generated/formatted_responses.jsonl"

with open(input_file, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f]  # Load each line as a separate JSON object

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(lines, f, indent=4, ensure_ascii=False)  # Save as a properly formatted list

print(f"Formatted JSON saved to {output_file}")

