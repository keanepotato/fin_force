import json

# Load your original JSON file
with open('/home/llm_scenario_modelling/baseline_prompting/results/distil_cf/distil_cf_test.json', 'r') as f:
    data = json.load(f)

# Process each item in the list
for item in data:
    original_headline = item['input_item']['headline']
    item['output']['original_headline'] = original_headline

# Save the updated JSON to a new file
with open('/home/llm_scenario_modelling/baseline_prompting/results/distil_cf/processed_distil_cf_test.json', 'w') as f:
    json.dump(data, f, indent=2)