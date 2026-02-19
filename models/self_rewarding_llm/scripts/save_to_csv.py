import json
import csv
def convert_jsonl_to_csv(input_file, output_file):
    """Convert a JSONL preference file to CSV format."""
    
    with open(input_file, 'r', encoding='utf-8') as jsonl_file, \
         open(output_file, 'w', encoding='utf-8', newline='') as csv_file:
        
        # Create CSV writer
        csv_writer = csv.writer(csv_file)
        
        # Write header
        csv_writer.writerow(['Prompt', 'Chosen Response', 'Rejected Response'])
        
        # Process each line in the JSONL file
        for line_num, line in enumerate(jsonl_file, 1):
            try:
                data = json.loads(line.strip())
                
                # Extract data - adjust these keys based on your actual JSONL structure
                prompt = data.get('prompt', '')
                chosen = data.get('chosen', '')
                rejected = data.get('rejected', '')
                
                # Write to CSV
                csv_writer.writerow([prompt, chosen, rejected])
                
            except json.JSONDecodeError:
                print(f"Error parsing JSON on line {line_num}. Skipping.")
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
    
    print(f"Conversion complete. Data saved to {output_file}")

if __name__ == "__main__":
    input_file = '/home/llm_scenario_modelling/baseline_models/self_rewarding_llm/data/CF1/generated/preferences.jsonl'
    output_file = '/home/llm_scenario_modelling/baseline_models/self_rewarding_llm/data/CF1/generated/preferences.csv'
    
    convert_jsonl_to_csv(input_file, output_file)