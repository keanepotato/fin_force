import os
import json
import yaml

def flatten_response(response_data):
    """
    Extract and format scenarios from the response data.
    
    Args:
        response_data (dict): A dictionary with output/scenarios structure
                              
    Returns:
        tuple: (prompt_text, completion_text) where prompt_text is the original headline
               and completion_text combines both opportunity and risk scenarios
    """
    # Check if we have the expected structure
    if "output" not in response_data or "scenarios" not in response_data["output"]:
        return None, "No valid scenario data found."
    
    scenarios = response_data["output"]["scenarios"][0]
    
    # Extract the original headline (prompt)
    original_headline = scenarios.get("original_headline", "No original headline available")
    
    # Extract both scenarios
    opportunity_scenario = scenarios.get("opportunity_counterfactual_scenario", "No opportunity scenario available")
    risk_scenario = scenarios.get("risk_counterfactual_scenario", "No risk scenario available")
    
    # Combine both scenarios into a formatted completion
    completion_text = f"Opportunity Scenario:\n{opportunity_scenario}\n\nRisk Scenario:\n{risk_scenario}"
    
    return original_headline, completion_text

def main(config_path="/home/llm_scenario_modelling/baseline_auto_align/self_rewarding_llm/process/configs/process_scenario_data.yaml"):
    # Load configuration from the YAML file
    with open(config_path, "r", encoding="utf8") as f:
        config = yaml.safe_load(f)
    
    # Retrieve configuration parameters
    prompt_file = config.get("prompt_file")
    json_input_file = config.get("json_input_file")  # Path to the single JSON file with all items
    output_file = config.get("output_file")
    source = config.get("source", "source_gpt4omini")
    
    # Read the static prompt instruction if the prompt_file is specified
    prompt_instruction = ""
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, "r", encoding="utf8") as pf:
            prompt_instruction = pf.read().strip()
    else:
        print("No prompt file specified or file does not exist. Skipping prompt instruction.")

    # Read the input JSON file
    with open(json_input_file, "r", encoding="utf8") as f:
        all_items = json.load(f)
    
    with open(output_file, "w", encoding="utf8") as out_f:
        for item in all_items:
            # Extract the item_index to use as prompt_id
            prompt_id = item.get("item_index")
            
            if "output" not in item:
                print(f"Warning: No output found for item {prompt_id}")
                continue
                
            # Extract the original headline and scenarios
            original_headline, completion_text = flatten_response(item)
            
            if original_headline is None:
                print(f"Warning: Could not extract headline/scenarios for item {prompt_id}")
                continue
                
            # Build the output dictionary
            output_item = {
                "prompt_id": str(prompt_id),  # Using item_index as the prompt_id
                "prompt": original_headline,
                "completion": completion_text,
                "source": source
            }
            
            # Write the JSON object as a new line in the output file
            out_f.write(json.dumps(output_item) + "\n")
    
    print(f"Output written to {output_file}")

if __name__ == "__main__":
    main()
