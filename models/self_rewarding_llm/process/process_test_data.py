import os
import json
import glob
import yaml

def main(config_path="/home/finaxai/llm_scenario_modelling/models/self_rewarding_llm/oxen_implementation/process/process_test_data.yaml"):
    # Load configuration from the YAML file
    with open(config_path, "r", encoding="utf8") as f:
        config = yaml.safe_load(f)
    
    # Retrieve configuration parameters
    prompt_file = config.get("prompt_file")
    input_dir = config.get("input_dir")
    output_file = config.get("output_file")
    source = config.get("source", "source_gpt4omini")
    
    # Read the static prompt instruction if the prompt_file is specified
    prompt_instruction = ""
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, "r", encoding="utf8") as pf:
            prompt_instruction = pf.read().strip()
    else:
        print("No prompt file specified or file does not exist. Skipping prompt instruction.")

    # Get all input text files (assumed to have a .txt extension)
    input_files = glob.glob(os.path.join(input_dir, "*.txt"))
    
    with open(output_file, "w", encoding="utf8") as out_f:
        i = 0
        for input_path in input_files:
            # Extract prompt_id from the filename (e.g., "2021-01-01" from "2021-01-01.txt")
            base_name = os.path.basename(input_path)
            prompt_id, _ = os.path.splitext(base_name)
            
            # Read the input text
            with open(input_path, "r", encoding="utf8") as inp_f:
                input_text = inp_f.read().strip()
            
            # Construct the full prompt: static instruction + input text (if prompt_instruction is available)
            if prompt_instruction:
                full_prompt = f"{prompt_instruction}\n{input_text}"
            else:
                full_prompt = input_text
            
            # Build the output dictionary without including any completion/response details
            output_item = {
                "prompt_id": str(i),  # Keeping the prompt id as an index
                "prompt": full_prompt,
                "source": source
            }
            
            # Write the JSON object as a new line in the output file
            out_f.write(json.dumps(output_item) + "\n")
            i += 1
    
    print(f"Output written to {output_file}")

if __name__ == "__main__":
    main()