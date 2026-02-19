import os
import glob
import json
import yaml
from tqdm import tqdm
from openai import OpenAI
from pydantic import BaseModel
import time
import instructor
import argparse
from typing import Type
from response_config import *

MODEL_CLASS_MAP = {
    "Counterfactuals_COT": Counterfactuals_COT,
    "Counterfactuals": Counterfactuals,
    "Counterfactual_masked": Counterfactual_masked,
}

def get_model_class(class_name: str) -> Type[BaseModel]:
    """Get the model class by name from the MODEL_CLASS_MAP."""
    if class_name not in MODEL_CLASS_MAP:
        raise ValueError(f"Unknown model class: {class_name}. Available classes: {list(MODEL_CLASS_MAP.keys())}")
    return MODEL_CLASS_MAP[class_name]

def openrouter_inference(messages=None, model=None, api_key=None, response_model=None):    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    client = instructor.from_openai(client, mode=instructor.Mode.TOOLS)

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        response_model=response_model,
        extra_body={"provider": {"require_parameters": True}},
    )

    # Convert the completion object to a dictionary with the needed fields
    response = {
        "original_headline": completion.original_headline,
        "opportunity_counterfactual_scenario": completion.opportunity_counterfactual,
        "risk_counterfactual_scenario": completion.risk_counterfactual
    }
    
    return response

def openai_inference(messages=None, model=None, api_key=None, response_model=None):

    # Initialize OpenAI client with the provided API key
    client = OpenAI(api_key=api_key)

    response = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=response_model,
    )

    # Parse andCounterfactuals
    response = response_model.model_validate(response.choices[0].message.parsed)
    response = response.dict()
    return response

def load_config_values(config_file):
    """
    Load configuration values from a YAML file and return them directly.
    
    Args:
        config_file (str): Path to the configuration YAML file
        
    Returns:
        tuple: Contains (api_key, model, system_role, json_file_path, output_results_file)
    """
    # Load configuration from the YAML file
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Load system role instructions from file
    system_role_path = config.get("system_role_file", "system_role.txt")
    with open(system_role_path, "r") as f:
        system_role = f.read()

    # PRINT THE PROMPT USED
    print(f"System role loaded from {system_role_path}:")
    print("-------------------------PROMPT-------------------------")
    print(system_role)
    print("---------------------------------------------------------")
    # SLEEP for 2 seconds to read the output
    time.sleep(2)

    # Load the few shot examples from the file
    few_shot_path = config.get("few_shot_file", None)
    if few_shot_path:
        with open(few_shot_path, "r") as f:
            few_shot_examples = f.read()
        print(f"Few-shot examples loaded from {few_shot_path}:")
        print("-------------------------USING FEW SHOT EXAMPLES-------------------------")
        print(few_shot_examples)
        print("---------------------------------------------------------")
        # SLEEP for 2 seconds to read the output
        time.sleep(2)
    else:
        few_shot_examples = None
        print("No few-shot examples provided.")
        print("---------------------------------------------------------")

    # LOADING CONFIG FILE

    model_response_class_name = config.get("response_model", None)
    
    if not model_response_class_name:
        raise ValueError("Response model response class not found in configuration.")
    
    response_model = get_model_class(model_response_class_name)
    print(f"Using response model: {model_response_class_name}")
    
    model = config.get("model")
    json_input_file_path = config.get("input_file")
    output_results_file = config.get("output_file")
    input_key = config.get("input_key")
    openrouter_api_key = config.get("openrouter_api_key")
    openai_api_key = config.get("openai_api_key")
    
    return model, system_role, json_input_file_path, output_results_file, input_key, openrouter_api_key, openai_api_key, few_shot_examples, response_model

def main(config_file=None):
    
    model, system_role, json_input_file_path, output_results_file, input_key, openrouter_api_key, openai_api_key, few_shot_examples, response_model = load_config_values(config_file)
    
    """Process a JSON file with multiple entries and save results in consolidated files."""
    
    if not os.path.exists(json_input_file_path):
        print(f"JSON input file not found: {json_input_file_path}")
        return False

    try:
        # Load the JSON data
        with open(json_input_file_path, 'r') as f:
            items = json.load(f)
            
        print(f"Found {len(items)} items to process")
        
        # Initialize output files if they don't exist
        if not os.path.exists(output_results_file):
            os.makedirs(os.path.dirname(output_results_file), exist_ok=True)
            with open(output_results_file, 'w') as f:
                json.dump([], f)
        
        # Load existing results to append to
        with open(output_results_file, 'r') as f:
            results = json.load(f)
        
        # Process each item
        for i, item in enumerate(tqdm(items, desc="Processing JSON items")):
            content = item[input_key]
            print(f"Processing content: {content}")

            if few_shot_examples:
                content = few_shot_examples.format(input=content)
            
            # Construct message for API
            messages = [
                {"role": "system", "content": system_role},
                {"role": "user", "content": content}
            ]
            
            try:
                max_retries = 3
                retry_count = 0
                success = False
                
                while retry_count < max_retries and not success:
                    try:
                        if model in ['gpt-4o', 'gpt-4o-mini']:
                            print(f"Using OpenAI API for inference (attempt {retry_count + 1}/{max_retries})")
                            response = openai_inference(
                                messages=messages, 
                                model=model, 
                                api_key=openai_api_key,
                                response_model=response_model
                                )
                        else:
                            print(f"Using OpenRouter API for inference (attempt {retry_count + 1}/{max_retries})")
                            response = openrouter_inference(
                                messages=messages, 
                                model=model, 
                                api_key=openrouter_api_key,
                                response_model=response_model
                                )
                        
                        print(f"Response: {response}")
                        
                        # Add to results with item identifier
                        results.append({
                            "item_index": i,
                            "input_item": item,
                            "output": response
                        })
                        
                        # Save updated results after each successful API call
                        with open(output_results_file, 'w') as f:
                            json.dump(results, f, indent=4)
                        
                        success = True  # Exit retry loop on success
                        print(f"Success! Item {i} processed and saved.")    
                        
                    except Exception as e:
                        retry_count += 1
                        print(f"Error on attempt {retry_count}/{max_retries} for item {i}: {e}")
                        if retry_count < max_retries:
                            print(f"Retrying in 2 seconds...")
                            time.sleep(2)  # Wait before retrying
                        else:
                            # All retries failed, save to failed items file
                            failed_items_file = os.path.splitext(output_results_file)[0] + "_failed.json"
                            
                            # Load existing failed items if file exists
                            failed_items = []
                            if os.path.exists(failed_items_file):
                                with open(failed_items_file, 'r') as f:
                                    failed_items = json.load(f)
                            
                            # Add current failed item
                            failed_items.append({
                                "item_index": i,
                                "input_item": item,
                                "error": str(e)
                            })
                            
                            # Save updated failed items
                            with open(failed_items_file, 'w') as f:
                                json.dump(failed_items, f, indent=4)
                                
                            print(f"Item {i} failed after {max_retries} attempts. Saved to {failed_items_file}")
                
            except Exception as e:
                print(f"Unexpected error processing item {i} with content '{content}': {e}")
                continue
                
    except Exception as e:
        print(f"Error processing JSON file: {e}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run inference using a specified config file.')
    parser.add_argument('--config', type=str, default=None, required=True,
                        help='Path to the configuration YAML file')
    args = parser.parse_args()
    
    # Run main function with the provided config file
    main(config_file=args.config)

