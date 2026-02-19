# uses the specified model to generate completions for each prompt

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
import pandas as pd
import os, sys
import re
import argparse
import json
import yaml
import sys
sys.path.append("/home/llm_scenario_modelling/baseline_auto_align/self_rewarding_llm/process")
from filter_risk_opp_responses import extract_chatml_completion_risk_opp
sys.path.append("/home/llm_scenario_modelling/baseline_auto_align/self_rewarding_llm/src")
from srlm.model import load_model


def load_config(yaml_path):
    """Load configuration from a YAML file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

# Set up argparse to accept command-line arguments.
parser = argparse.ArgumentParser(description="Generate responses using prompts.")
parser.add_argument("--config", help="Path to YAML configuration file", default=None)
parser.add_argument("--base_name", nargs="?", help="Tokenizer name / base model", default=None)
parser.add_argument("--model_name", nargs="?", help="Model name", default=None)
parser.add_argument("--prompts_file", nargs="?", help="Path to prompts file", default=None)
parser.add_argument("--responses_file", nargs="?", help="Path to responses file", default=None)
# parser.add_argument("--device", help="Device to load the model onto, e.g. 'cuda:1' or 'cpu'", default="cuda")
parser.add_argument("--mode", help="Mode of operation: inference (one sample) or training (4 samples)", default="training")
args = parser.parse_args()

if args.config:
    config = load_config(args.config)
    base_name = args.base_name or config.get("base_name")
    model_name = args.model_name or config.get("model_name")
    prompts_file = args.prompts_file or config.get("prompts_file")
    responses_file = args.responses_file or config.get("responses_file")
    # device = args.device if args.device != "cuda" or "device" not in config else config.get("device", "cuda")
    mode = config.get("mode", "training")
else:
    base_name = args.base_name
    model_name = args.model_name
    prompts_file = args.prompts_file
    responses_file = args.responses_file
    # device = args.device
    mode = args.mode

if None in (model_name, prompts_file, responses_file):
    print("Usage: python 02_gen_responses.py <model_name> <prompts_file> <responses_file>")
    sys.exit(1)

print("Configuration:")
print("Model Name:", model_name)
print("Prompts File:", prompts_file)
print("Responses File:", responses_file)
# print("Device:", device)
print("Mode:", mode)


def do_sample(model, tokenizer, prompt):
    with torch.no_grad():
        # Create prompt using apply_chat_template

        # system_message = """You are an expert financial analyst.
        # Read the provided news articles carefully.
        # Do not analyze each article individually.
        # Instead, in a concise single paragraph of not more than 300 words,
        # identify the most critical systematic drivers (factors expected to 
        # broadly affect the financial market) and discuss the risks
        # (potential negative outcomes) and opportunities
        # (potential positive outcomes) associated with
        # the identified drivers. Ensure each risk and opportunity
        # explicitly relates to financial market growth, innovation,
        # profitability, or stability. Do not output more than one paragraph,
        # if more than one paragraph is outputted, the response is incorrect."""

        system_message = """
    You are a financial expert tasked with generating minimally edited counterfactual scenarios based on a provided financial news that describes a market development. Your goal is to generate a risk counterfactual scenario and an opportunity counterfactual scenario, following from our requirements below:

    Risk Counterfactual Scenario:
    - Minimally edit the original market development headline to represent a plausible alternate scenario that represents an adverse shift in the market development.
    - The adverse shift should reflect an adverse market outcome or deterioration in market conditions.
    - The alternate scenario must be forward-looking, which means that it can plausibly occur after the original market development.

    Opportunity Counterfactual Scenario:
    - Minimally edit the original market development headline to represent a plausible alternate scenario that represents an positive shift in the market development.
    - The positive market shift reflects a beneficial market outcome or improvement in market conditions.
    - The alternate scenario must be forward-looking, which means that it can plausibly occur after the original market development.

    Input format:
    - A single financial news headline.

    Output format:
    Clearly separate your output into two sections:
    - Risk Counterfactual Scenario: [Your minimally edited risk scenario here.]
    - Opportunity Counterfactual Scenario: [Your minimally edited opportunity scenario here.] 
    """

        prompt_for_model = tokenizer.apply_chat_template([
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ], tokenize=False)

        # Print for debugging
        print(f"Prompt for model: {prompt_for_model}")

        model_inputs = tokenizer(prompt_for_model, return_tensors="pt")
        model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}
        streamer = TextStreamer(tokenizer)

        generated_ids = model.generate(
            **model_inputs,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            streamer=streamer,
            temperature=0.7,
            top_p=0.9,
            max_new_tokens=1000,
            repetition_penalty=1.2
        )

        answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        answer = answer[0]

        return answer
    

def extract_completion_only(answer):
    
    """
    Parse the response using extract_chatml_completion_risk_opp to get risk and opportunity
    counterfactuals, then combine them into a single formatted string.
    
    Args:
        answer (str): The full response from the model
        
    Returns:
        str: Combined risk and opportunity counterfactuals, or the original answer if extraction fails
    """
    try:
        parsed = extract_chatml_completion_risk_opp(answer)
        risk = parsed.get("risk")
        opportunity = parsed.get("opportunity")
        
        # Check if both risk and opportunity are missing
        if risk is None and opportunity is None:
            print("Warning: No risk or opportunity scenarios found in the response")
            return ""
        
        # Construct the combined result
        combined = ""
        if risk:
            combined += f"Risk Counterfactual Scenario:\n{risk}\n\n"
        if opportunity:
            combined += f"Opportunity Counterfactual Scenario:\n{opportunity}"
            
        return combined.strip()
    except Exception as e:
        print(f"Error extracting completion: {e}")
        return answer  # Return original answer if extraction fails

# Load the model and tokenizer from the specified path
model, tokenizer = load_model(base_name, model_name) # base name is basically the tokenizer, model_name is the model
# model.to(device)
model.eval()

df_prompts = pd.read_json(path_or_buf=prompts_file, lines=True)
# Shuffle the dataframe
df_prompts = df_prompts.sample(frac=1).reset_index(drop=True) # sample all rows and shuffle them

# Check if the responses file exists and load existing completions
if os.path.exists(responses_file):
    with open(responses_file, "r") as f:
        existing_completions = [json.loads(line) for line in f if line.strip()]  # Read line by line
else:
    existing_completions = []

completions = existing_completions  # Start with existing data

# Determine number of samples based on mode
# SAMPLING 4 EXAMPLES FOR TRAINING
num_samples = 1 if mode == "inference" else 4

for index, row in df_prompts.iterrows():
    print(f"Processing prompt {index + 1} of {len(df_prompts)}")

    prompt = row['prompt']
    prompt_id = row['prompt_id']

    for completion_sample in range(num_samples):
        print("-----------------------------------------------------------------------")
        print(f"Processing prompt {index + 1}, completion {completion_sample + 1}")

        answer = do_sample(model, tokenizer, prompt)
        completion = extract_completion_only(answer)

        new_entry = {"prompt_id": prompt_id, "prompt": prompt, "completion": completion}
        completions.append(new_entry)  # Append to the list

        print("\n\n")
        print(f"Extracted completion: {completion}")

        # Append new entries to JSON file without rewriting everything
        with open(responses_file, "a") as f:
            f.write(json.dumps(new_entry) + "\n")
