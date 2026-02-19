import torch
import pandas as pd
from transformers import TextStreamer
import sys
import uuid
import re
import random
from tqdm import tqdm

sys.path.append("/home/llm_scenario_modelling/baseline_auto_align/self_rewarding_llm/src")
from srlm.model import load_model

# if len(sys.argv) != 5:
#     print("Usage: python 01_gen_prompts.py <tokenizer_name> <model_name> <train.jsonl> <prompts.jsonl>")
#     exit()

# base_name = sys.argv[1]
# model_name = sys.argv[2]
# ift_dataset_file = sys.argv[3]
# generated_prompts_file = sys.argv[4]

# device = "cuda"  # the device to load the model onto

# # Total number of complete prompts (each with a fixed number of examples) to generate.
# num_prompts_to_generate = 100
# # Number of examples to collect for each prompt.
# target_examples_per_prompt = 20

import argparse
import yaml
import sys
import json

def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

# Set up argparse to accept command-line arguments.
parser = argparse.ArgumentParser(description="Generate prompts for SFT training.")
parser.add_argument("--config", help="Path to YAML configuration file", default=None)
parser.add_argument("--base_name", nargs="?", help="Tokenizer name / base model", default=None)
parser.add_argument("--model_name", nargs="?", help="Model name", default=None)
parser.add_argument("--training_data", nargs="?", help="Input SFT dataset file (JSONL)", default=None)
parser.add_argument("--generated_prompts_file", nargs="?", help="Output prompts JSONL file", default=None)
parser.add_argument("--num_prompts_to_generate", type=int, help="Number of prompts to generate", default=500)
# parser.add_argument("--device", help="Device to load the model onto, e.g. 'cuda:1' or 'cpu'", default="cuda")
args = parser.parse_args()

# If a YAML config file is provided, load it and override missing arguments.
if args.config:
    config = load_config(args.config)
    base_name = args.base_name or config.get("base_name")
    model_name = args.model_name or config.get("model_name")
    train_dataset_file = args.training_data or config.get("training_data")
    generated_prompts_file = args.generated_prompts_file or config.get("generated_prompts_file")
    num_prompts_to_generate = args.num_prompts_to_generate or config.get("num_prompts_to_generate", 500)
    # device = args.device

else:
    # Use command-line positional arguments.
    if None in (args.base_name, args.model_name, args.training_data, args.generated_prompts_file):
        print("Error: Missing required arguments. Please provide base_name, model_name, training_data, and generated_prompts_file.")
        sys.exit(1)
    base_name = args.base_name
    model_name = args.model_name
    train_dataset_file = args.training_data
    generated_prompts_file = args.generated_prompts_file
    # Set defaults if not using YAML.
    num_prompts_to_generate = args.num_prompts_to_generate
    # device = args.device

# Print configuration for verification.
print("Configuration:")
print("base_name:", base_name)
print("model_name:", model_name)
print("training_data:", train_dataset_file)
print("generated_prompts_file:", generated_prompts_file)
print("num_prompts_to_generate:", num_prompts_to_generate)


def read_jsonl_file(file_path):
    """Read a JSONL file into a pandas DataFrame."""
    return pd.read_json(file_path, lines=True)

def save_to_jsonl(df, file_path):
    """Save a DataFrame to a JSONL file."""
    df.to_json(file_path, orient='records', lines=True)

def generate_prompt_from_examples(examples):
    """
    Generate a prompt with clearly signposted task and examples.
    The task is enclosed in <task></task> tags.
    Note: The examples that follow are enclosed within <example></example> tags.
    """
    task_text = (
        "<task> Come up with one new financial news headline. Write only the financial news headline, with no further text or explanation. "
        "The examples below are enclosed in <example></example> tags. </task>"
    )
    
    examples_text = ""
    for item in examples:
        examples_text += f"<example>{item}</example>\n"
    
    prompt = task_text + "\n" + examples_text
    return prompt

def do_sample(model, tokenizer, examples):
    """Call the model to sample a generated response based on the provided examples."""
    with torch.no_grad():
        n_shot_prompt = generate_prompt_from_examples(examples)
        print("<" * 80)
        print(n_shot_prompt)
        print(">" * 80)
        
        model_inputs = tokenizer(n_shot_prompt, return_tensors="pt")
        model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

        streamer = TextStreamer(tokenizer)
        generated_ids = model.generate(
            **model_inputs,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            streamer=streamer,
            top_p=0.9,
            temperature=0.6,
            max_new_tokens=256
        )

        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        answer = decoded[0]
        return answer

def get_random_prompts(df):
    """
    Select examples from the dataset based on their source:
    - 6 examples from source "gpt-4o"
    - 2 examples from source "generated" (if available)
    
    If there aren't enough examples of a particular type, it will take as many as possible.
    If no "generated" examples exist, it will just use the examples from "gpt-4o".
    """
    # Filter dataframe by source
    gpt4o_examples = df[df['source'] == 'source_gpt-4o']
    generated_examples = df[df['source'] == 'generated']
    
    # Calculate how many to take from each source
    n_gpt4o = min(6, len(gpt4o_examples))
    
    # Only try to get generated examples if they exist
    selected_examples = []
    
    # Get examples from gpt-4o
    if n_gpt4o > 0:
        selected_gpt4o = gpt4o_examples.sample(n=n_gpt4o)['prompt'].tolist()
        selected_examples.extend(selected_gpt4o)
    
    # Only try to get generated examples if they exist
    if not generated_examples.empty:
        n_generated = min(2, len(generated_examples))
        selected_generated = generated_examples.sample(n=n_generated)['prompt'].tolist()
        selected_examples.extend(selected_generated)
    
    # If we don't have enough examples total, log a warning
    if len(selected_examples) < 8 and not generated_examples.empty:
        print(f"Warning: Not enough examples available. Using {len(selected_examples)} total examples.")
    
    # Shuffle to randomize the order
    random.shuffle(selected_examples)
    
    return selected_examples

def append_to_jsonl(records, file_path):
    """Append a list of records (dicts) as JSON lines to a file."""
    with open(file_path, "a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def extract_responses(answer, provided_examples):
    """
    Extract new prompts from the model's answer by finding text within <example></example> tags.
    Filter out any examples that exactly match the provided examples.
    """
    print("=" * 80)
    print("Extracting prompts from the generated answer...")
    print(answer)
    print("=" * 80)
    
    # Extract all text within <example></example> tags.
    pattern = r"<example>(.*?)</example>"
    extracted = re.findall(pattern, answer, flags=re.DOTALL)
    
    # Filter out any examples that match the provided ones.
    provided_clean = [ex.strip() for ex in provided_examples]
    new_prompts = [ex.strip() for ex in extracted if ex.strip() not in provided_clean]
    
    print("Generated Prompts:")
    print(new_prompts)
    return new_prompts

# Load model and tokenizer.
model, tokenizer = load_model(base_name, model_name)
# model.to(device)
model.eval()
train_df = read_jsonl_file(train_dataset_file) 

prompt_counter = 0
pbar = tqdm(total=num_prompts_to_generate, desc="Generated Prompts")

while prompt_counter < num_prompts_to_generate:
    # For each complete prompt, collect a set of unique examples.
    filtered_generated_examples = set()

    # Sample random examples from the dataset to use in the prompt.
    task_examples = get_random_prompts(train_df)
    answer = do_sample(model, tokenizer, task_examples)
    generated = extract_responses(answer, task_examples)
    for ex in generated:
        if ex and ex not in task_examples:
            filtered_generated_examples.add(ex)

    # Remove empty strings from current examples.
    filtered_generated_examples = set(filter(None, filtered_generated_examples))
    print("Current collected examples for this prompt:", len(filtered_generated_examples))
    # Once the target number of examples is reached, format them into a complete prompt.

    for example in filtered_generated_examples:
        prompt_id = str(uuid.uuid4())
        new_record = [{"prompt_id": prompt_id, "prompt": example, "source": "generated"}]  
        # Append the new record to the JSONL file.
        append_to_jsonl(new_record, generated_prompts_file)
        # Update counter and progress bar.
        prompt_counter += 1
        pbar.update(1)

pbar.close()
print("Finished generating prompts.")