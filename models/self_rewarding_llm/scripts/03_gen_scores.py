# for each completion from the previous step, uses m1 to generate a score

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextStreamer
import pandas as pd
import re
import os
import sys
import argparse
import yaml
import sys
sys.path.append("/home/llm_scenario_modelling/baseline_auto_align/self_rewarding_llm/src")
from srlm.model import load_model

# python scripts/gen_scores.py M0/models/sft M0/generated/responses.jsonl M0/generated/scores.jsonl
def load_config(config_path):
    """ Load configuration from a YAML file. """
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading YAML config: {e}")
        sys.exit(1)

# Set up argparse to accept command-line arguments.
parser = argparse.ArgumentParser(description="Generate scores using prompts.")

# Default config file location
parser.add_argument(
    "--config", help="Path to YAML configuration file", 
    default=None
)

# Positional arguments (optional, override YAML values if provided)
parser.add_argument("--base_name", nargs="?", help="Tokenizer name / base model", default=None)
parser.add_argument("--model_name", nargs="?", help="Model name", default=None)
parser.add_argument("--responses_file", nargs="?", help="Path to responses file", default=None)
parser.add_argument("--scores_file", nargs="?", help="Path to scores file", default=None)
# add the prompt file
parser.add_argument("--judging_prompt_file", nargs="?", help="Path to prompt file", default=None)
# Optional device argument (defaults to "cuda" if not provided)
# parser.add_argument("--device", help="Device to load the model onto, e.g. 'cuda:1' or 'cpu'", default="cuda")

# Parse arguments
args = parser.parse_args()

# Load YAML config if provided
if args.config:
    config = load_config(args.config)

    # Use command-line arguments if provided, otherwise fallback to YAML config values
    base_name = args.base_name or config.get("base_name")
    model_name = args.model_name or config.get("model_name")
    responses_file = args.responses_file or config.get("responses_file")
    scores_file = args.scores_file or config.get("scores_file")
    judging_prompt_file = args.judging_prompt_file or config.get("judging_prompt_file")

    # Use command-line --device if provided, otherwise fallback to YAML or default to "cuda"
    # device = args.device if args.device != "cuda" or "device" not in config else config.get("device", "cuda")
else:
    # If no config file is provided, rely only on command-line arguments
    base_name = args.base_name
    model_name = args.model_name
    responses_file = args.responses_file
    scores_file = args.scores_file
    # device = args.device
    judging_prompt_file = args.judging_prompt_file

# Validate required arguments
if None in (model_name, responses_file, scores_file):
    print("Usage: python 03_gen_scores.py <base_model_name> <model_name> <responses_file> <scores_file> <judging_prompt_file>")
    sys.exit(1)

# Print configuration summary
print("Configuration:")
print(f"Base Model Name: {base_name}")
print(f"Model Name: {model_name}")
print(f"Responses File: {responses_file}")
print(f"Scores File: {scores_file}")
# print(f"Device: {device}")
print(f"Judging Prompt File: {judging_prompt_file}")

def format_input_prompt(news, response):
    """
    Format the input prompt with news and response clearly delineated.
    
    Args:
        news (str): The news articles text
        response (str): The generated response to evaluate
        
    Returns:
        str: Formatted prompt with news in <news> tags and response in <response> tags
    """
    return f"<news>{news}</news>\n<response>{response}</response>"

def do_sample(model, tokenizer, prompt, chat_format=False):
    with torch.no_grad():
        # system_prompt, input_prompt = chat_format
        # prompt_for_model = tokenizer.apply_chat_template([
        #     {"role": "system", "content": system_prompt},
        #     {"role": "user", "content": input_prompt},
        # ], tokenize=False)

        prompt_for_model = f"<s>[INST] {prompt} [/INST]"

        print("-----------------------------------------------------------------------")
        print(f"Prompt for model: {prompt_for_model}")

        # print(f"Prompt for model: {prompt_for_model}")

        model_inputs = tokenizer(prompt_for_model, return_tensors="pt")
        model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

        streamer = TextStreamer(tokenizer)

        generated_ids = model.generate(
            **model_inputs,
            do_sample=True,
            streamer=streamer,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            max_new_tokens=200, # since the score is at the beginning; make the explanation short
            top_p = 0.9,
            temperature = 0.7,
        )

        # print(f"Q: {prompt}:")
        # print("-------------------------")

        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        answer = decoded[0]

        return answer

model, tokenizer = load_model(base_name, model_name)
# model.to(device)
model.eval()

df = pd.read_json(path_or_buf=responses_file, lines=True)

# Load the judging prompt


with open(judging_prompt_file, 'r') as file:
    llm_as_a_judge_prompt = file.read()
    # close the file
    file.close()


import os
import json

pattern = r"[Ss]core: (10|[1-9])"
results = []

# Check if the file exists
if not os.path.exists(scores_file):
    # Create the file by opening it in write mode
    with open(scores_file, 'w') as f:
        # Optionally, write an initial value, e.g., an empty string
        f.write('')
    print(f"{scores_file} created.")
else:
    print(f"{scores_file} already exists.")


with open(scores_file, "a") as f:
    for index, row in df.iterrows():
        prompt_id = row['prompt_id']
        news      = row['prompt']
        response  = row['completion']

        judge_prompt    = llm_as_a_judge_prompt.format(news=news, response=response)
        all_scores      = []
        all_reasonings  = []

        # draw 3 independent samples
        for _ in range(3):
            answer = do_sample(model, tokenizer, judge_prompt)
            match  = re.findall(pattern, answer)
            score  = int(match[0]) if match else -1

            if score != -1:
                all_scores.append(score)
                all_reasonings.append(answer)

        # compute average (or -1 if no valid scores)
        score = sum(all_scores) / len(all_scores) if all_scores else -1

        record = {
            "prompt_id":  prompt_id,
            "prompt":     news,
            "completion": response,
            "score":      score,
            "all_scores": all_scores,
            "all_explanations": all_reasonings
        }
        results.append(record)

        f.write(json.dumps(record) + "\n")
        f.flush()

print("Done!")