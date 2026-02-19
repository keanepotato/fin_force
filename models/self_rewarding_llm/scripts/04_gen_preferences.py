
import json
import os
import sys
import uuid

import argparse
import yaml

def load_config_from_yaml(yaml_file):
    """Load configuration from a YAML file."""
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

parser = argparse.ArgumentParser(
    description="Process scores and preferences files."
)
parser.add_argument(
    '--config', 
    help="Path to YAML config file with keys 'scores_file' and 'preferences_file'",
    default=None
)
parser.add_argument(
    '--scores_file', 
    nargs='?',
    help="Path to scores.jsonl"
)
parser.add_argument(
    '--preferences_file', 
    nargs='?',
    help="Path to preferences.jsonl"
)

args = parser.parse_args()

if args.config:
    config = load_config_from_yaml(args.config)
    scores_file = config.get('scores_file')
    preferences_file = config.get('preferences_file')
    if not scores_file or not preferences_file:
        parser.error("YAML config must contain 'scores_file' and 'preferences_file' keys.")
else:
    if not args.scores_file or not args.preferences_file:
        parser.error("Usage: python 04_gen_preferences.py <scores.jsonl> <preferences.jsonl> or use --config <config.yaml>")
    scores_file = args.scores_file
    preferences_file = args.preferences_file

print(f"Scores file: {scores_file}")
print(f"Preferences file: {preferences_file}")

# Group all the prompts by prompt_id
prompts = {}
with open(scores_file, "r") as f:
    for line in f:
        row = json.loads(line)

        prompt_id = row['prompt_id']
        if prompt_id not in prompts:
            prompts[prompt_id] = []

        prompts[row['prompt_id']].append(row)

# Iterate over prompts and look at high and low scores to generate preference pairs
# if the score is the same, skip
pairs = []

# the problem that we have with this is that there are too many -1 scores

for prompt_id, prompts in prompts.items():
    # find the best score
    best_score = -1
    best_prompt = None
    for prompt in prompts:
        if prompt['score'] > best_score:
            best_score = prompt['score']
            best_prompt = prompt
    # find the worst score
    worst_score = 100
    worst_prompt = None
    for prompt in prompts:
        if prompt['score'] < worst_score:
            worst_score = prompt['score']
            worst_prompt = prompt

    if None == best_prompt or None == worst_prompt:
        continue

    if best_score == worst_score:
        continue

    ## NOTE: MISSING ADDITIONAL FILTERING TECHNIQUES INCLUDING ROUGE-L, similarity check, keyword filtering, length filtering.
    
    pairs.append({
        "prompt_id": best_prompt['prompt_id'],
        "prompt": best_prompt['prompt'],
        "chosen": best_prompt['completion'],
        "rejected": worst_prompt['completion'],
        "score_chosen": best_prompt['score'],
        "score_rejected": worst_prompt['score']
    })



with open(preferences_file, "w") as f:
    for line in pairs:
        f.write(json.dumps(line) + "\n")

