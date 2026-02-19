
import argparse
import os
import time
import sys
from datasets import load_dataset
import yaml
sys.path.append("/home/llm_scenario_modelling/baseline_auto_align/self_rewarding_llm/src")
from srlm.trainer import Trainer
from srlm.model import load_model, create_peft_model

def collate_fn(tokenizer, x):

    system_prompt = """
    You are a financial expert tasked with generating minimally edited counterfactuals based on a provided financial headline that describes a market development. Your goal is to generate a risk counterfactual and an opportunity counterfactual, following from our requirements below:

    Risk Counterfactual:
    - Minimally edit the original market development headline to represent a plausible alternate counterfactual that represents an adverse shift in the market development.
    - The adverse shift should reflect an adverse market outcome or deterioration in market conditions.
    - The alternate counterfactual must be forward-looking, which means that it can plausibly occur after the original market development.

    Opportunity Counterfactual:
    - Minimally edit the original market development headline to represent a plausible alternate counterfactual that represents an positive shift in the market development.
    - The positive market shift reflects a beneficial market outcome or improvement in market conditions.
    - The alternate counterfactual must be forward-looking, which means that it can plausibly occur after the original market development.

    Input format:
    - A single financial news headline.

    Output format:
    Clearly separate your output into two sections:
    - Risk Counterfactual: [Your minimally edited risk counterfactual here.]
    - Opportunity Counterfactual: [Your minimally edited opportunity counterfactual here.] 
    """

    text = tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": x['prompt']},
        {"role": "assistant", "content": x['completion']},
    ], tokenize=False)
    return {"text": text}


# this assumes you have a list of list of dictionaries like this:
# conversations = [
#     [
#         {'from': 'system', 'value': 'You are a helpful AI assistant that provides accurate and concise information.'},
#         {'from': 'human', 'value': 'Hi there!'},
#         {'from': 'gpt', 'value': 'Hi how can I help?'},
#         # rest of conversation
#     ],
# from is like the "role" in the previous example, and value is the message content
#     # more conversations
# ]

# def apply_template(examples):
#     messages = examples["conversations"]
#     text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
#     return {"text": text}

# and then we can apply this template to the dataset like this:

# dataset = load_dataset("mlabonne/FineTome-100k", split="train")
# dataset = dataset.map(apply_template, batched=True)

# and then we place this dataset into the model through the SFT trainer.

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='SFT train a model.')
    parser.add_argument('--config', help='Path to YAML configuration file', default=None)
    parser.add_argument('-d', '--dataset', type=str, help='Input SFT dataset')
    parser.add_argument('-b', '--base_model', type=str, default=None, help='The base model we want to fine-tune')
    parser.add_argument('-m', '--model', type=str, default=None, help='The model name we want to fine-tune')
    parser.add_argument('-o', '--output', type=str, help='Output trained model')
    # add device
    # parser.add_argument('--device', type=str, default="cuda", help='Device to load the model onto, e.g. "cuda:1" or "cpu"')
    args = parser.parse_args()

    # If a YAML config is provided, load it and override missing arguments.
    if args.config:
        config = load_config(args.config)
        dataset_file = config.get("dataset")
        base_model = config.get("base_model")
        model_name = config.get("model")
        output_model = config.get("output")
        # device = config.get("device", "cuda")
    else:
        # Ensure required arguments are provided if no config file is used.
        if not args.dataset or not args.output:
            parser.error("Missing required arguments: --dataset and --output are required if no config is provided.")
        dataset_file = args.dataset
        base_model = args.base_model
        model_name = args.model
        output_model = args.output
        # device = args.device

    # Load the training dataset.
    # Example: you can download the dataset file with:
    # `oxen download datasets/Self-Rewarding-Language-Models M0/train/ift.jsonl`
    dataset = load_dataset("json", data_files={'train': dataset_file})
    dataset = dataset['train'].shuffle(seed=42)

    # Load the model.
    model, tokenizer = load_model(base_model, model_name) 
    # model.to(device)
    dataset = dataset.map(lambda x: collate_fn(tokenizer, x))

    print("First example in the dataset:")
    print(dataset['text'][0])

    # Time the training.
    start_time = time.time()

    model = create_peft_model(model)
    trainer = Trainer(output_model)
    trainer.train(model, tokenizer, dataset)
    end_time = time.time()
    print(f"Training time: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()