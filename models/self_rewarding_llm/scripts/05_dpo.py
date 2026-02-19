from datasets import load_dataset
import time
import yaml
import argparse
from unsloth import is_bfloat16_supported, PatchDPOTrainer
PatchDPOTrainer()
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments, EarlyStoppingCallback
from trl import DPOConfig, DPOTrainer
import sys, os
sys.path.append("/home/llm_scenario_modelling/baseline_auto_align/self_rewarding_llm/src")
from srlm.model import load_model

def load_config(yaml_path):
    """Load configuration from a YAML file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def get_prompt(example, tokenizer):
    prompt_sample = [
        {"role": "system", "content": "You are an expert financial analyst. Read the provided news articles carefully. Do not analyze each article individually. Instead, in a concise single paragraph of not more than 300 words, identify the most critical systematic drivers (factors expected to broadly affect the financial market) and discuss the risks (potential negative outcomes) and opportunities (potential positive outcomes) associated with the identified drivers. Ensure each risk and opportunity explicitly relates to financial market growth, innovation, profitability, or stability. Do not output more than one paragraph, if more than one paragraph is outputted, the response is incorrect."},
        {"role": "user", "content": example['prompt']}
    ]
    # NOTE: IDEALLY THE PROMPT SHOULD BE VANILLA -> I.E. JUST CONTAIN THE NEWS ARTICLES AND NOT THE SYSTEM PROMPT, AS THAT IS ALREADY GIVEN
    prompt_for_model = tokenizer.apply_chat_template(prompt_sample, tokenize=False)
    example['prompt'] = prompt_for_model

    example['chosen'] = str(example['chosen']) + tokenizer.eos_token
    example['rejected'] = str(example['rejected']) + tokenizer.eos_token

    return example

def main():
    # Set up argparse to accept command-line arguments
    parser = argparse.ArgumentParser(description="Train a model with DPO (Direct Preference Optimization).")
    parser.add_argument("--config", help="Path to YAML configuration file", 
                        default=None)
    parser.add_argument("--base_model_name", help="Base model name", default=None)
    parser.add_argument("--model_name", help="Model name", default=None)
    parser.add_argument("--dataset_file", help="Path to dataset JSON file with preferences", default=None)
    parser.add_argument("--output_dir", help="Directory for saving results", default=None)
    parser.add_argument("--batch_size", type=int, help="Batch size for training", default=None)
    parser.add_argument("--learning_rate", type=float, help="Learning rate", default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Gradient accumulation steps", default=None)
    parser.add_argument("--max_length", type=int, help="Maximum sequence length", default=None)
    parser.add_argument("--max_prompt_length", type=int, help="Maximum prompt length", default=None)
    # parser.add_argument("--device", help="Device to load the model onto, e.g. 'cuda:1' or 'cpu'", default="cuda")
    args = parser.parse_args()

    # Load config from YAML if provided, otherwise use command line arguments
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
        base_model_name = args.base_model_name or config.get("base_model_name")
        model_name = args.model_name or config.get("model_name")
        dataset_file = args.dataset_file or config.get("dataset_file")
        output_dir = args.output_dir or config.get("output_dir")
        batch_size = args.batch_size or config.get("batch_size", 1)
        learning_rate = args.learning_rate or config.get("learning_rate", 5e-5)
        gradient_accumulation_steps = args.gradient_accumulation_steps or config.get("gradient_accumulation_steps", 4)
        max_length = args.max_length or config.get("max_length", 2048)
        max_prompt_length = args.max_prompt_length or config.get("max_prompt_length", 512)
        # device = args.device or config.get("device", "cuda")
    else:
        # If no config file, require command line arguments
        if not (args.base_model_name and args.model_name and args.dataset_file and args.output_dir):
            parser.print_help()
            print("\nEither provide a config file or all required command line arguments")
            sys.exit(1)
        base_model_name = args.base_model_name
        model_name = args.model_name
        dataset_file = args.dataset_file
        output_dir = args.output_dir
        batch_size = args.batch_size or 4
        learning_rate = args.learning_rate or 5e-5
        gradient_accumulation_steps = args.gradient_accumulation_steps or 4
        max_length = args.max_length or 2048
        max_prompt_length = args.max_prompt_length or 512
        # device = args.device or "cuda"

    # make sure that the learning_rate is a float
    learning_rate = float(learning_rate)

    # Print configuration for verification
    print(f"Configuration:")
    print(f"  Base Model: {base_model_name}")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {dataset_file}")
    print(f"  Output Directory: {output_dir}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Max Length: {max_length}")
    print(f"  Max Prompt Length: {max_prompt_length}")

    # load the training dataset
    dataset = load_dataset("json", data_files={'train': dataset_file})
    dataset = dataset['train'].shuffle(seed=42)

    # Split dataset into training and validation sets (85% / 15%)
    split_dataset = dataset.train_test_split(test_size=0.15, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(eval_dataset)}")

    # Load model and tokenizer
    model, tokenizer = load_model(base_model_name, model_name)

    # filter out examples that do not have the required keys or are empty
    train_dataset = train_dataset.filter(lambda x: 'prompt' in x and 'chosen' in x and 'rejected' in x and x['prompt'] and x['chosen'] and x['rejected'])
    train_dataset = train_dataset.map(lambda x: get_prompt(x, tokenizer))

    eval_dataset = eval_dataset.filter(lambda x: 'prompt' in x and 'chosen' in x and 'rejected' in x and x['prompt'] and x['chosen'] and x['rejected'])
    eval_dataset = eval_dataset.map(lambda x: get_prompt(x, tokenizer))

    training_args = DPOConfig(
        output_dir=output_dir,
        logging_steps=1,
        per_device_train_batch_size=batch_size,
        lr_scheduler_type="cosine",
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps, # 16
        gradient_checkpointing=True,
        warmup_steps=2,
        num_train_epochs=2,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        beta=0.1,
        evaluation_strategy="steps",
        eval_steps=10, # to test
        save_strategy="steps",
        save_steps=10, # to test
        metric_for_best_model="eval_loss",    # monitor validation loss
        greater_is_better=False, # lower is better
        load_best_model_at_end=True
    )

    # 3. Create an EarlyStoppingCallback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=2,      # stop after 3 eval calls with no improvement
        early_stopping_threshold=0.0    # require any reduction in loss to reset patience
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[early_stopping],
    )

    # Time the training
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    print(f"Training time: {end_time - start_time} seconds")

    final_output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(final_output_dir)
    print(f"Model saved to {final_output_dir}")

if __name__ == "__main__":
    main()