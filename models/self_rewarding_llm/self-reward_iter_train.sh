#!/bin/bash
## Example command:
## Start from C0 and train for 2 iterations: Meaning that we will obtain C1 and C2
##   ./self-reward_iter_train.sh scripts unsloth/llama-3-8b-Instruct-bnb-4bit CF0 1
## where C0 is the base model version you want to start training from

CODE=$1
BASE_MODEL=$2
START_MODEL_NAME=$3
NUM_ITERATIONS=$4  # Number of iterations to train

# change to the data directory that we want; corresponding to the models
DATA_DIR=data/${BASE_MODEL//\//_}

# Set up the environment: update TOKENIZER_MODEL to match the reference configuration.
export PYTHONPATH=src
export TOKENIZER_MODEL=BASE_MODEL
export CUDA_VISIBLE_DEVICES=3
# DEVICE="cuda:3"

for ((i=0; i<$NUM_ITERATIONS; i++))
do
    # Extract the starting letter from START_MODEL_NAME
    START_LETTER="$(echo $START_MODEL_NAME | sed -E 's/([A-Z])[0-9]+/\1/')"

    # Increment model version dynamically based on the starting letter
    NEXT_MODEL_VERSION="$START_LETTER$(($(echo $START_MODEL_NAME | grep -o '[0-9]*') + i + 1))"
    # subsequent comes after next
    SUBSEQUENT_MODEL_VERSION="$START_LETTER$(($(echo $START_MODEL_NAME | grep -o '[0-9]*') + i + 2))"
    PREV_MODEL_VERSION="$START_LETTER$(($(echo $START_MODEL_NAME | grep -o '[0-9]*') + i))"
    
    # Create directories for this iteration
    echo "Creating directory: $DATA_DIR/$NEXT_MODEL_VERSION/generated"
    mkdir -p "$DATA_DIR/$NEXT_MODEL_VERSION/generated"

    # 1. Generate new prompts from a base LM and original SFT data
    python "$CODE/01_gen_prompts.py" \
      --base_name "$TOKENIZER_MODEL" \
      --model_name "$DATA_DIR/$PREV_MODEL_VERSION/models/dpo/final_checkpoint" \
      --training_data "$DATA_DIR/$NEXT_MODEL_VERSION/train/cfs_sft_data.jsonl" \
      --generated_prompts_file "$DATA_DIR/$NEXT_MODEL_VERSION/generated/prompts.jsonl" \
      --num_prompts_to_generate 500 \

    # 2. Generate responses for the prompts
    python "$CODE/02_gen_responses.py" \
      --base_name "$TOKENIZER_MODEL" \
      --model_name "$DATA_DIR/$PREV_MODEL_VERSION/models/dpo/final_checkpoint" \
      --prompts_file "$DATA_DIR/$NEXT_MODEL_VERSION/generated/prompts.jsonl" \
      --responses_file "$DATA_DIR/$NEXT_MODEL_VERSION/generated/responses.jsonl" \
      --mode "training"

    # 3. Generate scores for the responses
    python "$CODE/03_gen_scores.py" \
      --base_name "$TOKENIZER_MODEL" \
      --model_name "$DATA_DIR/$PREV_MODEL_VERSION/models/dpo/final_checkpoint" \
      --responses_file "$DATA_DIR/$NEXT_MODEL_VERSION/generated/responses.jsonl" \
      --scores_file "$DATA_DIR/$NEXT_MODEL_VERSION/generated/scores.jsonl" \
      --judging_prompt_file "/home/llm_scenario_modelling/baseline_auto_align/self_rewarding_llm/my_llm_judge_prompts/cf_additive.txt" \

    # 4. Create DPO data (preferences)
    python "$CODE/04_gen_preferences.py" \
      --scores_file "$DATA_DIR/$NEXT_MODEL_VERSION/generated/scores.jsonl" \
      --preferences_file "$DATA_DIR/$NEXT_MODEL_VERSION/generated/preferences.jsonl"

    # 5. Train DPO model
    python "$CODE/05_dpo.py" \
      --base_model_name "$TOKENIZER_MODEL" \
      --model_name "$DATA_DIR/$PREV_MODEL_VERSION/models/dpo/final_checkpoint" \
      --dataset_file "$DATA_DIR/$NEXT_MODEL_VERSION/generated/preferences.jsonl" \
      --output_dir "$DATA_DIR/$NEXT_MODEL_VERSION/models/dpo/" \
      --batch_size 1 \
      --learning_rate 1e-6 \
      --gradient_accumulation_steps 16 \
      --max_length 2048 \
      --max_prompt_length 1024 \

    # --- Data post-processing: filtering and concatenation ---

    # 1. Filter generated scores: keep entries with score > 4, select specific fields,
    #    and add a 'source' field with value "generated"
    python "/home/llm_scenario_modelling/baseline_auto_align/self_rewarding_llm/scripts/filter_scores.py" \
      --input "$DATA_DIR/$NEXT_MODEL_VERSION/generated/scores.jsonl" \
      --output "$DATA_DIR/$NEXT_MODEL_VERSION/generated/filtered_generated_for_next_iter.jsonl" \
      --min_score 7


    # 2. Create the new model version directory if it does not exist
    mkdir -p "$DATA_DIR/$SUBSEQUENT_MODEL_VERSION/train"

    # 3. Concatenate the original training data with the new generated data
    cat "$DATA_DIR/$NEXT_MODEL_VERSION/train/cfs_sft_data.jsonl" "$DATA_DIR/$NEXT_MODEL_VERSION/generated/filtered_generated_for_next_iter.jsonl" > "$DATA_DIR/$SUBSEQUENT_MODEL_VERSION/train/cfs_sft_data.jsonl"

    # NOTE: UNECESSARY FILTERING STEP IS COMMENTED OUT

done