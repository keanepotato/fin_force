#!/bin/bash

## ./self-reward_init_model.sh scripts unsloth/mistral-7b-instruct-v0.3-bnb-4bit C0 3

# unsloth/Qwen2.5-7B-Instruct-bnb-4bit
# unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit
# unsloth/mistral-7b-instruct-v0.3-bnb-4bit
# unsloth/Phi-4-mini-instruct-unsloth-bnb-4bit

### SCRIPT INVOLVES SUPERVISED FINE-TUNING THE MODEL VIA SFT BEFORE DPO

CODE=$1
BASE_MODEL=$2
MODEL_NAME=$3
GPU=${4:-0}

# this should change to the different models
# DATA_DIR=data
DATA_DIR=data/${BASE_MODEL//\//_}

# Increment model version while keeping the same starting letter
MODEL_VERSION="$(echo $MODEL_NAME | sed -E 's/([A-Z])[0-9]+/\1/')$(($(echo $MODEL_NAME | grep -o '[0-9]*') + 1))"
START_LETTER="$(echo $MODEL_NAME | sed -E 's/([A-Z])[0-9]+/\1/')"
SUBSEQUENT_MODEL_VERSION="$START_LETTER$(($(echo $START_MODEL_NAME | grep -o '[0-9]*') + i + 1))"
# DEVICE="cuda:3"
export CUDA_VISIBLE_DEVICES=$GPU

# export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=src
export TOKENIZER_MODEL=BASE_MODEL

# # Train the instruction following only
python "$CODE/00_sft.py" -d "$DATA_DIR/$MODEL_NAME/train/cfs_sft_data.jsonl" -b "$TOKENIZER_MODEL" -m "$BASE_MODEL" -o "$DATA_DIR/$MODEL_NAME/models/sft"

# # Pause to ensure SFT model is saved completely
echo "SFT training completed. Waiting for file system to sync..."
sleep 5

# mkdir -p "$DATA_DIR/$MODEL_NAME/generated"
echo "Created directory for generated data."
sleep 2

python "$CODE/01_gen_prompts.py" \
  --base_name "$TOKENIZER_MODEL" \
  --model_name "$DATA_DIR/$MODEL_NAME/models/sft/final_checkpoint" \
  --training_data "$DATA_DIR/$MODEL_NAME/train/cfs_sft_data.jsonl" \
  --generated_prompts_file "$DATA_DIR/$MODEL_NAME/generated/prompts.jsonl" \
  --num_prompts_to_generate 1000 \

# Pause to ensure prompts are generated and saved
echo "Prompts generated. Waiting for file system to sync..."
sleep 3

# Generate responses for the prompts
# It generates mulitple responses for the same prompt which will be then be rated by the gen_scores
python "$CODE/02_gen_responses.py" \
  --base_name  "$TOKENIZER_MODEL" \
  --model_name "$DATA_DIR/$MODEL_NAME/models/sft/final_checkpoint" \
  --prompts_file "$DATA_DIR/$MODEL_NAME/generated/prompts.jsonl" \
  --responses_file  "$DATA_DIR/$MODEL_NAME/generated/responses.jsonl" \
  --mode "training" \

# Pause to ensure responses are generated and saved
echo "Responses generated. Waiting for file system to sync..."
sleep 3

#!/usr/bin/env bash
set -euo pipefail

RESPONSES="$DATA_DIR/$MODEL_NAME/generated/responses.jsonl"
SCORES="$DATA_DIR/$MODEL_NAME/generated/scores.jsonl"
SPLIT_DIR="$DATA_DIR/$MODEL_NAME/generated/splits"

# ─── CONFIG ────────────────────────────────────────────────────────────────
# How many scoring processes should run *per* GPU?
JOBS_PER_GPU=3

# list of GPUs you want to use
GPUS=(2 3)
NUM_GPUS=${#GPUS[@]}

# total parallel jobs = GPUs × jobs_per_gpu
TOTAL_JOBS=$(( NUM_GPUS * JOBS_PER_GPU ))
# ────────────────────────────────────────────────────────────────────────────

mkdir -p "$SPLIT_DIR"
: > "$SCORES"

# compute lines per chunk (ceil)
TOTAL_LINES=$(wc -l <"$RESPONSES")
CHUNK_SIZE=$(( (TOTAL_LINES + TOTAL_JOBS - 1) / TOTAL_JOBS ))

# split into exactly $TOTAL_JOBS files: splits/part_00.jsonl … part_N.jsonl
split -d -l "$CHUNK_SIZE" --additional-suffix=.jsonl \
      "$RESPONSES" "$SPLIT_DIR/part_"

# launch all scoring jobs in parallel
for job_id in $(seq 0 $((TOTAL_JOBS-1))); do
  PART="$SPLIT_DIR/part_$(printf "%02d" $job_id).jsonl"
  OUT_PART="${SCORES%.jsonl}_part${job_id}.jsonl"

  (
    # assign this job to GPU round-robin
    GPU_IDX=$(( job_id % NUM_GPUS ))
    export CUDA_VISIBLE_DEVICES="${GPUS[$GPU_IDX]}"

    python "$CODE/03_gen_scores.py" \
      --base_name           "$TOKENIZER_MODEL" \
      --model_name          "$DATA_DIR/$MODEL_NAME/models/sft/final_checkpoint" \
      --responses_file      "$PART" \
      --scores_file         "$OUT_PART" \
      --judging_prompt_file "/home/llm_scenario_modelling/baseline_auto_align/self_rewarding_llm/my_llm_judge_prompts/cf_additive.txt"
  ) &

done

# wait for *all* jobs on *all* GPUs
wait

# stitch them back together in order
for job_id in $(seq 0 $((TOTAL_JOBS-1))); do
  cat "${SCORES%.jsonl}_part${job_id}.jsonl" >> "$SCORES"
done

# clean up
rm "$SPLIT_DIR"/part_*.jsonl "${SCORES%.jsonl}"_part*.jsonl

echo "Scores generated. Syncing…"
sleep 3

export CUDA_VISIBLE_DEVICES=$GPU   # reset if you like

# now generate preferences as before
python "$CODE/04_gen_preferences.py" \
    --scores_file      "$SCORES" \
    --preferences_file "$DATA_DIR/$MODEL_NAME/generated/preferences.jsonl"

echo "Preferences generated. Syncing…"
sleep 3

# Pause to ensure preferences are generated and saved
echo "Preferences generated. Waiting for file system to sync..."
sleep 3

# Train DPO model
python $CODE/05_dpo.py \
  --base_model_name "$TOKENIZER_MODEL" \
  --model_name "$DATA_DIR/$MODEL_NAME/models/sft/final_checkpoint" \
  --dataset_file "$DATA_DIR/$MODEL_NAME/generated/preferences.jsonl" \
  --output_dir "$DATA_DIR/$MODEL_NAME/models/dpo/" \
  --batch_size 1 \
  --learning_rate 1e-6 \
  --gradient_accumulation_steps 16 \
  --max_length 2048 \
  --max_prompt_length 1024 \

echo "DPO training completed. Waiting for file system to sync..."
sleep 5

python "/home/llm_scenario_modelling/baseline_auto_align/self_rewarding_llm/scripts/filter_scores.py" \
  --input "$DATA_DIR/$MODEL_NAME/generated/scores.jsonl" \
  --output "$DATA_DIR/$MODEL_NAME/generated/filtered_generated_for_next_iter.jsonl" \
  --min_score 7

echo "Scores filtered. Waiting for file system to sync..."
sleep 3

# 2. Create the new model version directory if it does not exist
mkdir -p "$DATA_DIR/$SUBSEQUENT_MODEL_VERSION/train"

echo "Created directory for next model version: $SUBSEQUENT_MODEL_VERSION"
sleep 2

# 3. Concatenate the original training data with the new generated ift data
cat "$DATA_DIR/$MODEL_NAME/train/cfs_sft_data.jsonl" "$DATA_DIR/$MODEL_NAME/generated/filtered_generated_for_next_iter.jsonl" > "$DATA_DIR/$SUBSEQUENT_MODEL_VERSION/train/cfs_sft_data.jsonl"