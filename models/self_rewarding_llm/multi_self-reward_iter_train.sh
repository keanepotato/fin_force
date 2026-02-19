#!/bin/bash
## Example command:
## Start from C0 and train for 2 iterations: Meaning that we will obtain C1 and C2
##   ./multi_self-reward_iter_train.sh scripts unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit C0 1 3
## where C0 is the base model version you want to start training from

CODE=$1
BASE_MODEL=$2
START_MODEL_NAME=$3
NUM_ITERATIONS=$4  # Number of iterations to train
GPU=${4:-0}

# change to the data directory that we want; corresponding to the models
DATA_DIR=data/${BASE_MODEL//\//_}

# Set up the environment: update TOKENIZER_MODEL to match the reference configuration.
export PYTHONPATH=src
export TOKENIZER_MODEL=BASE_MODEL
export CUDA_VISIBLE_DEVICES=$GPU
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

    # Pause to ensure prompts are generated and saved
    echo "Prompts generated. Waiting for file system to sync..."
    sleep 3

    # ─── 2. RESPONSES: Generate responses for the prompts
    set -euo pipefail

    # ─── CONFIG ────────────────────────────────────────────────────────────────
    # How many generation processes per GPU?
    JOBS_PER_GPU=3

    # Which GPUs to use?
    GPUS=(2 3)
    NUM_GPUS=${#GPUS[@]}

    # Paths
    PROMPTS="$DATA_DIR/$NEXT_MODEL_VERSION/generated/prompts.jsonl"
    RESPONSES="$DATA_DIR/$NEXT_MODEL_VERSION/generated/responses.jsonl"
    SPLIT_DIR="$DATA_DIR/$NEXT_MODEL_VERSION/generated/resp_splits"
    # ────────────────────────────────────────────────────────────────────────────

    TOTAL_JOBS=$(( NUM_GPUS * JOBS_PER_GPU ))

    mkdir -p "$SPLIT_DIR"
    : > "$RESPONSES"

    # compute lines per chunk (ceil)
    TOTAL_LINES=$(wc -l <"$PROMPTS")
    CHUNK_SIZE=$(( (TOTAL_LINES + TOTAL_JOBS - 1) / TOTAL_JOBS ))

    # split into exactly $TOTAL_JOBS files: resp_splits/part_00.jsonl … part_N.jsonl
    split -d -l "$CHUNK_SIZE" --additional-suffix=.jsonl \
          "$PROMPTS" "$SPLIT_DIR/part_"

    # launch all generation jobs in parallel
    for job_id in $(seq 0 $((TOTAL_JOBS-1))); do
      PART_PROMPTS="$SPLIT_DIR/part_$(printf "%02d" "$job_id").jsonl"
      OUT_PART="${RESPONSES%.jsonl}_part${job_id}.jsonl"

      (
        # round-robin assign GPU
        GPU_IDX=$(( job_id % NUM_GPUS ))
        export CUDA_VISIBLE_DEVICES="${GPUS[$GPU_IDX]}"

        python "$CODE/02_gen_responses.py" \
          --base_name      "$TOKENIZER_MODEL" \
          --model_name     "$DATA_DIR/$PREV_MODEL_VERSION/models/dpo/final_checkpoint" \
          --prompts_file   "$PART_PROMPTS" \
          --responses_file "$OUT_PART" \
          --mode           "training"
      ) &
    done

    # wait for all jobs to finish
    wait

    # stitch them back together in order
    for job_id in $(seq 0 $((TOTAL_JOBS-1))); do
      cat "${RESPONSES%.jsonl}_part${job_id}.jsonl" >> "$RESPONSES"
    done

    # cleanup
    rm "$SPLIT_DIR"/part_*.jsonl "${RESPONSES%.jsonl}"_part*.jsonl

    echo "Responses generated. Syncing…"
    sleep 3

    # reset to your main GPU if needed
    export CUDA_VISIBLE_DEVICES=$GPU

    # Pause to ensure responses are generated and saved
    echo "Responses generated. Waiting for file system to sync..."
    sleep 3

    # ─── 3. SCORING: Generate scores for the responses, in 4 GPU-specific chunks (in parallel):
    set -euo pipefail

    RESPONSES="$DATA_DIR/$NEXT_MODEL_VERSION/generated/responses.jsonl"
    SCORES="$DATA_DIR/$NEXT_MODEL_VERSION/generated/scores.jsonl"
    SPLIT_DIR="$DATA_DIR/$NEXT_MODEL_VERSION/generated/splits"

    # ─── CONFIG ────────────────────────────────────────────────────────────────
    # How many scoring processes should run *per* GPU?
    JOBS_PER_GPU=3

    # list of GPUs you want to use
    GPUS=(1 2 3)
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
              --model_name          "$DATA_DIR/$PREV_MODEL_VERSION/models/dpo/final_checkpoint" \
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

    # # 3. Concatenate the original training data with the new generated data
    cat "$DATA_DIR/$NEXT_MODEL_VERSION/train/cfs_sft_data.jsonl" "$DATA_DIR/$NEXT_MODEL_VERSION/generated/filtered_generated_for_next_iter.jsonl" > "$DATA_DIR/$SUBSEQUENT_MODEL_VERSION/train/cfs_sft_data.jsonl"

    # NOTE: UNECESSARY FILTERING STEP IS COMMENTED OUT

done