#!/bin/bash

# Check if a model name is provided as an argument
if [ -z "$1" ]; then
  echo "No model name provided. Usage: $0 'model_path_or_name'"
  exit 1
fi

# Assign the first argument to the MODEL_NAME variable
MODEL_NAME=$1

# Execute the lm_eval command with the specified model
lm_eval --model hf \
    --model_args pretrained=$MODEL_NAME,trust_remote_code=True \
    --tasks mmlu -f 5 \
    --device cuda:0 \
    --batch_size 1 && \

lm_eval --model hf \
    --model_args pretrained=$MODEL_NAME,trust_remote_code=True \
    --tasks winogrande -f 5\
    --device cuda:0 \
    --batch_size 4 && \

lm_eval --model hf \
    --model_args pretrained=$MODEL_NAME,trust_remote_code=True \
    --tasks arc_challenge -f 25\
    --device cuda:0 \
    --batch_size 2 && \

lm_eval --model hf \
    --model_args pretrained=$MODEL_NAME,trust_remote_code=True \
    --tasks hellaswag -f 10 \
    --device cuda:0 \
    --batch_size 2 && \

lm_eval --model hf \
    --model_args pretrained=$MODEL_NAME,trust_remote_code=True \
    --tasks gsm8k -f 5 \
    --device cuda:0 \
    --batch_size 2 && \

lm_eval --model hf \
    --model_args pretrained=$MODEL_NAME,trust_remote_code=True \
    --tasks truthfulqa_mc2 \
    --device cuda:0 \
    --batch_size 4
