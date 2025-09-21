#!/bin/bash
#
# ljy@20250220
#

MODEL="pretrain_model"
DATASET="your dataset name"
LANGUAGE="c"

python data_preprocess.py \
    --config_name=${MODEL} \
    --model_name_or_path=${MODEL} \
    --tokenizer_name=${MODEL} \
    --program_language=${LANGUAGE} \
    --dataset=dataset/${DATASET}.jsonl \
    --dataset_name=${DATASET} \
    --code_length 512 \
    --data_flow_length 96 \
    --control_flow_length 20 \
    --vul_relation_length 12 \
    --seed 123456
