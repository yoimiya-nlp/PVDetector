#!/bin/bash
#
# ljy@20250220
#

MODEL="pretrain_model"
DATASET="vulcnn"
FROM_CHECKPOINT="vulcnn-v1"
TO_CHECKPOINT="vulcnn-v1"

python finetune.py \
    --output_dir=saved_models \
    --config_name=${MODEL} \
    --model_name_or_path=${MODEL} \
    --tokenizer_name=${MODEL} \
    --do_train \
    --do_test \
    --dataset=${DATASET} \
    --from_checkpoint=${FROM_CHECKPOINT} \
    --to_checkpoint=${TO_CHECKPOINT} \
    --train_data_file=${DATASET}_train.pkl \
    --eval_data_file=${DATASET}_val.pkl \
    --test_data_file=${DATASET}_test.pkl \
    --epoch 2 \
    --code_length 512 \
    --data_flow_length 96 \
    --control_flow_length 20 \
    --vul_relation_length 12 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee saved_models/train.log
