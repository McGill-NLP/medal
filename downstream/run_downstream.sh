#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python run_downstream.py \
    --savedir "directory to save the model" \
    --model "model type (rnn, rnnsoft, electra)" \
    --task "downstream tasks (mimic-diagnosis, mimic-mortality)" \
    --data_dir "root directory of train/valid/test data files" \
    --data_filename "data file filename (should be the same for train/valid/test)" \
    --diag_to_idx_path "path to diag_to_idx file" \
    --embs_path "path to pretrained fasttext embeddings" \
    --epochs 10 \
    --lr 2e-6 \
    --use_scheduler \
    -bs 8 \
    --save_every 1 \
    --dropout 0.1 \
    --rnn_layers 3 \
    --da_layers 1 \
    --hidden_size 512 \
    --eval_every 30000 \
    --pretrained_model "path to the model pretrained on medal" \