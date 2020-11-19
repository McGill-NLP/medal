#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 python run.py \
    --savedir "directory to save the model" \
    --model "model type (rnn, rnnsoft, electra)" \
    --data_dir "root directory of train/valid/test data files" \
    --adam_path "path to the adam mapping file" \
    --embs_path "path to pretrained fasttext embeddings" \
    --data_filename "data file filename (should be the same for train/valid/test)" \
    --epochs 10 \
    --lr 2e-6 \
    --use_scheduler \
    -bs 8 \
    --save_every 1 \
    --dropout 0.1 \
    --rnn_layers 3 \
    --da_layers 1 \
    --hidden_size 512 \
    --eval_every 200000 \
    # --pretrained_model "path to pretrained model"