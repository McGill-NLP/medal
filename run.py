import argparse
import os
import time
import pandas as pd

import torch
import torch.optim as optim
from torch import nn

from models.rnn import RNN
from models.lstm_sa import RNNAtt
from models.electra import Electra
from transformers import ElectraTokenizer
from utils import load_dataframes, load_model, train_loop
from models.tokenizer_and_dataset import \
    FastTextTokenizer, EmbeddingsDataset, HuggingfaceDataset

from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", help="Directory for storing experiments", required=True)
    parser.add_argument("--model", help="Type of model", \
        choices=['rnnsoft', 'electra', 'rnn'], required=True)

    parser.add_argument("--data_dir", help="Root path for train/valid/test data files", required=True)
    parser.add_argument("--data_filename", \
        help="Data file name (train/valid/test should have same name, in different dirs)", required=True)
    parser.add_argument("--adam_path", help="Path to ADAM abbreviations mapping table", required=True)
    parser.add_argument("--embs_path", help="Path to pretrained Fasttext embeddings")

    parser.add_argument("--pretrained_model", help="Path to previously trained model")

    parser.add_argument("--use_scheduler", help="Whether to use lr scheduler", action="store_true")
    parser.add_argument("--lr", help="Learning rate", default=0.001, type=float)
    parser.add_argument("--clip", help="Gradient clipping", default=0, type=float)
    parser.add_argument("--dropout", help="Drop out rate", default=0.1, type=float)
    parser.add_argument("--epochs", help="Number of epochs", default=5, type=int)
    parser.add_argument("--accum_num", help="Number of batches for gradient accumulation", default=1, type=int)
    parser.add_argument("--save_every", help="Frequency of checkpoint", default=40, type=int)
    parser.add_argument("--eval_every", help="Frequency of evaluation", default=10000, type=int)
    parser.add_argument("-bs", "--batchsize", help="Batch Size", default=128, type=int)

    parser.add_argument("--hidden_size", help="Hidden layer size", type=int)
    parser.add_argument("--rnn_layers", help="Number of RNN layers", default=1, type=int)
    parser.add_argument("--da_layers", help="Number of decision attention layers", default=1, type=int)
    parser.add_argument("--ncpu", help="Number of CPU cores", default=4, type=int)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    model_desc = "\n".join([f"{k}: {v}" for k, v in vars(args).items()])
    print("configurations:\n" + model_desc)

    EXPERIMENT_DIR = args.savedir
    N_EPOCHS = args.epochs
    BATCH_SIZE = args.batchsize
    N_CPU_CORES = args.ncpu
    MODEL_TYPE = args.model
    USE_PRETRAIN = True if args.pretrained_model else False

    # Prelim
    torch.set_num_threads(N_CPU_CORES)
    DEVICE = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train, valid, test, label_to_ix = \
        load_dataframes(data_dir=args.data_dir, data_filename=args.data_filename, adam_path=args.adam_path)
    print("Data loaded")

    # Create tokenizer objects
    if MODEL_TYPE == "electra":
        tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
    else:
        # Create word index and load Fasttext embedding matrix
        tokenizer = FastTextTokenizer(verbose=True)
        tokenizer.build_word_index(train.TEXT, valid.TEXT, test.TEXT, list(label_to_ix.keys()))
        tokenizer.build_embedding_matrix(args.embs_path)

    # Create torch Dataset objects
    if MODEL_TYPE == "electra":
        train_data = HuggingfaceDataset(train, tokenizer=tokenizer, device=DEVICE)
        valid_data = HuggingfaceDataset(valid, tokenizer=tokenizer, device=DEVICE)
    else:
        train_data = EmbeddingsDataset(train, tokenizer=tokenizer, device=DEVICE)
        valid_data = EmbeddingsDataset(valid, tokenizer=tokenizer, device=DEVICE)
    print("Dataset created")

    # Define network, loss function and optimizer
    if MODEL_TYPE == "rnn":   # reproduce from Xing Han
        LSTM_PARAMS = dict(
            batch_first=True,
            num_layers=args.rnn_layers,
            dropout=args.dropout,
            bidirectional=True,
            hidden_size=args.hidden_size
        )
        net = RNN(
            embedding_dim=tokenizer.embedding_matrix.shape[1],
            output_size=len(label_to_ix),
            rnn_params=LSTM_PARAMS,
            device=DEVICE
        )
    elif MODEL_TYPE == "rnnsoft":
        net = RNNAtt(
            rnn_layers=args.rnn_layers,
            da_layers=args.da_layers,
            output_size=len(label_to_ix),
            d_model=args.hidden_size,
            device=DEVICE,
            dropout_rate=args.dropout,
        )
    elif MODEL_TYPE == "electra":
        net = Electra(
            output_size=len(label_to_ix),
            device=DEVICE,
        )
    print('model: {}'.format(net))
    if USE_PRETRAIN:
        net = load_model(net, args.pretrained_model, DEVICE)
    if torch.cuda.device_count() > 1:
        net.to(DEVICE)
        print("Using", torch.cuda.device_count(), "GPUs")
        net = nn.DataParallel(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=8) \
        if args.use_scheduler else None

    # Create save directory
    time_stamp = time.strftime("%m-%d-%H-%M", time.localtime())
    save_dir = os.path.join(EXPERIMENT_DIR, time_stamp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save configs
    model_desc_output = [": ".join([str(k), str(v)]) for k, v in vars(args).items()]
    with open(os.path.join(save_dir, 'configs.txt'), 'w') as file:
        file.writelines("\n".join(model_desc_output))

    # Set up tensorboard
    writer = SummaryWriter(f"runs/{MODEL_TYPE}-{time_stamp}")

    # Train network
    net, logs = train_loop(
        net, MODEL_TYPE, optimizer, criterion, train_data, valid_data, save_dir=save_dir, n_epochs=N_EPOCHS, \
            batch_size=BATCH_SIZE, verbose=True, scheduler=scheduler, save_every=args.save_every, \
            eval_every=args.eval_every, clip=args.clip, writer=writer, accum_num=args.accum_num,
    )

    # Save Model
    torch.save(net, os.path.join(save_dir, 'model.pt'))

    # Save Logs
    log_df = pd.DataFrame(logs)
    log_df.to_csv(os.path.join(save_dir, 'logs.csv'))
