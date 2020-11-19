import argparse
import os
import time
import sys

import torch
import torch.optim as optim
from torch import nn
from transformers import ElectraTokenizer
from utils import load_mimic_mortality, load_mimic_diagnosis, load_model, predict, evaluate, train_loop
from lstm import RNN
from lstm_sa import RNNAtt
from electra import Electra

from tokenizer_and_dataset import FastTextTokenizer, MimicDataset, HuggingfaceDataset

from torch.utils.tensorboard import SummaryWriter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", help="Directory for storing experiments", required=True)
    parser.add_argument("--model", help="Type of model", \
        choices=['rnnsoft', 'electra', 'rnn'], required=True)
    parser.add_argument("--task", help="Downstream task", \
        choices=['mimic-mortality', 'mimic-diagnosis'], required=True)
    parser.add_argument("--test", help="Test model on test set", action='store_true', default=False)

    parser.add_argument("--data_dir", help="Root path for train/valid/test data files", required=True)
    parser.add_argument("--data_filename", \
        help="Data file name (train/valid/test should have same name, in different dirs)", required=True)
    parser.add_argument("--diag_to_idx_path", required=False, help="Path to the diag_to_idx file")
    parser.add_argument("--embs_path", help="Path to pretrained Fasttext embeddings")

    parser.add_argument("--pretrained_model", help="Path to previously trained model")

    parser.add_argument("--use_scheduler", help="Whether to use lr scheduler", action="store_true")
    parser.add_argument("--lr", help="Learning rate", default=0.001, type=float)
    parser.add_argument("--dropout", help="Drop out rate", default=0.1, type=float)
    parser.add_argument("--epochs", help="Number of epochs", default=5, type=int)
    parser.add_argument("--save_every", help="Frequency of checkpoint", default=40, type=int)
    parser.add_argument("--eval_every", help="Frequency of evaluation", default=10000, type=int)
    parser.add_argument("-bs", "--batchsize", help="LSTM Batch Size", default=128, type=int)

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
    TASK = args.task
    TEST = args.test
    MODEL_TYPE = args.model
    USE_PRETRAIN = True if args.pretrained_model else False

    if TEST and not USE_PRETRAIN:
        raise Exception("no model preovided for testing")

    if not USE_PRETRAIN:
        print("No pretrained model provided. Will train from scratch.")

    # Prelim
    torch.set_num_threads(N_CPU_CORES)
    DEVICE = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    if TASK in ['mimic-mortality']:
        train, valid, test = load_mimic_mortality(args.data_dir, args.data_filename)
    elif TASK in ['mimic-diagnosis']:
        train, valid, test, diag_to_idx = \
            load_mimic_diagnosis(args.data_dir, args.data_filename, args.diag_to_idx_path)
    print("Data loaded")

    # Create tokenizer objects
    if MODEL_TYPE == "electra":
        tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
    else:
        # Create word index and load Fasttext embedding matrix
        tokenizer = FastTextTokenizer(verbose=True)
        tokenizer.build_word_index(train.TEXT, valid.TEXT, test.TEXT)
        tokenizer.build_embedding_matrix(args.embs_path)

    if TASK in ['mimic-mortality']:
        output_size = 1
        label_col = 'LABEL_NUM'
    elif TASK in ['mimic-diagnosis']:
        output_size = len(diag_to_idx)
        label_col = 'DIAG'

    # Create torch Dataset objects
    if MODEL_TYPE in ["rnnsoft", "rnn"]:
        if TEST:
            test_data = MimicDataset(test, tokenizer=tokenizer, task=TASK, label_col=label_col, output_size=output_size, device=DEVICE)
        else:
            train_data = MimicDataset(train, tokenizer=tokenizer, task=TASK, label_col=label_col, output_size=output_size, device=DEVICE)
            valid_data = MimicDataset(valid, tokenizer=tokenizer, task=TASK, label_col=label_col, output_size=output_size, device=DEVICE)
    else:
        if TEST:
            test_data = HuggingfaceDataset(test, tokenizer=tokenizer, task=TASK, label_col=label_col, output_size=output_size, device=DEVICE)
        else:
            train_data = HuggingfaceDataset(train, tokenizer=tokenizer, task=TASK, label_col=label_col, output_size=output_size, device=DEVICE)
            valid_data = HuggingfaceDataset(valid, tokenizer=tokenizer, task=TASK, label_col=label_col, output_size=output_size, device=DEVICE)
    print("Dataset created")

    # Define network, loss function and optimizer
    if MODEL_TYPE == "rnn":
        LSTM_PARAMS = dict(
            batch_first=True,
            num_layers=args.rnn_layers,
            dropout=args.dropout,
            bidirectional=True,
            hidden_size=args.hidden_size
        )
        net = RNN(
            embedding_dim=tokenizer.embedding_matrix.shape[1],
            output_size=output_size,
            rnn_params=LSTM_PARAMS,
            device=DEVICE
        )
        net.to(DEVICE)
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs")
            net = nn.DataParallel(net)
        if USE_PRETRAIN:
            net = load_model(net, args.pretrained_model, DEVICE)
    elif MODEL_TYPE == "rnnsoft":
        net = RNNAtt(
            rnn_layers=args.rnn_layers,
            da_layers=args.da_layers,
            output_size=output_size,
            d_model=args.hidden_size,
            device=DEVICE,
            dropout_rate=args.dropout,
        )
        net.to(DEVICE)
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs")
            net = nn.DataParallel(net)
        if USE_PRETRAIN:
            net = load_model(net, args.pretrained_model, DEVICE)
    elif MODEL_TYPE == "electra":
        net = Electra(
            output_size=output_size,
            device=DEVICE,
        )
        if USE_PRETRAIN:
            net = load_model(net, args.pretrained_model, DEVICE)

    print('model: {}'.format(net))
    if TASK in ['mimic-mortality', 'mimic-diagnosis']:
        criterion = nn.BCELoss()
    if not TEST:
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

    if not TEST:
        writer = SummaryWriter(f"runs/{TASK}/{MODEL_TYPE}-{time_stamp}")
        # Train network
        net, logs = train_loop(
            net, optimizer, criterion, train_data, valid_data, save_dir=save_dir, task=TASK, n_epochs=N_EPOCHS, \
                batch_size=BATCH_SIZE, verbose=True, scheduler=scheduler, save_every=args.save_every, \
                eval_every=args.eval_every, writer=writer,
        )
    else:
        # Test
        logs = {k: [] for k in ['test_loss', 'test_metric']}
        if TASK == 'mimic-diagnosis':
            logs['test_top_5_recall'] = []
            logs['test_top_30_recall'] = []
        test_loader = DataLoader(
            range(len(test)), 
            shuffle=False, 
            batch_size=BATCH_SIZE
        )
        if TASK == 'mimic-mortality':
            test_preds = predict(net, test_loader, test_data, verbose=True).cpu().numpy()
            np.save(os.path.join(save_dir, 'test_preds.npy'), test_preds)
            test_loss, test_metric = evaluate(net, test_loader, test_data, criterion, verbose=True, task=TASK)
        elif TASK == 'mimic-diagnosis':
            test_loss, test_metrics = evaluate(net, test_loader, test_data, criterion, verbose=True, task=TASK)
            test_metric = test_metrics['top_10_recall']
        print(f"Test Loss: {test_loss:.4f} \tTest Metric:{test_metric:.4f}")
        if TASK == 'mimic-diagnosis':
            print(f"Test Top 5 Recall: {test_metrics['top_5_recall']:.4f} \tTest Top 30 Recall:{test_metrics['top_30_recall']:.4f}")
        print("="*50)
        logs['test_loss'].append(test_loss)
        logs['test_metric'].append(test_metric)
        if TASK == 'mimic-diagnosis':
            logs['test_top_5_recall'].append(test_metrics['top_5_recall'])
            logs['test_top_30_recall'].append(test_metrics['top_30_recall'])

    # Save Model
    if not TEST:
        torch.save(net.state_dict(), os.path.join(save_dir, 'model.pt'))

    # Save Logs
    log_df = pd.DataFrame(logs)
    log_df.to_csv(os.path.join(save_dir, 'logs.csv'))