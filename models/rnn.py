import torch
from torch import nn

default_rnn_params = dict(
    batch_first=True,
    num_layers=3,
    dropout=0.5,
    bidirectional=True,
    hidden_size=512
)

class RNN(nn.Module):
    def __init__(self, embedding_dim=300, output_size=22555, rnn_params=default_rnn_params, device='cpu'):
        super().__init__()
        self.output_size = output_size
        self.device = device
        self.embedding_dim = embedding_dim
            
        self.rnn = nn.LSTM(self.embedding_dim, **rnn_params).to(device)
        rnn_out_size = self.rnn.hidden_size * 2 if self.rnn.bidirectional else rnn.hidden_size
        self.linear = nn.Linear(rnn_out_size, output_size).to(device)

    def forward(self, sents, locs):
        sents = torch.tensor(sents, dtype=torch.float32).to(self.device)
        
        # Run LSTM
        self.rnn.flatten_parameters()
        h_t, _ = self.rnn(sents)
        
        # Locate the abbreviations, this is what we will be predicting
        abvs = torch.stack([
            h_t[n, idx, :] 
            for n, idx in enumerate(locs)
        ])
        out = self.linear(abvs)

        return out