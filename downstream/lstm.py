import torch
from torch import nn

class RNN(nn.Module):
    def __init__(self, output_size, rnn_params, embedding_dim=300, device='cpu'):
        super().__init__()
        self.output_size = output_size
        self.device = device
        self.embedding_dim = embedding_dim
            
        self.rnn = nn.LSTM(self.embedding_dim, **rnn_params).to(device)
        rnn_out_size = self.rnn.hidden_size * 2 if self.rnn.bidirectional else self.rnn.hidden_size
        self.linear = nn.Linear(rnn_out_size, output_size).to(device)
        self.dropout = nn.Dropout(0.5).to(device)

    def forward(self, sents):
        self.rnn.flatten_parameters()
        sents = torch.tensor(sents, dtype=torch.float32).to(self.device)
        h_t, _ = self.rnn(sents)

        # Pool and pass through a FF layer before outputting prediction
        out = torch.max(h_t, dim=1)[0]
        out = self.dropout(out)
        out = self.linear(out)
        out = torch.sigmoid(out).squeeze()

        return out