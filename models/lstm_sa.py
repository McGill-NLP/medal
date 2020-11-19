import torch
import torch.nn.functional as F
from torch import nn

# RNN + soft attention
class AttentionModule(nn.Module):
    def __init__(self, d_model, d_k=None, device='cpu', dropout=None):
        super().__init__()
        if not d_k:
            d_k = d_model
        self.W = nn.Parameter(torch.randn(d_model, d_model, device=device))    # (M * M)
        self.bias = nn.Parameter(torch.randn(1, device=device))
        self.dropout = dropout
        self.norm = nn.LayerNorm(d_model).to(device)

    def forward(self, key, query, value):
        key = self.norm(key)
        query = self.norm(query)
        value = self.norm(value)
        # key: (B * N * M) -> (B * N * d_k)
        # query: (B * N * M) -> (B * N * d_k)
        # value: (B * N * M) -> (B * N * d_k)
        query_W_key = torch.bmm(torch.matmul(query, self.W), key.transpose(-2, -1)) # (B * N * N)
        if self.dropout:
            query_W_key = self.dropout(query_W_key)
        weights = F.softmax(torch.tanh(query_W_key + self.bias), dim=-1)  # (B * N * N)
        return weights, torch.bmm(weights, value)

class RNNAtt(nn.Module):
    def __init__(self, rnn_layers=3, da_layers=1, output_size=22555, embedding_dim=300, d_model=512, dropout_rate=0.1, device='cpu'):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.rnn = nn.LSTM(embedding_dim, hidden_size=d_model, bidirectional=True, \
                                    dropout=dropout_rate, num_layers=rnn_layers, batch_first=True).to(device)
        rnn_out_size = d_model * 2             
        self.dropout = nn.Dropout(p=dropout_rate)       
        self.norm = nn.LayerNorm(rnn_out_size).to(device)
        self.attentions = nn.ModuleList([AttentionModule(d_model=rnn_out_size, d_k=rnn_out_size, device=device, dropout=self.dropout) \
                                            for _ in range(da_layers)])                            

        self.output = nn.Linear(rnn_out_size, output_size).to(device)

    def forward(self, sents, locs):
        sents = torch.tensor(sents, dtype=torch.float32).to(self.device)
        self.rnn.flatten_parameters()
        sents, _ = self.rnn(sents)
        for layer in self.attentions:
            sents = self.norm(sents)
            _, sents = layer(sents, sents, sents)
        abbs = torch.stack([sents[n, idx, :] for n, idx in enumerate(locs)])  # (B * M)
        return self.output(abbs)