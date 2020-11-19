import torch
import torch.nn.functional as F
from torch import nn

class AttentionModule(nn.Module):
    def __init__(self, d_model, d_k=None, device='cpu', dropout=None):
        super().__init__()
        if not d_k:
            d_k = d_model
        # self.sqrt_d_k = math.sqrt(d_k)
        self.W = nn.Parameter(torch.randn(d_model, d_model, device=device))    # (M * M)
        self.bias = nn.Parameter(torch.randn(1, device=device))
        self.dropout = dropout

    def forward(self, key, query, value):
        if query.shape[0] > 1:  # not using a single vector as query for all samples in batch
            query_W_key = torch.bmm(torch.matmul(query, self.W), key.transpose(-2, -1)) # (B * N * N)
        else:
            query_W_key = torch.matmul(key, torch.matmul(query, self.W).transpose(-2, -1)).transpose(-2, -1)
        if self.dropout:
            query_W_key = self.dropout(query_W_key)
        weights = F.softmax(torch.tanh(query_W_key + self.bias), dim=-1)  # (B * N * N)
        return weights, torch.bmm(weights, value)

class RNNAtt(nn.Module):
    def __init__(self, rnn_layers, da_layers, output_size, embedding_dim=300, d_model=512, dropout_rate=0.1, device='cpu'):
        super().__init__()
        self.device = device
        self.rnn = nn.LSTM(embedding_dim, hidden_size=d_model, bidirectional=True, \
                                    dropout=dropout_rate, num_layers=rnn_layers, batch_first=True).to(device)
        rnn_out_size = d_model * 2             
        self.dropout = nn.Dropout(p=dropout_rate)       
        self.attentions = nn.ModuleList([AttentionModule(d_model=rnn_out_size, d_k=rnn_out_size, device=device, dropout=self.dropout) \
                                            for _ in range(da_layers)])                            
        self.cls_query = nn.Parameter(torch.randn(1, rnn_out_size, device=device)) # a learnable vector acting as query for output att
        self.cls_att = AttentionModule(d_model=rnn_out_size, d_k=rnn_out_size, device=device, dropout=self.dropout)
        self.output = nn.Linear(rnn_out_size, output_size).to(device)

    def forward(self, sents):
        sents = torch.tensor(sents, dtype=torch.float32).to(self.device)
        self.rnn.flatten_parameters()
        sents, _ = self.rnn(sents)
        for layer in self.attentions:
            _, sents = layer(sents, sents, sents)
        _, out = self.cls_att(key=sents, query=self.cls_query, value=sents)
        # out = self.dropout(torch.max(sents, dim=1)[0])    # try max pooling
        return torch.sigmoid(self.output(out)).squeeze()