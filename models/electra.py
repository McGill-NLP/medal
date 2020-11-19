import torch
from torch import nn
from transformers import AutoModel, ElectraConfig

# ELECTRA
class Electra(nn.Module):
    def __init__(self, output_size=24005, device='cpu'):
        super().__init__()
        self.device = device
        config = ElectraConfig.from_pretrained('google/electra-small-discriminator')
        self.electra = AutoModel.from_config(config).to(device)
        self.output = nn.Linear(self.electra.config.hidden_size, output_size).to(device)

    def forward(self, sents, locs):
        sents = torch.tensor(sents).to(self.device)
        sents = self.electra(sents)[0]
        abbs = torch.stack([sents[n, idx, :] for n, idx in enumerate(locs)])  # (B * M)
        return self.output(abbs)