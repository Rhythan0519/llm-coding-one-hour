import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, 
                 rank=2, alpha=1.0, bias=True):
        assert rank >= 0
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.base = nn.Linear(in_features, out_features, bias=bias)
        for p in self.base.parameters():
            p.requires_grad_(False)
        if rank > 0:
            self.lora_A = nn.Parameter(torch.empty(rank, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, rank))
            nn.init.normal_(self.lora_A, mean=0.0, std=0.01)
            nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        base_output = self.base(x)
        if self.rank == 0:
            return base_output
        else:
            lora_output = F.linear(F.linear(x, self.lora_A), self.lora_B)
            return (lora_output + base_output) * (self.alpha / self.rank)
        
def train_lora_on_toy_data():
    torch.manual_seed(0)
    x = torch.randn(32, 4)
    target = torch.randn(32, 6)

    layer = LoRALinear(4, 6, rank=2, alpha=4.0)
    trainable_params = [p for p in layer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=0.1)
