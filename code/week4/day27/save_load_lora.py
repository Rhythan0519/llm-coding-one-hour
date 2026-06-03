import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, 
                 rank=2, alpha=1.0, bias=True):
        super().__init__()
        assert rank >= 0
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.base = nn.Linear(in_features, out_features, bias=bias)
        for p in self.base.parameters():
            p.requires_grad_(False)

        if rank > 0:
            self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features))
            self.lora_B = nn.Parameter(torch.empty(self.out_features, self.rank))
            nn.init.normal_(self.lora_A, mean=0.0, std=0.01)
            nn.init.zeros_(self.lora_B)
    def forward(self, x):
        base_output = self.base(x)
        if self.rank == 0:
            return base_output
        else:
            lora = F.linear(F.linear(x, self.lora_A), self.lora_B) * (self.alpha / self.rank)
            return lora + base_output
        
def get_lora_state_dict(model):
    lora_state = {}
    for name, tensor in model.state_dict().items():
        if "lora_" in name:
            lora_state[name] = tensor
    return lora_state

def load_lora_state_dict(model, lora_state):
    state_dict = model.state_dict()
    state_dict.update(lora_state)
    model.load_state_dict(state_dict)
    return model

def test():
    torch.manual_seed(0)
    x = torch.randn(4, 3)
    model1 = LoRALinear(3, 5, rank=2, alpha=4.0)
    model2 = LoRALinear(3, 5, rank=2, alpha=4.0)
    model2.base.load_state_dict(model1.base.state_dict())
    with torch.no_grad():
        model1.lora_A.fill_(0.1)
        model1.lora_B.fill_(0.2)
    lora_state = get_lora_state_dict(model1)
    load_lora_state_dict(model2, lora_state)
    y1 = model1(x)
    y2 = model2(x)

    assert len(lora_state) > 0
    assert all("lora_" in key for key in lora_state.keys())
    assert torch.allclose(y1, y2, atol=1e-6)

    print("Input shape:", x.shape)
    print("Num LoRA tensors:", len(lora_state))
    print("Output shape:", y1.shape)
    print("Test passed.")


if __name__ == "__main__":
    test()
