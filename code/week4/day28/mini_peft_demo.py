import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=2, alpha=1.0, bias=True):
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
        lora_output = F.linear(F.linear(x, self.lora_A), self.lora_B) * (self.alpha / self.rank)
        return base_output + lora_output

class TinyLoRAClassifier(nn.Module):
    def __init__(self, in_features, num_classes, rank=2):
        super().__init__()
        self.classifier = LoRALinear(in_features, num_classes, rank=rank)
    
    def forward(self, x):
        return self.classifier(x)
    
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

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    ratio = trainable / total
    stats = {
        "total": total, 
        "trainable": trainable, 
        "frozen": frozen, 
        "trainable_ratio": ratio, 
    }
    return stats

def run_mini_peft_demo():
    torch.manual_seed(0)
    x = torch.randn(64, 4)
    labels = torch.randint(0, 3, (64, ))
    model = TinyLoRAClassifier(4, 3, rank=2)
    stats = count_parameters(model)
    with torch.no_grad():
        logits_before = model(x)
        loss_before = F.cross_entropy(logits_before, labels)
    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]
    optimizer = torch.optim.Adam(trainable_params, lr=0.1)
    for _ in range(20):
        logits = model(x)
        loss = F.cross_entropy(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits_after = model(x)
        loss_after = F.cross_entropy(logits_after, labels)
    lora_state = get_lora_state_dict(model)
    model2 = TinyLoRAClassifier(4, 3, rank=2)
    model2.classifier.base.load_state_dict(
        model.classifier.base.state_dict()
    )
    load_lora_state_dict(model2, lora_state)
    with torch.no_grad():
        y1 = model(x)
        y2 = model2(x)
        reload_same = torch.allclose(y1, y2, atol=1e-6)

    return loss_before, loss_after, stats, reload_same

def test():
    result = run_mini_peft_demo()
    loss_before, loss_after, stats, reload_same = result

    assert loss_after < loss_before
    assert stats["trainable_ratio"] < 1.0
    assert reload_same is True

    print("Loss before:", float(loss_before))
    print("Loss after:", float(loss_after))
    print("Trainable ratio:", stats["trainable_ratio"])
    print("Reload same:", reload_same)
    print("Test passed.")


if __name__ == "__main__":
    test()