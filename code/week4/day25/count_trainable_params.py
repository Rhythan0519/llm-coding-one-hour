import torch
import torch.nn as nn
def count_parameters(model):
    """
    Returns:
        stats: dict with total/trainable/frozen/trainable_ratio
    """
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

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(4, 8)
        self.b = nn.Linear(8, 2)

def test():
    model = TinyModel()
    for p in model.a.parameters():
        p.requires_grad = False

    stats = count_parameters(model)

    expected_total = sum(p.numel() for p in model.parameters())
    expected_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert stats["total"] == expected_total
    assert stats["trainable"] == expected_trainable
    assert stats["frozen"] == stats["total"] - stats["trainable"]
    assert 0.0 <= stats["trainable_ratio"] <= 1.0

    print("Total params:", stats["total"])
    print("Trainable params:", stats["trainable"])
    print("Trainable ratio:", stats["trainable_ratio"])
    print("Test passed.")


if __name__ == "__main__":
    test()