import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, embedding_dim) * 0.02)

    def forward(self, token_ids):
        embeddings = self.weight.data[token_ids, :]
        return embeddings
    
def test():
    torch.manual_seed(0)
    token_ids = torch.tensor([[1, 4, 3], [0, 2, 2]], dtype=torch.long)
    mine = Embedding(10, 8)
    ref = nn.Embedding(10, 8)
    ref.weight.data.copy_(mine.weight.data)

    my_out = mine(token_ids)
    ref_out = ref(token_ids)

    assert my_out.shape == (2, 3, 8)
    assert torch.allclose(my_out, ref_out, atol=1e-6)
    assert token_ids.dtype == torch.long

    print("Input shape:", token_ids.shape)
    print("Output shape:", my_out.shape)
    print("Test passed.")

if __name__ == "__main__":
    test()