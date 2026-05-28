import torch

class KVCache:
    def __init__(self):
        self.k = None
        self.v = None

    def update(self, new_k, new_v):
        """
        Args:
            new_k: [B, H, 1, D]
            new_v: [B, H, 1, D]

        Returns:
            k: [B, H, T, D]
            v: [B, H, T, D]
        """
        assert new_k.shape == new_v.shape
        assert new_k.dim() == 4
        assert new_k.size(2) == 1

        if self.k is None or self.v is None:
            self.k = new_k
            self.v = new_v
        else:
            self.k = torch.cat([self.k, new_k], dim=2)
            self.v = torch.cat([self.v, new_v], dim=2)
        return self.k, self.v

    def reset(self):
        self.k, self.v = None, None
    @property
    def length(self):
        if self.k is None:
            length = 0
        else:
            length = self.k.shape[2]
        return length
    
def test():
    torch.manual_seed(0)
    cache = KVCache()

    assert cache.length == 0

    for step in range(3):
        new_k = torch.randn(2, 4, 1, 8)
        new_v = torch.randn(2, 4, 1, 8)
        k, v = cache.update(new_k, new_v)

        assert k.shape == (2, 4, step + 1, 8)
        assert v.shape == (2, 4, step + 1, 8)
        assert cache.length == step + 1
        assert torch.allclose(k[:, :, -1:, :], new_k)
        assert torch.allclose(v[:, :, -1:, :], new_v)

    cache.reset()
    assert cache.k is None
    assert cache.v is None
    assert cache.length == 0

    print("Last K shape:", k.shape)
    print("Last V shape:", v.shape)
    print("Cache length after reset:", cache.length)
    print("Test passed.")


if __name__ == "__main__":
    test()