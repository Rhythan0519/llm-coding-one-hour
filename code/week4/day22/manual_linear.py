import torch
import torch.nn.functional as F

def manual_linear(x, weight, bias=None):
    """
    Args:
        x: [B, in_features]
        weight: [out_features, in_features]
        bias: [out_features] or None

    Returns:
        y: [B, out_features]
    """
    assert x.dim() == 2
    assert weight.dim() == 2
    assert x.shape[-1] == weight.shape[-1]

    y = x @ weight.T
    if bias is not None:
        y = y + bias
    
    return y

def test():
    torch.manual_seed(0)
    x = torch.randn(4, 3)
    weight = torch.randn(5, 3)
    bias = torch.randn(5)

    y = manual_linear(x, weight, bias)
    ref = F.linear(x, weight, bias)

    assert y.shape == (4, 5)
    assert torch.allclose(y, ref, atol=1e-6)

    y_no_bias = manual_linear(x, weight, None)
    ref_no_bias = F.linear(x, weight, None)
    assert torch.allclose(y_no_bias, ref_no_bias, atol=1e-6)

    print("Input shape:", x.shape)
    print("Weight shape:", weight.shape)
    print("Output shape:", y.shape)
    print("Test passed.")


if __name__ == "__main__":
    test()