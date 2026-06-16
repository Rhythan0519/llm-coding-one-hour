def patchify(images, patch_size: int):
    assert patch_size > 0
    B, C, H, W = images.shape
    assert H % patch_size == 0 and W % patch_size == 0
    grid_h = H // patch_size
    grid_w = W // patch_size
    patch_dim = patch_size * patch_size * C
    patches = images.reshape(B, C, grid_h, patch_size, grid_w, patch_size)
    patches = patches.permute(0, 2, 4, 3, 5, 1)
    patches = patches.reshape(B, grid_h * grid_w, patch_dim)
    return patches

def unpatchify(patches, patch_size, channels, height, width):
    B, N, _ = patches.shape
    grid_h = height // patch_size
    grid_w = width // patch_size
    patches = patches.reshape(B, grid_h, grid_w, patch_size, patch_size, channels)
    patches = patches.permute(0, 5, 1, 3, 2, 4)
    images = patches.reshape(B, channels, height, width)
    return images