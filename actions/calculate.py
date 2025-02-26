import torch
import numpy as np

def compute_area(mask):
    """Calculate white pixel area from a binary tensor mask."""
    if isinstance(mask, np.ndarray):  # Convert NumPy to PyTorch tensor
        mask = torch.from_numpy(mask)
    return torch.sum(mask > 0).item()  # Count nonzero pixels

def compute_volume(area_2c, area_4c, scaling_factor=1.0):
    """Compute left ventricle volume using Simpson’s Biplane method."""
    area_2c = torch.tensor(area_2c, dtype=torch.float32)  # Ensure tensor format
    area_4c = torch.tensor(area_4c, dtype=torch.float32)
    return (8 / (3 * torch.pi)) * torch.sqrt(area_2c * area_4c) * scaling_factor


def compute_ef(es_mask_2c, ed_mask_2c,es_mask_4c,ed_mask_4c):
# Example tensor masks (binary tensors of shape [H, W])
# Replace these with your actual PyTorch tensor masks
# es_mask_2c = torch.randint(0, 2, (256, 256))  # Simulated 2C ES mask
# ed_mask_2c = torch.randint(0, 2, (256, 256))  # Simulated 2C ED mask
# es_mask_4c = torch.randint(0, 2, (256, 256))  # Simulated 4C ES mask
# ed_mask_4c = torch.randint(0, 2, (256, 256))  # Simulated 4C ED mask

# Compute areas
    ed_area_2c = compute_area(ed_mask_2c)
    es_area_2c = compute_area(es_mask_2c)
    ed_area_4c = compute_area(ed_mask_4c)
    es_area_4c = compute_area(es_mask_4c)

    # Scaling factor to convert pixel area to real-world units (adjust based on image resolution)
    scaling_factor = 0.1  # Example value, modify accordingly

    # Compute volumes
    edv = compute_volume(ed_area_2c, ed_area_4c, scaling_factor)
    esv = compute_volume(es_area_2c, es_area_4c, scaling_factor)

    # Calculate Ejection Fraction
    ef = ((edv - esv) / edv) * 100
    print(f"Ejection Fraction: {ef:.2f}%")
    return ef # Output: Ejection Fraction in percentage
