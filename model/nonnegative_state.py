"""Hard physical C >= 0 projection used inside every recurrent update."""
import torch


def project_concentration(z, mask, norm):
    s = z.new_tensor(norm['concentration']['std'][0])
    m = z.new_tensor(norm['concentration']['mean'][0])
    lower = -m / s
    lower = torch.where(lower*s+m < 0, torch.nextafter(lower, lower.new_tensor(float('inf'))), lower)
    return z.clamp_min(lower) * mask
