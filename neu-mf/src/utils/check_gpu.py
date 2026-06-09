import torch

def check_gpu():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"Device: {device} ({gpu.name}, {gpu.total_memory // 1024**2} MB VRAM)")
    else:
        print(f"Device: {device}")
    return device

