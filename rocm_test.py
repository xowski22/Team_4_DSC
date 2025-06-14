import torch

print("Torch version: ", torch.__version__)
print("Cuda: ", torch.cuda.is_available())
print("ROCM: ", torch.version.hip if hasattr(torch.version, 'hip') else "Not detected")


if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    print("GPU name:", torch.cuda.get_device_name(0))
    print("GPU memory:", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    x = torch.randn(1000, 1000).cuda()
    y = torch.mm(x, x.t())
    print("GPU tensor operations work")
