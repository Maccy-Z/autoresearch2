import torch

x = torch.randn(512, 512, device="cuda")
while True:
    x @ x

