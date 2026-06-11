import torch
from torch import nn
import torch.nn.functional as F


class ModelBase(nn.Module):
    def __init__(self, W1, W2):
        super().__init__()

        self.W1 = torch.nn.Parameter(W1.clone())
        self.W2 = torch.nn.Parameter(W2.clone())


    def forward(self, x):
        x = F.linear(x, self.W1)
        x = F.relu(x)
        x = F.linear(x, self.W2)
        return x


def generate_parameters(dim, expansion, seed, device="cuda"):
    """ Fixed weights and inputs for consistency """
    G = torch.Generator(device=device).manual_seed(seed)

    hdim = dim * expansion
    W1 = torch.empty(hdim, dim, device=device)
    W2 = torch.empty(dim, hdim, device=device)

    torch.nn.init.xavier_uniform_(W1, generator=G)
    torch.nn.init.xavier_uniform_(W2, generator=G)

    x = torch.randn(10_000, dim, device=device, generator=G)
    y = torch.randn(10_000, dim, device=device, generator=G)
    return W1, W2, x, y


def exact_solution(W1, W2, x, y):
    model = ModelBase(W1, W2)
    model.to("cuda")

    y_hat = model(x)
    loss = (y_hat - y).pow(2).mean()
    loss.backward()

    W1_g = model.W1.grad.detach().clone()
    W2_g = model.W2.grad.detach().clone()
    return y_hat, W1_g, W2_g


def run_base():
    dim, expansion = 2048, 4
    W1, W2, x, y = generate_parameters(dim, expansion, 1)
    preds, W1_g, W2_g = exact_solution(W1, W2, x, y)

    print(preds[0])
    print(W1_g[0])
    print(W2_g[0])


if __name__ == "__main__":
    run_base()