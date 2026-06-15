import torch
import torch.nn as nn

from layer_versions import FFNv4 as FFN


def generate_parameters(dim, G, dtype, expansion=4, device="cuda"):
    hdim = dim * expansion
    W1 = torch.empty(hdim, dim, device=device, dtype=dtype)
    torch.nn.init.xavier_uniform_(W1, generator=G)

    W2 = torch.empty(dim, hdim, device=device, dtype=dtype)
    torch.nn.init.xavier_uniform_(W2, generator=G)

    W1 = W1 + 0.01*W1.std()
    W2 = W2 #+ W2.std()
    return W1, W2


class DeepFFN(nn.Module):
    def __init__(self, layers, hidm, dtype):
        super().__init__()
        G = torch.Generator(device="cuda").manual_seed(0)

        self.W1s, self.W2s = nn.ParameterList(), nn.ParameterList()
        for i in range(layers):
            W1, W2 = generate_parameters(hidm, G, dtype=dtype)

            self.W1s.append(nn.Parameter(W1))
            self.W2s.append(nn.Parameter(W2))

    def forward(self, x):
        """ x.shape = [BS, dim] """
        for W1, W2 in zip(self.W1s, self.W2s):
            x = x + FFN.apply(x, W1, W2)
        return x


def evaluate_step():
    layers = 12
    hdim = 4096
    dtype = torch.bfloat16

    model = DeepFFN(layers, hdim, dtype=dtype)
    x = torch.randn(10_000, hdim, dtype=dtype, device="cuda")

    y = model(x)
    loss = (y - x).pow(2).mean()
    print(y.std())
    print(f'Loss = {loss.detach().item()}, y.std = {y.std().detach().item()}')

    allocated = torch.cuda.memory_allocated("cuda")
    print(f"VRAM allocated by tensors: {allocated / 1024**2:.2f} MB")

    loss.backward()


def run_base():
    torch.set_float32_matmul_precision("high")
    # torch.set_printoptions(precision=10)
    torch.manual_seed(0)
    evaluate_step()


if __name__ == "__main__":
    run_base()
