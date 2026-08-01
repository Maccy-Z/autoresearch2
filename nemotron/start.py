import os
import sys
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB_SPARSE_ROOT = os.path.join(PROJECT_ROOT, "lib_sparse")
for path in (PROJECT_ROOT, LIB_SPARSE_ROOT):
    if path not in sys.path:
        sys.path.append(path)
os.environ["TRITON_DEBUG"] = "1"

import torch
from transformers import AutoTokenizer
import time
from cprint import c_print

from utils import print_max_memory
from llm import NemotronHForCausalLM
from lib_sparse.shared.utils import TensorBuffer

# MODEL_NAME = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
MODEL_NAME = "nvidia/Nemotron-H-8B-Base-8K"

MAX_TRAIN_TOKENS = 1000
with open("sample_text.txt", "r") as f:
    prompt = f.read()


def setup_hooks(model):
    def hook(w):
        # print(w.grad.norm())
        w.grad = None
        return

    for n, p in model.named_parameters():
        p.register_post_accumulate_grad_hook(hook)


def calculate_loss(model: NemotronHForCausalLM, text, tokenizer, device, max_tokens=MAX_TRAIN_TOKENS):
    tokenizer_kwargs = {"return_tensors": "pt"}
    if max_tokens is not None:
        tokenizer_kwargs.update(
            {
                "max_length": max_tokens,
                "truncation": True,
            }
        )

    inputs = tokenizer(text, **tokenizer_kwargs).to(device)
    assert inputs["input_ids"].shape[-1] == max_tokens
    # For causal LM training, labels are usually the same token IDs as inputs.
    # The model shifts them internally when computing next-token loss.
    labels = inputs["input_ids"].clone()

    outputs = model(**inputs, labels=labels, use_cache=False)
    return outputs.loss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model: NemotronHForCausalLM = NemotronHForCausalLM.from_pretrained(
        MODEL_NAME, dtype=dtype, trust_remote_code=True).to(device)
    setup_hooks(model)
    model.train()

    model.config.sparse_ffn = True
    sparse_data = TensorBuffer(40_000_000)
    sparse_data.init_buffer()
    # sparse_data = None
    model.config.sparse_data = sparse_data

    # Warmup
    c_print("Starting Warmup", color="cyan")
    for _ in range(5):
        loss = calculate_loss(model, prompt, tokenizer, device, max_tokens=MAX_TRAIN_TOKENS)
        loss.backward()
        model.zero_grad()
        if sparse_data is not None:
            sparse_data.reset_buffer()

    # Timing
    c_print("Starting Timing Run", color="cyan")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    st = time.perf_counter()
    loss = calculate_loss(model, prompt, tokenizer, device, max_tokens=MAX_TRAIN_TOKENS)
    # print_max_memory("After forward pass")

    loss.backward()
    torch.cuda.synchronize()
    print_max_memory("After backward pass")
    et = time.perf_counter()

    print(f"Total Time: {et - st:.4f} seconds")

    # Validation
    print("-"*50)
    print(f'Loss: {loss.detach().cpu() = }')

    if sparse_data is not None:
        if sparse_data.offset > sparse_data.size:
            c_print(f"Warning: Too many values detected, sparse_data.offset={sparse_data.offset.cpu().item()}, {sparse_data.size = }. "
                    f"Results may be incorrect and the program may crash unexpectedly.", color="bright_red")

if __name__ == "__main__":
    torch.set_printoptions(precision=6)
    main()
