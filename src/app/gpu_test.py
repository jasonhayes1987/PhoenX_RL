import os
import time
import torch
import torch.nn as nn
from torch import optim  # Explicit import to fix linter
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

# Confirm CUDA availability and force clear cache
print("CUDA Available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
torch.cuda.empty_cache()  # Clear any lingering memory

log_dir = "/workspaces/RL_Agents/src/app/torch_profiler_logs"
os.makedirs(log_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Even heavier model (multi-conv to simulate vision RL, e.g., ViT-like)
model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Linear(1024, 1024)  # Adjusted in_features to match tensor sizes
).to(device)
optimizer = optim.Adam(model.parameters())  # Use optim.Adam for linter # pylint: disable=E1101  # Suppress linter

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=10, repeat=3),  # Even longer active for dominant GPU time
    on_trace_ready=tensorboard_trace_handler(log_dir),
    record_shapes=True,
    profile_memory=True,
    with_stack=True  # For detailed call stacks
) as prof:
    for i in range(100):  # Way more iterations to dominate with GPU work
        # Larger batch for conv input
        a = torch.randn(4, 3, 1024, 1024, device=device)  # Bigger batch/images for heavier conv load
        b = torch.randn(1024, 1024, device=device)
        conv_out = model[0:4](a)  # Multi-conv forward (very GPU-heavy)
        flattened = conv_out.view(4, -1)[:, :1024]  # Slice to match b's rows (4x1024)
        loss = (flattened @ b).sum() + model[4](b).sum()  # Matrix mul + linear forward
        loss.backward()  # Simulate backprop like in agent.learn()
        optimizer.step()
        torch.cuda.synchronize()  # Force GPU sync to ensure kernels complete and are measured
        prof.step()  # Advance profiler step
        time.sleep(0.005)  # Shorter delay to keep it running but spaced

print(f"Traces exported to: {log_dir}")
os.system(f"ls {log_dir}")  # List generated files for verification