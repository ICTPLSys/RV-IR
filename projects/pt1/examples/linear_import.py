import os
import sys

home_dir = os.path.expanduser("~")
python_paths = [
    os.path.join(home_dir, "RV-IR/build/tools/torch-mlir/python_packages/torch_mlir"),
    os.path.join(home_dir, "RV-IR/projects/pt1/python/"),
    os.path.join(home_dir, "RV-IR/projects/pt1/python/torch_mlir"),
]

for p in python_paths:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ["PYTHONPATH"] = os.pathsep.join(python_paths + sys.path)


import torch
import torch.nn as nn
from torch_mlir import torchscript

class LinearModel(nn.Module):
    def __init__(self, in_dim=2048, out_dim=512):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(out_dim, in_dim)
        )

    def forward(self, x):
        w = self.weight.t()

        y = torch.matmul(x, w)

        return y
model = LinearModel(2048, 512).eval()
x = torch.empty(1, 128, 2048)

print("Input shape:", x.shape)

mlir_module = torchscript.compile(
    model,
    (x,),
    output_type="linalg-on-tensors",
    use_tracing=True
)

with open("linear.mlir", "w") as f:
    f.write(mlir_module.operation.get_asm())

print("Saved to linear.mlir")