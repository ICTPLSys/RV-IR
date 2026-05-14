import argparse
import os
import sys
from io import BytesIO

# 添加PYTHONPATH环境变量
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MLIR_OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "mlir_output")
_DEFAULT_OUTPUT = os.path.join(_DEFAULT_MLIR_OUTPUT_DIR, "resnet18.mlir")

home_dir = os.path.expanduser("~")
python_paths = [
    os.path.join(home_dir, "RV-IR/build/tools/torch-mlir/python_packages/torch_mlir"),
    os.path.join(home_dir, "RV-IR/projects/pt1/python/"),
    os.path.join(home_dir, "RV-IR/projects/pt1/python/torch_mlir"),
]
for path in python_paths:
    if path not in sys.path:
        sys.path.insert(0, path)
os.environ["PYTHONPATH"] = os.pathsep.join(python_paths + sys.path)

import requests
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch_mlir import torchscript


def main():
    parser = argparse.ArgumentParser(description="Compile ResNet18 to linalg IR.")
    parser.add_argument(
        "--output",
        type=str,
        default=_DEFAULT_OUTPUT,
        help="Output MLIR file path (default: mlir_output/resnet18.mlir next to this script).",
    )
    parser.add_argument(
        "--large-elements-limit",
        type=int,
        default=10,
        help="Elide large tensor attributes in printed MLIR (same idea as mnistnet_to_linalg.py).",
    )
    args = parser.parse_args()

    weights = torchvision.models.ResNet18_Weights.DEFAULT
    model = torchvision.models.resnet18(weights=weights).eval()

    IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)",
    }
    response = requests.get(IMAGE_URL, headers=headers, timeout=10)
    response.raise_for_status()

    image = Image.open(BytesIO(response.content)).convert("RGB")

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    input_tensor = preprocess(image).unsqueeze(0)

    print("Input shape:", input_tensor.shape)

    mlir_module = torchscript.compile(
        model,
        input_tensor,
        output_type="linalg-on-tensors",
        use_tracing=True,  # 更新了 torch 版本之后需要加上这个参数，才可以执行
    )

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(
            mlir_module.operation.get_asm(
                large_elements_limit=args.large_elements_limit
            )
        )

    print(f"Linalg IR written to: {args.output}")


if __name__ == "__main__":
    main()
