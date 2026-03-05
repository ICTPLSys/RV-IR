import os
import sys
# 添加PYTHONPATH环境变量
home_dir = os.path.expanduser("~")
python_paths = [
    os.path.join(home_dir, "RV-IR/build/tools/torch-mlir/python_packages/torch_mlir"),
    os.path.join(home_dir, "RV-IR/projects/pt1/python/"),
    os.path.join(home_dir, "RV-IR/projects/pt1/python/torch_mlir")
]
for path in python_paths:
    if path not in sys.path:
        sys.path.insert(0, path)
os.environ["PYTHONPATH"] = os.pathsep.join(python_paths + sys.path)

import torch
import torchvision
import torchvision.transforms as transforms
from torch_mlir import torchscript
from PIL import Image
import requests
from io import BytesIO

weights = torchvision.models.ResNet18_Weights.DEFAULT
model = torchvision.models.resnet18(weights=weights).eval()

IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/2/26/YellowLabradorLooking_new.jpg"

headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"
}
response = requests.get(IMAGE_URL, headers=headers, timeout=10)
response.raise_for_status()

image = Image.open(BytesIO(response.content)).convert("RGB")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])  

input_tensor = preprocess(image).unsqueeze(0)

print("Input shape:", input_tensor.shape)

mlir_module = torchscript.compile(
    model,
    input_tensor,
    output_type="linalg-on-tensors",
    use_tracing=True  #更新了torch版本之后需要加上这个参数，才可以执行，有什么什么影响还不确定
)

with open("resnet18.mlir", "w") as f:
    f.write(mlir_module.operation.get_asm())

#..............下方代码可输出较为简便的mlir，数据被隐藏，方便调试

# resnet18 = torchvision.models.resnet18(pretrained=True)
# resnet18.eval()

# module = torchscript.compile(resnet18, torch.ones(1, 3, 224, 224), output_type="torch")

# print("TORCH OutputType\n", module.operation.get_asm(large_elements_limit=10))
# module = torchscript.compile(
#     resnet18, torch.ones(1, 3, 224, 224), output_type="linalg-on-tensors"
# )
# # 核心新增：将Linalg IR写入文件
# with open("resnet18_simple.mlir", "w", encoding="utf-8") as f:
#     f.write(module.operation.get_asm(large_elements_limit=10))
# print(
#     "LINALG_ON_TENSORS OutputType\n", module.operation.get_asm(large_elements_limit=10)
# )

