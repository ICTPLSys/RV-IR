#------------------下方代码可以完整的打印Llama3.2模型的mlir--------------------
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
from transformers import AutoModelForCausalLM
from torch_mlir import torchscript
# import os
MODEL_PATH = os.path.expanduser("~/.cache/modelscope/hub/models/LLM-Research/Llama-3___2-1B")

SEQ_LEN = 128

torch.set_grad_enabled(False)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="cpu",
    dtype=torch.float32,
    local_files_only=True,
).eval()

layer = model.model.layers[0]  #使用llama3.2模型的第0个decoder
config = model.config

hidden = config.hidden_size
num_heads = config.num_attention_heads
head_dim = hidden // num_heads

class LlamaDecoderBlock(torch.nn.Module):
    def __init__(self, layer, config):
        super().__init__()
        self.ln1 = layer.input_layernorm
        self.q_proj = layer.self_attn.q_proj
        self.k_proj = layer.self_attn.k_proj
        self.v_proj = layer.self_attn.v_proj
        self.o_proj = layer.self_attn.o_proj
        self.ln2 = layer.post_attention_layernorm
        self.mlp = layer.mlp

        self.num_q_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.num_q_heads

    def forward(self, x):
        # x: [B, S, hidden]
        B, S, _ = x.shape

        #---- RMSNorm ----
        h = self.ln1(x)

        #---- QKV projections ----
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        #---- reshape ----  
        q = q.view(B, S, self.num_q_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        #---- GQA expand KV ----
        if self.num_q_heads != self.num_kv_heads:
            repeat = self.num_q_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        #---- Attention ----
        scores = torch.matmul(q, k.transpose(-2, -1))
        probs = torch.softmax(scores, dim=-1)
        attn = torch.matmul(probs, v)

        #---- merge heads ----
        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(B, S, self.num_q_heads * self.head_dim)

        h = x + self.o_proj(attn)

        # ---- MLP ----
        h2 = self.ln2(h)
        out = h + self.mlp(h2)
        return out


block = LlamaDecoderBlock(layer, model.config)
x = torch.randn(1, SEQ_LEN, model.config.hidden_size)

mlir_module = torchscript.compile(
    block,
    x,
    output_type="linalg-on-tensors",
    use_tracing=True,
)
#这个输出的文件包含单层decoder的所有参数，生成的文件很大（484MB），暂时不使用这个
# with open("llama3_decoder_block.mlir", "w") as f:
#     f.write(mlir_module.operation.get_asm())

with open("llama3_decoder_block.mlir", "w") as f:
    f.write(
        mlir_module.operation.get_asm(
            large_elements_limit=32   # 或16/64
        )
    )

print("✔Exported llama3_decoder_block.mlir")