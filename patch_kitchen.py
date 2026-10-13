import glob
import os
import re

# 1. Исправление типизации list[...] для Python 3.10
path_pattern = "/usr/local/lib/python3.10/dist-packages/comfy_kitchen/**/*.py"
files = glob.glob(path_pattern, recursive=True)
pattern = re.compile(r"\blist\[(int|bool|float|str)\]")

for file_path in files:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    if pattern.search(content):
        content = pattern.sub(r"typing.Sequence[\1]", content)

        if "from __future__ import annotations" in content:
            if "import typing" not in content:
                content = content.replace(
                    "from __future__ import annotations",
                    "from __future__ import annotations\nimport typing",
                )
        else:
            if "import typing" not in content:
                content = "import typing\n" + content

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

print("comfy_kitchen typing successfully patched!")

# 2. Прямой fallback для dequantize_mxfp8 в quantization.py
quant_path = "/usr/local/lib/python3.10/dist-packages/comfy_kitchen/backends/eager/quantization.py"
if os.path.exists(quant_path):
    with open(quant_path, "r", encoding="utf-8") as f:
        q_code = f.read()

    old_call = 'impl = registry.get_implementation("dequantize_mxfp8", kwargs=kwargs)'
    
    # Безопасный и экономный по памяти fallback сразу в bfloat16
    new_call = """try:
        impl = registry.get_implementation("dequantize_mxfp8", kwargs=kwargs)
    except Exception:
        scale = block_scales.to(torch.bfloat16)
        if scale.ndim < qx.ndim:
            scale = scale.unsqueeze(-1)
        if qx.numel() != scale.numel():
            scale = scale.repeat_interleave(qx.numel() // scale.numel(), dim=-1)
        return qx.to(torch.bfloat16) * scale"""

    if old_call in q_code:
        q_code = q_code.replace(old_call, new_call)
        with open(quant_path, "w", encoding="utf-8") as f:
            f.write(q_code)
        print("comfy_kitchen dequantize_mxfp8 fallback successfully patched!")
    else:
        print("dequantize_mxfp8 call pattern not found, checking alternatives...")

# 3. Полный перехват int8_attention и prequantize в comfy_kitchen
kitchen_sage = "/usr/local/lib/python3.10/dist-packages/comfy_kitchen/sage_attention.py"
if os.path.exists(kitchen_sage):
    with open(kitchen_sage, "r", encoding="utf-8") as f:
        sage_code = f.read()

    # Перенаправляем квантованное внимание VAE и модели напрямую на PyTorch SDPA
    attn_override = (
        "def int8_attention(query, key, value, *args, **kwargs):\n"
        "    import torch.nn.functional as F\n"
        "    return F.scaled_dot_product_attention(query, key, value)\n\n"
        "def _orig_int8_attention("
    )

    if "def int8_attention(" in sage_code and "_orig_int8_attention" not in sage_code:
        sage_code = sage_code.replace("def int8_attention(", attn_override)
        print("comfy_kitchen int8_attention redirected to PyTorch SDPA!")

    bypass_func = (
        "def prequantize_int8_attention(q, k, v, *args, **kwargs):\n"
        "    return q, k, v, None, None, None\n\n"
        "def _orig_prequantize_int8_attention("
    )

    if "def prequantize_int8_attention(" in sage_code and "_orig_prequantize_int8_attention" not in sage_code:
        sage_code = sage_code.replace("def prequantize_int8_attention(", bypass_func)
        print("comfy_kitchen prequantize_int8_attention bypassed!")

    with open(kitchen_sage, "w", encoding="utf-8") as f:
        f.write(sage_code)
