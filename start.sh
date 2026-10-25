#!/bin/bash

echo "=== Setting up models directory ==="

if [ -d "/runpod-volume/ComfyUI/models" ]; then
    echo "Found models in /runpod-volume/ComfyUI/models. Creating symlink..."
    rm -rf /workspace/ComfyUI/models
    ln -s /runpod-volume/ComfyUI/models /workspace/ComfyUI/models
elif [ -d "/runpod-volume/models" ]; then
    echo "Found models in /runpod-volume/models. Creating symlink..."
    rm -rf /workspace/ComfyUI/models
    ln -s /runpod-volume/models /workspace/ComfyUI/models
else
    echo "WARNING: Models directory not found in /runpod-volume!"
fi

echo "Models linked status:"
ls -la /workspace/ComfyUI/models || true

# 1. Архитектура CUDA для RTX 4090 и защита от фрагментации
export TORCH_CUDA_ARCH_LIST="8.9"
export CUDA_ARCH="89"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:64"

# 2. Обход проверки SageAttention в KJNodes с архитектурой sm89 для RTX 4090
python3 -c '
path = "/workspace/ComfyUI/custom_nodes/ComfyUI-KJNodes/nodes/ltxv_nodes.py"
try:
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()
    code = code.replace("if _cuda_archs is None:", "if False:")
    code = code.replace("_cuda_archs[0]", "\"sm89\"")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    print("Successfully patched Kijai SageAttention for sm89!")
except Exception as e:
    print(f"Patch warning: {e}")
' || true

# 3. Патч для Krea 2 Refiner (MXFP8) и безопасный запуск SageAttention
python3 -c '
ops_path = "/workspace/ComfyUI/comfy/ops.py"
try:
    with open(ops_path, "r", encoding="utf-8") as f:
        code = f.read()
    if "torch.float8_e8m0fnu" in code:
        code = code.replace("torch.float8_e8m0fnu", "getattr(torch, \"float8_e8m0fnu\", torch.uint8)")
    if "scale=\"recalculate\"" in code:
        code = code.replace("scale=\"recalculate\"", "scale=None")
    with open(ops_path, "w", encoding="utf-8") as f:
        f.write(code)
    print("Successfully patched comfy/ops.py!")
except Exception as e:
    print(f"Ops patch warning: {e}")

float_path = "/workspace/ComfyUI/comfy/float.py"
try:
    with open(float_path, "r", encoding="utf-8") as f:
        code = f.read()
    if "torch.float8_e8m0fnu" in code:
        code = code.replace("torch.float8_e8m0fnu", "getattr(torch, \"float8_e8m0fnu\", torch.uint8)")
    old_round = "return _ck_stochastic_rounding_fp8(value, rng, dtype)"
    new_round = "return value.to(dtype) if hasattr(value, \"to\") else _ck_stochastic_rounding_fp8(value, rng, dtype)"
    if old_round in code:
        code = code.replace(old_round, new_round)
        print("Successfully replaced stochastic_rounding with fast native cast!")
    with open(float_path, "w", encoding="utf-8") as f:
        f.write(code)
    print("Successfully patched comfy/float.py!")
except Exception as e:
    print(f"Float patch warning: {e}")

kitchen_sage = "/usr/local/lib/python3.10/dist-packages/comfy_kitchen/sage_attention.py"
try:
    with open(kitchen_sage, "r", encoding="utf-8") as f:
        code = f.read()

    # Заменяем только чистую функцию int8_attention, которую зовёт VAE
    attn_override = (
        "def int8_attention(query, key, value, *args, **kwargs):\n"
        "    import torch.nn.functional as F\n"
        "    return F.scaled_dot_product_attention(query, key, value)\n\n"
        "def _orig_int8_attention("
    )

    if "def int8_attention(" in code and "_orig_int8_attention" not in code:
        code = code.replace("def int8_attention(", attn_override)
        print("Successfully bypassed int8_attention to PyTorch SDPA in start.sh!")

    with open(kitchen_sage, "w", encoding="utf-8") as f:
        f.write(code)
except Exception as e:
    print(f"Kitchen Sage patch warning: {e}")

site_path = "/usr/local/lib/python3.10/dist-packages/sitecustomize.py"
try:
    with open(site_path, "w", encoding="utf-8") as f:
        f.write("import torch\nif not hasattr(torch, \"float8_e8m0fnu\"):\n    torch.float8_e8m0fnu = torch.uint8\n")
    print("Successfully created sitecustomize.py for PyTorch MXFP8!")
except Exception as e:
    print(f"Sitecustomize warning: {e}")
' || true

# 4. Запуск ComfyUI в фоне (выгрузка неактивных моделей при смене этапа)
cd /workspace/ComfyUI
python -u main.py --listen 127.0.0.1 --port 8188 --reserve-vram 0.5 --lowvram > /workspace/comfyui.log 2>&1 &
COMFY_PID=$!

echo "Waiting for ComfyUI and custom nodes to initialize (PID: $COMFY_PID)..."
while true; do
    if ! kill -0 $COMFY_PID 2>/dev/null; then
        echo "=========================================="
        echo "ERROR: ComfyUI crashed! Dumping log:"
        echo "=========================================="
        cat /workspace/comfyui.log
        echo "=========================================="
        sleep 60
        exit 1
    fi

    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:8188/object_info/easy%20seed")
    if [ "$STATUS" -eq 200 ]; then
        echo "ComfyUI and ComfyUI-Easy-Use are fully loaded and ready!"
        break
    fi
    sleep 2
done

# 5. Запуск RunPod Serverless Handler
python -u rp_handler.py
