import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import uuid
import boto3
import runpod
import torch

try:
    import comfy_kitchen
    import torch.nn.functional as F
    comfy_kitchen.int8_attention = lambda q, k, v, *a, **kw: F.scaled_dot_product_attention(q, k, v)
except Exception:
    pass

# Полифил для форматов квантования MXFP8 (Krea 2 Refiner)
if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = torch.uint8

COMFY_URL = "http://127.0.0.1:8188"
LOG_PATH = "/workspace/comfyui.log"
COMFY_OUTPUT_DIR = "/workspace/ComfyUI/output"

S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL", "https://storage.yandexcloud.net")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "")
S3_ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.environ.get("S3_SECRET_KEY", "")

def ensure_comfyui_ready():
    print("=== [INIT] Checking models symlink ===", flush=True)
    if os.path.exists("/runpod-volume/ComfyUI/models"):
        os.system("rm -rf /workspace/ComfyUI/models && ln -s /runpod-volume/ComfyUI/models /workspace/ComfyUI/models")
    elif os.path.exists("/runpod-volume/models"):
        os.system("rm -rf /workspace/ComfyUI/models && ln -s /runpod-volume/models /workspace/ComfyUI/models")

    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
    os.environ["CUDA_ARCH"] = "89"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:128"

    try:
        path = "/workspace/ComfyUI/custom_nodes/ComfyUI-KJNodes/nodes/ltxv_nodes.py"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                code = f.read()
            code = code.replace("if _cuda_archs is None:", "if False:")
            code = code.replace("_cuda_archs[0]", "\"sm89\"")
            with open(path, "w", encoding="utf-8") as f:
                f.write(code)
            print("[INIT] SageAttention patched cleanly for sm89", flush=True)
    except Exception as e:
        print(f"[INIT] Patch warning: {e}", flush=True)
    
    try:
        req = urllib.request.Request(f"{COMFY_URL}/object_info/easy%20seed")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                print("=== [INIT] ComfyUI already running! ===", flush=True)
                return
    except Exception:
        pass

    print("=== [INIT] Starting ComfyUI process in background... ===", flush=True)
    log_file = open(LOG_PATH, "w", encoding="utf-8")
    proc = subprocess.Popen(
        ["python", "-u", "main.py", "--listen", "127.0.0.1", "--port", "8188", "--reserve-vram", "0.5", "--normalvram"],
        cwd="/workspace/ComfyUI",
        stdout=log_file,
        stderr=subprocess.STDOUT
    )

    for _ in range(150):
        if proc.poll() is not None:
            if os.path.exists(LOG_PATH):
                with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
                    print(f.read()[-3000:], flush=True)
            raise RuntimeError("ComfyUI crashed on startup")

        try:
            req = urllib.request.Request(f"{COMFY_URL}/object_info/easy%20seed")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    print("=== [INIT] ComfyUI is fully ready! ===", flush=True)
                    return
        except Exception:
            pass
        time.sleep(2)

    raise TimeoutError("ComfyUI startup timed out")

def upload_file_to_s3(local_path, s3_key, content_type="image/png"):
    try:
        session = boto3.session.Session()
        s3_client = session.client(
            service_name='s3',
            endpoint_url=S3_ENDPOINT_URL,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            region_name='ru-central1'
        )
        s3_client.upload_file(
            Filename=local_path,
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            ExtraArgs={'ContentType': content_type}
        )
        file_url = f"https://storage.yandexcloud.net/{S3_BUCKET_NAME}/{s3_key}"
        return file_url, None
    except Exception as e:
        err_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[ERROR] S3 Upload failed: {err_msg}", flush=True)
        return None, err_msg

def get_comfy_logs(tail_lines=60):
    if not os.path.exists(LOG_PATH):
        return "Log file not found."
    try:
        with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            return "".join(lines[-tail_lines:])
    except Exception as e:
        return f"Error reading log: {e}"

def queue_prompt(prompt_workflow):
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request(
        f"{COMFY_URL}/prompt", 
        data=data, 
        headers={'Content-Type': 'application/json'}
    )
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8', errors='replace')
        print("=== COMFYUI REJECTED WORKFLOW (400) ===", flush=True)
        print(error_body, flush=True)
        raise RuntimeError(f"ComfyUI Validation Error: {error_body}")

def get_history(prompt_id):
    try:
        with urllib.request.urlopen(f"{COMFY_URL}/history/{prompt_id}") as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception:
        return {}

def handler(event):
    job_input = event.get("input", {})
    
    with open("image_api_ready.json", "r", encoding="utf-8") as f:
        wf = json.load(f)

    for node_id, node_data in wf.items():
        if "inputs" in node_data:
            for k, v in node_data["inputs"].items():
                if isinstance(v, str) and "\\" in v:
                    node_data["inputs"][k] = v.replace("\\", "/")

    # Параметры задачи
    prompt_text = job_input.get("prompt", "masterpiece, highly detailed")
    seed = int(job_input.get("seed", time.time()))
    aspect_ratio = job_input.get("aspect_ratio", "16:9 (Widescreen)")
    megapixels = float(job_input.get("megapixels", 4.0))
    sampler_name = job_input.get("sampler_name", "euler")
    scheduler = job_input.get("scheduler", "simple")
    ref_fidelity = float(job_input.get("ref_identity_strength", job_input.get("ref_fidelity", 0.9)))
    
    first_pass_steps = int(job_input.get("steps", job_input.get("first_pass_steps", 10)))
    second_pass_details = int(job_input.get("second_pass_details", 20))
    second_pass_steps = int(job_input.get("second_pass_steps", 4))
    
    enable_krea = bool(job_input.get("enable_krea_refiner", True))
    krea_details = int(job_input.get("krea_refiner_details", 24))
    krea_steps = int(job_input.get("krea_refiner_steps", 4))
    krea_seed = int(job_input.get("krea_seed", seed))
    krea_scale_by = float(job_input.get("krea_scale_by", 1.0))
    krea_color_match = float(job_input.get("krea_color_match", 0.6))
    krea_keep_likeness = bool(job_input.get("krea_keep_likeness", False))

    user_id = str(job_input.get("user_id", "1")).strip()
    image_urls = job_input.get("image_urls", [])

    # 1. Промпты
    if "1" in wf and "inputs" in wf["1"]:
        wf["1"]["inputs"]["prompt"] = prompt_text
    if "2" in wf and "inputs" in wf["2"]:
        wf["2"]["inputs"]["edit_instruction"] = prompt_text
    if "9835:9821" in wf and "inputs" in wf["9835:9821"]:
        wf["9835:9821"]["inputs"]["text"] = prompt_text
    if "9835:9818" in wf and "inputs" in wf["9835:9818"]:
        wf["9835:9818"]["inputs"]["prompt"] = prompt_text

    # 2. Сила идентичности референсов
    if "9649" in wf and "inputs" in wf["9649"]:
        wf["9649"]["inputs"]["Xf"] = ref_fidelity
    if "2" in wf and "inputs" in wf["2"]:
        wf["2"]["inputs"]["source_fidelity"] = ref_fidelity

    # 3. Разрешение и соотношение сторон
    aspect_map = {
        "16:9": "16:9 (Widescreen)",
        "9:16": "9:16 (Portrait Widescreen)",
        "1:1": "1:1 (Square)",
        "2:3": "2:3 (Portrait Photo)",
        "3:2": "3:2 (Photo)",
        "4:3": "4:3 (Standard)",
        "3:4": "3:4 (Portrait Standard)",
        "21:9": "21:9 (Ultrawide)"
    }
    
    # Берем первые цифры (например, "9:16" из "9:16 (Portrait)")
    clean_ratio = str(aspect_ratio).split()[0] if aspect_ratio else "16:9"
    final_aspect = aspect_map.get(clean_ratio, "16:9 (Widescreen)")

    if "9664:9663" in wf and "inputs" in wf["9664:9663"]:
        wf["9664:9663"]["inputs"]["aspect_ratio"] = final_aspect
        wf["9664:9663"]["inputs"]["megapixels"] = megapixels

    # 4. Семплер и шедулер MiniMax
    if "9658:9360" in wf and "inputs" in wf["9658:9360"]:
        wf["9658:9360"]["inputs"]["sampler_name"] = sampler_name
    if "9658:9361" in wf and "inputs" in wf["9658:9361"]:
        wf["9658:9361"]["inputs"]["scheduler"] = scheduler
    if "9658:9363" in wf and "inputs" in wf["9658:9363"]:
        wf["9658:9363"]["inputs"]["item_0"] = sampler_name
        wf["9658:9363"]["inputs"]["item_1"] = scheduler

    # 5. Шаги первого и второго проходов
    if "9671" in wf and "inputs" in wf["9671"]:
        wf["9671"]["inputs"]["Xi"] = first_pass_steps
        wf["9671"]["inputs"]["Xf"] = first_pass_steps
    if "9669" in wf and "inputs" in wf["9669"]:
        wf["9669"]["inputs"]["Xi"] = second_pass_details
        wf["9669"]["inputs"]["Xf"] = second_pass_details
    if "9670" in wf and "inputs" in wf["9670"]:
        wf["9670"]["inputs"]["Xi"] = second_pass_steps
        wf["9670"]["inputs"]["Xf"] = second_pass_steps

    # 6. Сиды
    if "9640" in wf and "inputs" in wf["9640"]:
        wf["9640"]["inputs"]["seed"] = seed
    if "9639" in wf and "inputs" in wf["9639"]:
        wf["9639"]["inputs"]["seed"] = seed
    if "9791" in wf and "inputs" in wf["9791"]:
        wf["9791"]["inputs"]["seed"] = krea_seed

    # 7. Параметры Krea Refiner
    if "9762" in wf and "inputs" in wf["9762"]:
      wf["9762"]["inputs"]["Xi"] = krea_details
      wf["9762"]["inputs"]["Xf"] = krea_details
    if "9763" in wf and "inputs" in wf["9763"]:
      wf["9763"]["inputs"]["Xi"] = krea_steps
      wf["9763"]["inputs"]["Xf"] = krea_steps

    # Держим масштаб 1.0, чтобы сохранить разрешение MiniMax
    if "9835:9812" in wf and "inputs" in wf["9835:9812"]:
      wf["9835:9812"]["inputs"]["scale_by"] = krea_scale_by

    if "9835:9814" in wf and "inputs" in wf["9835:9814"]:
      wf["9835:9814"]["inputs"]["strength"] = krea_color_match
    if "9835:9848" in wf and "inputs" in wf["9835:9848"]:
      wf["9835:9848"]["inputs"]["switch"] = krea_keep_likeness
    if "9835:9849" in wf and "inputs" in wf["9835:9849"]:
      wf["9835:9849"]["inputs"]["switch"] = krea_keep_likeness

    # Переключатель выхода: Krea 2 или чистый MiniMax
    if not enable_krea and "8" in wf and "inputs" in wf["8"]:
        wf["8"]["inputs"]["images"] = ["9644", 0]

    # 8. Ссылки на референсы
    if "3" in wf and "inputs" in wf["3"]:
        wf["3"]["inputs"]["image_urls"] = json.dumps(image_urls) if isinstance(image_urls, list) else image_urls

    # 9. LoRA MiniMax (цепочка стандартных слотов LoraLoaderModelOnly)
    minimax_loras = job_input.get("loras", [])
    # Файл-заглушка для пустых слотов (с силой 0.0), чтобы ComfyUI не ругался на 'None'
    DUMMY_LORA = "cinematic_fantasy.safetensors"

    for slot_idx in range(1, 5):
        slot_node_id = f"9657:{slot_idx}"
        if slot_node_id in wf and "inputs" in wf[slot_node_id]:
            if slot_idx <= len(minimax_loras):
                lora_item = minimax_loras[slot_idx - 1]
                lora_name = str(lora_item.get("name") or lora_item.get("file") or "").strip()
                if lora_name and not lora_name.endswith(".safetensors"):
                    lora_name += ".safetensors"
                strength = float(lora_item.get("strength", 0.8))

                wf[slot_node_id]["inputs"]["lora_name"] = lora_name
                wf[slot_node_id]["inputs"]["strength_model"] = strength
            else:
                # Ставим валидный файл, но выключаем силу в 0.0
                wf[slot_node_id]["inputs"]["lora_name"] = DUMMY_LORA
                wf[slot_node_id]["inputs"]["strength_model"] = 0.0

    # 10. LoRA Krea 2
    krea_loras = job_input.get("krea_loras", [])
    if krea_loras and "9787" in wf and "inputs" in wf["9787"]:
        for idx, lora_item in enumerate(krea_loras, start=1):
            lora_name = str(lora_item.get("name") or lora_item.get("file") or "").strip()
            if not lora_name:
                continue
            if not lora_name.endswith(".safetensors"):
                lora_name += ".safetensors"
            strength = float(lora_item.get("strength", 0.9))
            wf["9787"]["inputs"][f"lora_{idx}"] = {
                "on": True,
                "lora": lora_name,
                "strength": strength
            }

    # 11. Настройка внимания
    if "9598:9121" in wf and "inputs" in wf["9598:9121"]:
      wf["9598:9121"]["inputs"][
          "sage_attention"
      ] = "auto"  # Для MiniMax оставляем auto (дает супер-качество!)

    if "9598:9712" in wf and "inputs" in wf["9598:9712"]:
      wf["9598:9712"]["inputs"]["attention"] = "original"

    if "9749:164" in wf and "inputs" in wf["9749:164"]:
      wf["9749:164"]["inputs"][
          "sage_attention"
      ] = "disabled"  # Для Krea 2 СТРОГО disabled, чтобы не было черного экрана!

    queued = queue_prompt(wf)
    prompt_id = queued.get("prompt_id")
    if not prompt_id:
        return {"status": "error", "message": "Failed to queue workflow", "raw": queued}

    max_wait = 900
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        history = get_history(prompt_id)
        if prompt_id in history:
            job_data = history[prompt_id]
            outputs = job_data.get("outputs", {})
            status_info = job_data.get("status", {})
            
            save_node_out = outputs.get("8", {})
            raw_images = save_node_out.get("images", [])
            
            uploaded_images = []
            for item in raw_images:
                filename = item.get("filename")
                subfolder = item.get("subfolder", "")
                if filename:
                    local_filepath = os.path.join(COMFY_OUTPUT_DIR, subfolder, filename)
                    ext = os.path.splitext(filename)[1] or ".png"
                    unique_name = f"{uuid.uuid4().hex}{ext}"
                    s3_key = f"users/user_{user_id}/output/{unique_name}"
                    
                    if os.path.exists(local_filepath):
                        s3_url, upload_err = upload_file_to_s3(local_filepath, s3_key, content_type="image/png")
                        uploaded_images.append({
                            "filename": filename,
                            "s3_key": s3_key,
                            "url": s3_url,
                            "upload_error": upload_err
                        })
                    else:
                        uploaded_images.append({
                            "filename": filename,
                            "error": f"File not found on disk: {local_filepath}"
                        })
            
            comfy_stdout = get_comfy_logs(80)
            
            return {
                "status": "success" if uploaded_images else "incomplete_or_failed",
                "prompt_id": prompt_id,
                "execution_status": status_info,
                "outputs": outputs,
                "images": uploaded_images,
                "image_url": uploaded_images[0]["url"] if uploaded_images and "url" in uploaded_images[0] else None,
                "comfy_console_log": comfy_stdout
            }
        time.sleep(1)

    return {
        "status": "timeout", 
        "prompt_id": prompt_id,
        "comfy_console_log": get_comfy_logs(80)
    }

if __name__ == "__main__":
    ensure_comfyui_ready()
    runpod.serverless.start({"handler": handler})
