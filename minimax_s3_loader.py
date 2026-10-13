import io
import json
import av
import numpy as np
from PIL import Image
import requests
import torch
import torchaudio

# -------------------------------------------------------------
# ЭМУЛЯЦИЯ / ДУБЛЕРЫ НОД: ComfyUI-Easy-Use
# -------------------------------------------------------------
class EasySeedBypass:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "get_seed"
    CATEGORY = "EasyUse/Seed"

    def get_seed(self, seed):
        return (seed,)

class EasyShowAnythingBypass:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "text": ("STRING", {"forceInput": False, "default": ""}),
                "anything": ("*",),
            },
        }
    RETURN_TYPES = ("*", "STRING")
    RETURN_NAMES = ("output", "text")
    FUNCTION = "show"
    CATEGORY = "EasyUse/Logic"

    def show(self, text="", anything=None, **kwargs):
        return (anything, str(text))

class EasyCleanGpuBypass:

  @classmethod
  def INPUT_TYPES(s):
    return {
        "required": {},
        "optional": {
            "anything": ("*",),
        },
    }

  RETURN_TYPES = ("*",)
  RETURN_NAMES = ("output",)
  FUNCTION = "clean"
  CATEGORY = "EasyUse/Fix"

  def clean(self, anything=None, **kwargs):
    print(
        "=== [CleanVRAM] Глубокая очистка VRAM перед этапом Krea 2... ===",
        flush=True,
    )
    try:
      import comfy.model_management as mm

      mm.unload_all_models()
      mm.soft_empty_cache()
    except Exception as e:
      print(f"[CleanVRAM] Warning: {e}", flush=True)

    import gc

    gc.collect()
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
      torch.cuda.ipc_collect()
      # Сбрасываем пиковые счетчики аллокатора
      torch.cuda.reset_peak_memory_stats()

    print(
        "=== [CleanVRAM] Видеопамять полностью сброшена и готова под Krea! ===",
        flush=True,
    )
    return (anything,)

class EasyIndexAnythingBypass:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "index": ("INT", {"default": 0, "min": 0, "max": 9999}),
            },
            "optional": {
                "any": ("*",),
            },
        }
    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "get_index"
    CATEGORY = "EasyUse/Logic"

    def get_index(self, index=0, any=None, **kwargs):
        if isinstance(any, (list, tuple)):
            if 0 <= index < len(any):
                return (any[index],)
            return (any[0] if len(any) > 0 else None,)
        return (any,)

# -------------------------------------------------------------
# ЗАГРУЗЧИК ИЗ YANDEX S3 И СПЛИТТЕР РЕФЕРЕНСОВ
# -------------------------------------------------------------
class MiniMax_S3_BatchLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image_urls": ("STRING", {"multiline": True, "default": "[]"}),
                "video_urls": ("STRING", {"multiline": True, "default": "[]"}),
                "audio_urls": ("STRING", {"multiline": True, "default": "[]"}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("references",)
    FUNCTION = "load_media"
    CATEGORY = "MiniMax/Serverless"

    def _download_bytes(self, url):
        print(f"[MiniMax_S3] Скачивание: {url}", flush=True)
        res = requests.get(url.strip(), timeout=30)
        res.raise_for_status()
        return io.BytesIO(res.content)

    def _load_image(self, url):
        buf = self._download_bytes(url)
        img = Image.open(buf).convert("RGB")
        img_np = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(img_np)[None,]

    def load_media(self, image_urls="[]", video_urls="[]", audio_urls="[]", **kwargs):
        def parse_urls(raw):
            raw = str(raw).strip()
            if not raw or raw == "[]":
                return []
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, list) else [parsed]
            except Exception:
                return [line.strip() for line in raw.split("\n") if line.strip().startswith("http")]

        img_list = parse_urls(image_urls)[:9]
        print(f"=== [MiniMax_S3] Запрошено картинок-референсов: {len(img_list)} ===", flush=True)

        refs = {"pictures": [], "videos": [], "video_audios": [], "audios": []}

        for url in img_list:
            try:
                refs["pictures"].append(self._load_image(url))
            except Exception as e:
                print(f"[MiniMax_S3] Ошибка скачивания картинки {url}: {e}", flush=True)

        print(f"=== [MiniMax_S3] Успешно загружено картинок в память: {len(refs['pictures'])} ===", flush=True)
        return (refs,)

class MiniMaxH3ReferenceSplitter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"references": ("*",)}}

    RETURN_NAMES = (
        "picture_1", "picture_2", "picture_3",
        "picture_4", "picture_5", "picture_6",
        "picture_7", "picture_8", "picture_9",
        "video_1", "video_2", "video_3",
        "video_audio_1", "video_audio_2", "video_audio_3",
        "audio_1", "audio_2", "audio_3",
        "pictures", "videos", "audios",
    )
    RETURN_TYPES = tuple(["*"] * len(RETURN_NAMES))
    FUNCTION = "split"
    CATEGORY = "MiniMax"

    def split(self, references=None):
        if not isinstance(references, dict):
            return tuple([None] * len(self.RETURN_NAMES))

        pics = references.get("pictures", [])
        vids = references.get("videos", [])
        v_auds = references.get("video_audios", [])
        auds = references.get("audios", [])

        out = []
        for i in range(9):
            out.append(pics[i] if i < len(pics) else None)
        for i in range(3):
            out.append(vids[i] if i < len(vids) else None)
        for i in range(3):
            out.append(v_auds[i] if i < len(v_auds) else None)
        for i in range(3):
            out.append(auds[i] if i < len(auds) else None)
        out.extend([pics, vids, auds])

        return tuple(out)

class ImageExists:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "switch": ("*",),
                "image": ("*",),
            },
        }

    RETURN_TYPES = ("*", "BOOLEAN")
    RETURN_NAMES = ("image", "exists")
    FUNCTION = "check_exists"
    CATEGORY = "MiniMax"

    def check_exists(self, switch=None, image=None):
        val = switch if switch is not None else image

        if isinstance(val, (list, tuple)):
            val = val[0] if len(val) > 0 else None

        exists = False
        if val is not None and isinstance(val, torch.Tensor):
            if val.numel() > 64 * 64 * 3:
                exists = True
            elif val.numel() > 0 and (val.shape[-2] > 64 or val.shape[-3] > 64):
                exists = True

        if not exists or val is None:
            val = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        return (val, bool(exists))

NODE_CLASS_MAPPINGS = {
    "MiniMax_S3_BatchLoader": MiniMax_S3_BatchLoader,
    "MiniMaxH3MediaLoader": MiniMax_S3_BatchLoader,
    "MiniMaxH3ReferenceSplitter": MiniMaxH3ReferenceSplitter,
    "ImageExists": ImageExists,
    "easy seed": EasySeedBypass,
    "easy showAnything": EasyShowAnythingBypass,
    "easy cleanGpuUsed": EasyCleanGpuBypass,
    "easy indexAnything": EasyIndexAnythingBypass,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMax_S3_BatchLoader": "MiniMax S3 Batch Loader",
    "MiniMaxH3MediaLoader": "MiniMax H3 Media Loader",
    "MiniMaxH3ReferenceSplitter": "MiniMax H3 Reference Splitter",
    "ImageExists": "Image Exists",
    "easy seed": "Prompt Seed Bypass",
    "easy showAnything": "Show Anything Bypass",
    "easy cleanGpuUsed": "Clean VRAM Bypass",
    "easy indexAnything": "Index Anything Bypass",
}
