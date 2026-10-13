FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /workspace

# Системные пакеты
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1-mesa-glx git curl dos2unix build-essential \
    && rm -rf /var/lib/apt/lists/*

# Обновляем PyTorch и torchvision до 2.4+
RUN pip install --no-cache-dir --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Установка нативного SageAttention
RUN pip install --no-cache-dir sageattention

# Клонируем ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/ComfyUI

WORKDIR /workspace/ComfyUI
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir runpod requests av triton boto3 pillow

# Патчер comfy_kitchen
COPY patch_kitchen.py /workspace/patch_kitchen.py
RUN python3 /workspace/patch_kitchen.py

# Клонируем кастомные ноды
WORKDIR /workspace/ComfyUI/custom_nodes
RUN git clone https://github.com/Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder.git
RUN git clone https://github.com/kijai/ComfyUI-SolAttn_triton.git
RUN git clone https://github.com/kijai/ComfyUI-KJNodes.git
RUN git clone https://github.com/rgthree/rgthree-comfy.git
RUN git clone https://github.com/yolain/ComfyUI-Easy-Use.git
RUN git clone https://github.com/theUpsider/ComfyUI-Logic.git
RUN git clone https://github.com/chflame163/ComfyUI_LayerStyle.git
RUN git clone https://github.com/Smirnov75/ComfyUI-mxToolkit.git
RUN git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git
RUN git clone https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes.git
RUN git clone https://github.com/StableLlama/ComfyUI-basic_data_handling.git
RUN git clone https://github.com/lbouaraba/comfyui-krea2edit.git

# Новые ноды для генерации изображений:
RUN git clone https://github.com/astropuzzo/ComfyUI-MiniMax-H3-Image-Studio.git
RUN git clone https://github.com/Kvento/Comfyui_Minimax_h3_latent_Upscaler_ND.git
RUN git clone https://github.com/mickmumpitz/ComfyUI-Mickmumpitz-Nodes.git

# Устанавливаем зависимости нод
RUN if [ -f ComfyUI-Fantastic-MiniMaxH3-PromptBuilder/requirements.txt ]; then pip install --no-cache-dir -r ComfyUI-Fantastic-MiniMaxH3-PromptBuilder/requirements.txt; fi
RUN if [ -f ComfyUI-KJNodes/requirements.txt ]; then pip install --no-cache-dir -r ComfyUI-KJNodes/requirements.txt; fi
RUN if [ -f ComfyUI-Easy-Use/requirements.txt ]; then pip install --no-cache-dir -r ComfyUI-Easy-Use/requirements.txt; fi
RUN if [ -f ComfyUI_LayerStyle/requirements.txt ]; then pip install --no-cache-dir -r ComfyUI_LayerStyle/requirements.txt; fi
RUN if [ -f ComfyUI-basic_data_handling/requirements.txt ]; then pip install --no-cache-dir -r ComfyUI-basic_data_handling/requirements.txt; fi
RUN if [ -f ComfyUI-MiniMax-H3-Image-Studio/requirements.txt ]; then pip install --no-cache-dir -r ComfyUI-MiniMax-H3-Image-Studio/requirements.txt; fi
RUN if [ -f comfyui-krea2edit/requirements.txt ]; then pip install --no-cache-dir -r comfyui-krea2edit/requirements.txt; fi

# S3 загрузчик ссылок и сплиттер
RUN mkdir -p /workspace/ComfyUI/custom_nodes/MiniMax_S3_Loader
COPY minimax_s3_loader.py /workspace/ComfyUI/custom_nodes/MiniMax_S3_Loader/__init__.py

# Переносим скрипты хэндлера, воркфлоу и старта
WORKDIR /workspace/ComfyUI
COPY image_api_ready.json /workspace/ComfyUI/image_api_ready.json
COPY rp_handler.py /workspace/ComfyUI/rp_handler.py
COPY start.sh /workspace/ComfyUI/start.sh

RUN dos2unix /workspace/ComfyUI/start.sh && chmod +x /workspace/ComfyUI/start.sh

CMD ["/workspace/ComfyUI/start.sh"]
