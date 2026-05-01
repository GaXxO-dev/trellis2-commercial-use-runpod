FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV OPENCV_IO_ENABLE_OPENEXR=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV TORCH_CUDA_ARCH_LIST="8.0;9.0"
ENV MAX_JOBS=4

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    ffmpeg libgl1-mesa-glx libglib2.0-0 libjpeg-dev libwebp-dev ninja-build git g++-12 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel ninja

RUN echo "torch==2.6.0" > /tmp/constraints.txt && \
    echo "torchvision==0.21.0" >> /tmp/constraints.txt && \
    echo "flash-attn==2.7.3" >> /tmp/constraints.txt

RUN pip install --no-cache-dir \
    torch==2.6.0+cu124 \
    torchvision==0.21.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

RUN pip install --no-cache-dir \
    https://github.com/GaXxO-dev/TRELLIS.2-commercial-use/releases/download/v0.1.0/drtk-0.1.0+cuda124-cp310-cp310-linux_x86_64.whl

COPY requirements-inference.txt /tmp/requirements-inference.txt
RUN pip install --no-cache-dir --constraint /tmp/constraints.txt \
    -r /tmp/requirements-inference.txt

RUN pip install --no-cache-dir --constraint /tmp/constraints.txt kornia timm psutil

RUN pip install --no-cache-dir runpod boto3 requests

RUN pip install --no-cache-dir --constraint /tmp/constraints.txt \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

RUN git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/CuMesh --recursive \
    && pip install --no-cache-dir --no-build-isolation /tmp/CuMesh \
    && rm -rf /tmp/CuMesh

RUN git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/FlexGEMM --recursive \
    && pip install --no-cache-dir --no-build-isolation /tmp/FlexGEMM \
    && rm -rf /tmp/FlexGEMM

RUN git clone --recursive \
    https://github.com/GaXxO-dev/TRELLIS.2-commercial-use.git /app/TRELLIS.2

RUN pip install --no-cache-dir --no-build-isolation /app/TRELLIS.2/o-voxel

ENV HF_HOME="/runpod-volume/huggingface-cache"
ENV PYTHONPATH="/app/TRELLIS.2"
ENV TRELLIS_PATH="/app/TRELLIS.2"

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
