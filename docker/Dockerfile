FROM nvcr.io/nvidia/pytorch:21.10-py3

ENV DEBIAN_FRONTEND noninteractive

RUN rm -rf /var/lib/apt/lists/* &&  rm -rf /etc/apt/sources.list.d/*\
    /etc/apt/sources.list.d/cuda.list \
    /etc/apt/sources.list.d/nvidia-ml.list && \
    sed -i s@/archive.ubuntu.com/@/mirrors.ustc.edu.cn/@g /etc/apt/sources.list && \
    apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install LaneDet 
RUN conda clean --all
RUN git clone https://github.com/turoad/lanedet.git /lanedet
WORKDIR /lanedet
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ && \
  pip install --no-cache-dir -e .
