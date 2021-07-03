# this is for rsaicp

ARG PYTORCH="1.7.1"
ARG CUDA="11.0"
ARG CUDNN="8"

# FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
FROM image.rsaicp.com/base/cuda:11.0-cudnn8-runtime-ubuntu18.04

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
        build-essential \
        cmake \
        curl \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        python3 \
        python3-pip \
        python3-setuptools \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

# Install pytorch and  mmcv
# RUN conda clean --all
RUN pip3 install --upgrade pip
RUN pip install torch===1.7.1+cu110 torchvision===0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
RUN mkdir /work

# RUN git clone https://github.com/open-mmlab/mmsegmenation.git /mmsegmentation
# WORKDIR /mmsegmentation
COPY /home/csl/code/rsaicp_CD /work
# RUN cd /work/
# RUN pip install -r requirements/build.txt
# RUN pip install --no-cache-dir -e .

CMD ["python3", "-u", "/work/main.py"]