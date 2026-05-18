FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG python=3.10
ENV PYTORCH_VERSION=2.3.1+cu118
ENV TORCHVISION_VERSION=0.18.1+cu118
ENV CUDA_VERSION_SHORT=cu118

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PYTHON_VERSION=${python}
ENV DEBIAN_FRONTEND=noninteractive

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
  build-essential \
  ca-certificates \
  curl \
  libgl1-mesa-glx \
  libgtk2.0-dev \
  libjpeg-dev \
  libpng-dev \
  python${PYTHON_VERSION} \
  python${PYTHON_VERSION}-dev \
  && rm -rf /var/lib/apt/lists/*
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
  python3 get-pip.py && \
  rm get-pip.py
# The above appears to install pip into /usr/local/, but some tools expect /usr/bin/.
RUN ln -sf /usr/local/bin/pip /usr/bin/pip
RUN ln -sf /usr/local/bin/pip3 /usr/bin/pip3

# Install Pytorch
RUN pip install --no-cache-dir \
  torch==${PYTORCH_VERSION} \
  torchvision==${TORCHVISION_VERSION} \
  -f https://download.pytorch.org/whl/${CUDA_VERSION_SHORT}/torch_stable.html


# Install python dependencies
ARG WORKSPACE=/home/dgp
WORKDIR ${WORKSPACE}
COPY requirements.txt requirements-dev.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements-dev.txt

# Settings for S3
RUN aws configure set default.s3.max_concurrent_requests 100 && \
  aws configure set default.s3.max_queue_size 10000

# Copy workspace and setup PYTHONPATH
COPY . ${WORKSPACE}
ENV PYTHONPATH="${WORKSPACE}"
