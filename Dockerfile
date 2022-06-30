FROM nvidia/cuda:11.1-devel-ubuntu18.04

ARG python=3.7
ENV PYTORCH_VERSION=1.8.1+cu111
ENV TORCHVISION_VERSION=0.9.1+cu111

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PYTHON_VERSION=${python}
ENV DEBIAN_FRONTEND=noninteractive

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

# Temporary fix for invalid GPG key see
# https://forums.developer.nvidia.com/t/gpg-error-http-developer-download-nvidia-com-compute-cuda-repos-ubuntu1804-x86-64/212904
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

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
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
  python get-pip.py && \
  rm get-pip.py

# Install Pytorch
RUN pip install --no-cache-dir \
  torch==${PYTORCH_VERSION} \
  torchvision==${TORCHVISION_VERSION} \
  -f https://download.pytorch.org/whl/${PYTORCH_VERSION/*+/}/torch_stable.html

# Install python dependencies
ARG WORKSPACE=/home/dgp
WORKDIR ${WORKSPACE}
COPY requirements.txt requirements-dev.txt /tmp/
RUN pip install --no-cache-dir cython==0.29.21 numpy==1.19.4
RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements-dev.txt

# Settings for S3
RUN aws configure set default.s3.max_concurrent_requests 100 && \
    aws configure set default.s3.max_queue_size 10000

# Copy workspace and setup PYTHONPATH
COPY . ${WORKSPACE}
ENV PYTHONPATH="${WORKSPACE}:$PYTHONPATH"
