FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

ARG python=3.6

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV PYTHON_VERSION=${python}
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
  build-essential \
  ca-certificates \
  libgl1-mesa-glx \
  libgtk2.0-dev \
  libjpeg-dev \
  libpng-dev \
  python${PYTHON_VERSION} \
  python${PYTHON_VERSION}-dev
RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
  python get-pip.py && \
  rm get-pip.py

# Repo specific dependencies
# Setup requirements in workspace
ARG WORKSPACE=/home/dgp
WORKDIR ${WORKSPACE}
COPY dev_requirements.txt /tmp/
RUN pip install cython==0.29.10 numpy==1.16.3
RUN pip install -r /tmp/dev_requirements.txt --ignore-installed

# Settings for S3
RUN aws configure set default.s3.max_concurrent_requests 100 && \
    aws configure set default.s3.max_queue_size 10000

# Copy workspace and setup PYTHONPATH
COPY . ${WORKSPACE}
ENV PYTHONPATH="${WORKSPACE}:$PYTHONPATH"

# Set up streamlit configs
ARG STREAMLIT_CONFIG_DIR=/root/.streamlit
RUN mkdir -p ${STREAMLIT_CONFIG_DIR} && \
    touch ${STREAMLIT_CONFIG_DIR}/credentials.toml && \
    printf '[general]\nemail = "dummy@domain.com"' > ${STREAMLIT_CONFIG_DIR}/credentials.toml
