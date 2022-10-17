# Copyright 2022 Woven Planet. All rights reserved.
FROM dgp:latest

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
  openjdk-11-jre-headless

ARG WORKSPACE=/home/dgp2wicker
WORKDIR ${WORKSPACE}
COPY . ${WORKSPACE}
COPY sample_wickerconfig.json /root/wickerconfig.json
RUN pip install --editable .
ENV PYTHONPATH="${WORKSPACE}:$PYTHONPATH"
