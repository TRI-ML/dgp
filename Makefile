# Copyright 2019-2020 Toyota Research Institute.  All rights reserved.
PYTHON ?= python3
PACKAGE_NAME ?= dgp
WORKSPACE ?= /home/$(PACKAGE_NAME)
DOCKER_IMAGE_NAME ?= $(PACKAGE_NAME)
DOCKER_IMAGE ?= $(DOCKER_IMAGE_NAME):latest
DOCKER_OPTS ?= \
	-it \
	--rm \
	--shm-size=1G \
	-e DISPLAY=${DISPLAY} \
	-v $(PWD):$(WORKSPACE) \
	-v /var/run/docker.sock:/var/run/docker.sock \
	-v ~/.ssh:/root/.ssh \
	-v /tmp/.X11-unix/X0:/tmp/.X11-unix/X0 \
	--net=host --ipc=host

# Unit tests
UNITTEST ?= nosetests
UNITTEST_OPTS ?= --nologcapture -v -s

all: clean test

clean:
	$(PYTHON) setup.py clean && \
	rm -rf build dist && \
	find . -name "*.pyc" | xargs rm -f && \
	find . -name "__pycache__" | xargs rm -rf
	find . -name "*egg-info" | xargs rm -rf && \
	find dgp/proto -name "*_grpc.py" | xargs rm -rf
	find dgp/proto -name "*_pb2.py" | xargs rm -rf
	find dgp/contribs/pd -name "*_pb2.py" | xargs rm -rf

build-proto:
	PYTHONPATH=$(PWD):$(PYTHONPATH) \
	$(PYTHON) setup.py build_py

test: clean build-proto
	PYTHONPATH=$(PWD):$(PYTHONPATH) \
	$(UNITTEST) $(UNITTEST_OPTS) $(PWD)/tests/

docker-build:
	docker build \
	--build-arg WORKSPACE=$(WORKSPACE) \
	-t $(DOCKER_IMAGE) .

docker-start-interactive:
	nvidia-docker run \
	$(DOCKER_OPTS) \
	$(DOCKER_IMAGE) bash

docker-start:
	nvidia-docker run \
	-d --name $(DOCKER_IMAGE_NAME) \
	$(DOCKER_OPTS) $(DOCKER_IMAGE)

docker-exec:
	nvidia-docker exec -it $(DOCKER_IMAGE_NAME) $(COMMAND)

docker-stop:
	docker stop $(DOCKER_IMAGE_NAME)

docker-run-tests: build-proto
	nvidia-docker run \
	--name $(DOCKER_IMAGE_NAME)-tests \
	$(DOCKER_OPTS) $(DOCKER_IMAGE) \
	$(UNITTEST) $(UNITTTEST_OPTS) $(WORKSPACE)/tests

docker-start-visualizer:
	nvidia-docker run \
	--name $(DOCKER_IMAGE_NAME) \
	$(DOCKER_OPTS) $(DOCKER_IMAGE) \
	streamlit run $(WORKSPACE)/dgp/scripts/visualizer.py
