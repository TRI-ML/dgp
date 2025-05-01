# Copyright 2019-2021 Toyota Research Institute.  All rights reserved.
PYTHON_EXEC ?= python3
PACKAGE_NAME ?= dgp
WORKSPACE ?= /home/$(PACKAGE_NAME)
DOCKER_IMAGE_NAME ?= $(PACKAGE_NAME)
DOCKER_IMAGE ?= $(DOCKER_IMAGE_NAME):latest
DOCKER_EXTRA_OPTS ?=
DOCKER_COMMON_OPTS ?= \
	-it \
	--rm \
	--shm-size=1G \
	-e AWS_DEFAULT_REGION \
	-e AWS_ACCESS_KEY_ID \
	-e AWS_SECRET_ACCESS_KEY \
	-e DISPLAY=${DISPLAY} \
	-v $(PWD):$(WORKSPACE) \
	-v /var/run/docker.sock:/var/run/docker.sock \
	-v /tmp/.X11-unix/X0:/tmp/.X11-unix/X0 \
	--net=host --ipc=host \
	$(DOCKER_EXTRA_OPTS)
DOCKER_ROOT_OPTS ?= $(DOCKER_COMMON_OPTS) \
	-v ~/.ssh:/root/.ssh \
	-v ~/.aws:/root/.aws

# DGP_xxx are environment variables for "scripts/with_the_same_user".
DOCKER_USER_OPTS ?= $(DOCKER_COMMON_OPTS) \
	-v ~/.ssh:$(HOME)/.ssh \
	-v ~/.aws:$(HOME)/.aws \
	-e DGP_HOME=$(HOME) \
	-e DGP_USER=$$(id -u -n) \
	-e DGP_UID=$$(id -u) \
	-e DGP_GROUP=$$(id -g -n) \
	-e DGP_GID=$$(id -g)

# Unit tests
UNITTEST ?= pytest
UNITTEST_OPTS ?= -v

.PHONY: clean build develop

all: clean test


build:
	PYTHONPATH=$(PWD):$(PYTHONPATH) \
	DGP_DEV_VERSION=$(DEV_VERSION) $(PYTHON_EXEC) setup.py bdist_wheel

clean:
	rm -rf build dist && \
	find . -name "*.pyc" | xargs rm -f && \
	find . -name "__pycache__" | xargs rm -rf
	find . -name "*egg-info" | xargs rm -rf && \
	find dgp/proto -name "*_grpc.py" | xargs rm -rf
	find dgp/proto -name "*_pb2.py" | xargs rm -rf
	find dgp/contribs/pd -name "*_pb2.py" | xargs rm -rf
	find . -name "*eggs" | xargs rm -rf &
	$(PYTHON_EXEC) setup.py clean

develop:
	DGP_DEV_VERSION=$(DEV_VERSION) $(PYTHON_EXEC) -m pip install --editable ".[dev]"

docker-build:
	docker build \
	--build-arg WORKSPACE=$(WORKSPACE) \
	-t $(DOCKER_IMAGE) .

docker-exec: docker-build
	docker exec -it $(DOCKER_IMAGE_NAME) $(COMMAND)

docker-run-tests: docker-build
	docker run --name $(DOCKER_IMAGE_NAME)-tests $(DOCKER_ROOT_OPTS) $(DOCKER_IMAGE) make test


docker-start-interactive: docker-build
	docker run \
	$(DOCKER_USER_OPTS) \
	$(DOCKER_IMAGE) \
	bash --login $(WORKSPACE)/scripts/with_the_same_user bash

docker-stop:
	docker stop $(DOCKER_IMAGE_NAME)

setup-linters:
	pre-commit install
	pre-commit install --hook-type commit-msg

test: develop
	PYTHONPATH=$(PWD):$(PYTHONPATH) \
	$(UNITTEST) $(UNITTEST_OPTS) $(PWD)/tests/ -vv

unlink-githooks:
	unlink .git/hooks/pre-push && unlink .git/hooks/pre-commit
