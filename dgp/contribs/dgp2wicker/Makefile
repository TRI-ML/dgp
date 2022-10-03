# Copyright 2022 Woven Planet. All rights reserved.
PYTHON ?= python3
PACKAGE_NAME ?= dgp2wicker
WORKSPACE ?= /home/$(PACKAGE_NAME)
DOCKER_IMAGE_NAME ?= $(PACKAGE_NAME)
DOCKER_IMAGE ?= $(DOCKER_IMAGE_NAME):latest
DOCKER_OPTS ?= \
	-it \
	--rm \
	--shm-size=62G \
	-e AWS_DEFAULT_REGION \
	-e AWS_ACCESS_KEY_ID \
	-e AWS_SECRET_ACCESS_KEY \
	-e AWS_SESSION_TOKEN\
	-e AWS_PROFILE \
	-e VAULT_ASSUMED_ROLE \
	-e WICKER_CONFIG_PATH \
	-e DISPLAY=${DISPLAY} \
	-v $(PWD):$(WORKSPACE)

develop:
	pip install --editable .

clean:
	$(PYTHON) setup.py clean && \
	rm -rf build dist && \
	find . -name "*.pyc" | xargs rm -f && \
	find . -name "__pycache__" | xargs rm -rf
	find . -name "*egg-info" | xargs rm -rf

docker-build:
	docker build \
	--build-arg WORKSPACE=$(WORKSPACE) \
	-t $(DOCKER_IMAGE) .

docker-run:
	docker run \
	--name $(PACKAGE_NAME) \
	$(DOCKER_OPTS) $(DOCKER_IMAGE) $(COMMAND)

docker-start-interactive:
	docker run \
	$(DOCKER_OPTS) \
	$(DOCKER_IMAGE) bash