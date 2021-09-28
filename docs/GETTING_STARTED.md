Getting Started
===============

## Prerequisites

- Linux or macOS
- Python 3.6+
- [docker](https://docs.docker.com/engine/install/)
- CUDA 10.0+

## Installation

Dcoekrized environment is encouraged for all DGP contributors and users. One can build the docker image locally via:

```sh
make docker-build
```

To check if DGP docker image is built successfully, run the unit tests via:

```sh
make docker-run-tests
```

Alternatively, one can use python virtual environments. Please see [virtual environment setup](VIRTUAL_ENV.md) for instructions.


## Develop within docker
In order to start development, the quickest way to get started would
be use the interactive docker mode via:
```sh
make docker-start-interactive
```
The DGP base directory is mounted within the
docker container, and gives you a sandbox to develop quickly without
needing to set up a local virtual environment.

Within the interactive docker container (after `make docker-start-interactive`), you can now build the proto definitions (`make build-proto`) and run the tests (`make test`) to make sure everything is functioning properly.
```sh
make build-proto
make test
```