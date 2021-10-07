# Getting Started

## Prerequisites

- Linux or macOS
- Python 3.6+
- [docker](https://docs.docker.com/engine/install/)
- CUDA 10.0+
- (Optional) [AWS](AWS.md)

## Installation

Dockerized environment is encouraged for all DGP contributors and users. Alternatively, you can use python virtual environments. Please see [virtual environment setup](VIRTUAL_ENV.md) for instructions.

To setup DGP docker image:

- You can pull the latest master docker via:

   ```sh
   dgp$ docker pull ghcr.io/tri-ml/dgp:master && docker image tag ghcr.io/tri-ml/dgp:master dgp:latest
   ```
   ---or---

- Build the docker from scratch via:

    ```sh
    dgp$ make docker-build
    ```

Inspect if docker image `dgp:latest` has been pulled or built successfully:

```sh
dgp$ docker inspect --type=image dgp:latest
```
If you get a response, then you already have DGP docker image on the machine!

To check if DGP docker image is built successfully, run the unit tests via:

```sh
dgp$ make docker-run-tests
```

## Develop within docker
In order to start development, the quickest way to get started would
be use the interactive docker mode via:
```sh
dgp$ make docker-start-interactive
```
The DGP base directory is mounted within the
docker container, and gives you a sandbox to develop quickly without
needing to set up a local virtual environment.

Within the interactive docker container (after `make docker-start-interactive`), you can now build the proto definitions (`make build-proto`) and run the tests (`make test`) to make sure everything is functioning properly.
```sh
dgp$ make build-proto
dgp$ make test
```