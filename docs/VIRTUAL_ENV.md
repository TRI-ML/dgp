# Virtual Environment Installation

DGP can be installed into a virtual environment and is a convenient way of
developing locally and testing small changes.

To make a new virtual environment named `env`:

```sh
dgp$ python3 -m venv env
```

Activate your virtual environment.

```sh
dgp$ source env/bin/activate
```

While active, running python will run the virtual environment's python, and
installing packages with pip will install them to the virtual environment's
site-packages. You will need to install a few dependencies prior to installing
DGP.

```sh
dgp$ pip install --upgrade pip
dgp$ pip install cython==0.29.21 numpy==1.19.4
dgp$ pip install -r requirements.txt -r requirements-dev.txt
```

Finally DGP can be installed. Installing in editable mode means that changes you
make to DGP locally will be reflected in the version installed in pip which
makes local development very convenient. From DGP repository root:

```sh
dgp$ pip install --editable .
```

To check if DGP is installed correctly, run the unit tests by running:

```sh
dgp$ make test
```

Run DGP CLI via:

```sh
dgp$ dgp_cli --help
```

While most changes will be reflected automatically in editable mode, any changes
to the protos, which require their own build step, will need to be run manually.
To update any changes to the protos, please run make develop again.

Finally to deactivate the environment

```sh
dgp$ deactivate
```

To set up linters, run

```
dgp$ make setup-linters
```

Note that if your environment's Python version does not match the version DGP
targets, your linting experience might not match that in CI.
