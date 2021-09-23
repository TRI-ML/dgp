Virtual Environment Installation
================================
DGP can be installed into a virtual environment and is a convenient way of developing locally and testing small changes.

To make a new virtual environment named ```env```:

```sh
python3 -m venv env
```

Activate your virtual environment.
```sh
source env/bin/activate
```
 While active, running python will run the virtual environment's python, and installing packages with pip will install them to the virtual environment's site-packages. You will need to install a few dependencies prior to installing DGP.

```sh
pip install --upgrade pip
pip install cython==0.29.10 numpy==1.19.4 protobuf==3.6.1
```

Finally DGP can be installed. Installing in editable mode means that changes you make to DGP locally will be reflected in the version installed in pip which makes local development very convenient. From DGP repo root:
```sh
pip install --editable .
```
Alternatively, you can install cython, numpy, protobuf and DGP by running:
```sh
make develop
```

To check if DGP is installed correctly, run the unit tests by running:
```sh
make test
``` 

While most changes will be reflected automatically in editable mode, any changes to the protos, which require their own build step, will need to be run manually. To update any changes to the protos, please run make develop again. 


Finally to deactivate the environment
```sh
deactivate
```
