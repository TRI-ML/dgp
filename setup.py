# Copyright 2025 Toyota Research Institute. All rights reserved.
import logging
import os
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install import install

logger = logging.getLogger(__file__)

__version__ = "2.0.0"

_ROOT_DIRPATH = Path(__file__).parent.absolute()

# Specify the dev version for development.
_DEV_VERSION = str(os.environ.get("DGP_DEV_VERSION", ""))

_VERSION = f"{__version__}.{_DEV_VERSION}" if _DEV_VERSION else __version__

with open("requirements.txt", encoding="utf-8") as f:
    requirements = f.read().splitlines()

with open("requirements-dev.txt", encoding="utf-8") as f:
    requirements_dev = f.read().splitlines()

install_requires = requirements + [
    "protobuf>=4.0.0,<5.0.0",
]
setup_requires = [
    "protobuf>=4.0.0,<5.0.0",
    "grpcio-tools<1.66.0",  # for protobuf 4.X.X support.
]


def _build_py():
    from grpc_tools import command

    command.build_package_protos(_ROOT_DIRPATH)


class _CustomBuildPyCommand(build_py):
    def run(self):
        _build_py()
        build_py.run(self)


class _CustomInstallCommand(install):
    def run(self):
        _build_py()
        install.run(self)


class _CustomDevelopCommand(develop):
    def run(self):
        _build_py()
        develop.run(self)


packages = find_packages(exclude=["tests"])

setup(
    name="dgp",
    version=_VERSION,
    description="Dataset Governance Policy (DGP) for Autonomous Vehicle ML datasets.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Toyota Research Institute",
    author_email="ml@tri.global",
    url="https://github.com/TRI-ML/dgp",
    packages=packages,
    entry_points={
        "console_scripts": [
            "dgp_cli=dgp.cli:cli",
        ],
    },
    include_package_data=True,
    install_requires=install_requires,
    setup_requires=setup_requires,
    extras_require={
        "dev": requirements_dev,
    },
    zip_safe=False,
    python_requires=">=3.8",
    cmdclass={
        "build_py": _CustomBuildPyCommand,
        "install": _CustomInstallCommand,
        "develop": _CustomDevelopCommand,
    },
)
