# Copyright 2019 Toyota Research Institute. All rights reserved.
import importlib
import os

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.install import install


def build_protos():
    SETUP_DIR = os.path.dirname(os.path.abspath(__file__))
    from grpc.tools import command
    command.build_package_protos(SETUP_DIR)


class CustomBuildPyCommand(build_py):
    def run(self):
        build_protos()
        build_py.run(self)


class CustomInstallCommand(install):
    def run(self):
        build_protos()
        install.run(self)


class CustomDevelopCommand(develop):
    def run(self):
        build_protos()
        develop.run(self)


__version__ = importlib.import_module('dgp').__version__

with open('requirements.txt', encoding='utf-8') as f:
    requirements = f.read().splitlines()

with open('requirements-dev.txt', encoding='utf-8') as f:
    requirements_dev = f.read().splitlines()

packages = find_packages(exclude=['tests'])
setup(
    name="dgp",
    version=__version__,
    description="Dataset Governance Policy (DGP) for Autonomous Vehicle ML datasets.",
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author="Toyota Research Institute",
    author_email='ml@tri.global',
    url="https://github.com/TRI-ML/dgp",
    packages=packages,
    entry_points={'console_scripts': [
        'dgp_cli=dgp.cli:cli',
    ]},
    include_package_data=True,
    setup_requires=['cython==0.29.21', 'grpcio==1.41.0', 'grpcio-tools==1.41.0'],
    install_requires=requirements,
    extras_require={'dev': requirements_dev},
    zip_safe=False,
    python_requires='>=3.6',
    cmdclass={
        'install': CustomInstallCommand,
        'develop': CustomDevelopCommand,
        'build_py': CustomBuildPyCommand
    }
)
