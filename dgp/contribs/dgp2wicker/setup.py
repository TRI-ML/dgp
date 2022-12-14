# Copyright 2022 Woven Planet NA. All rights reserved.
"""Setup.py for dgp2wicker"""
import importlib

from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.install import install


class CustomInstallCommand(install):
    """Custom install command"""
    def run(self):
        install.run(self)


class CustomDevelopCommand(develop):
    """Custom develop command"""
    def run(self):
        develop.run(self)


__version__ = importlib.import_module('dgp2wicker').__version__

with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = f.read().splitlines()

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

packages = find_packages(exclude=['tests'])
setup(
    name="dgp2wicker",
    version=__version__,
    description="Tools to convert TRI's DGP to L5's Wicker format.",
    long_description=readme,
    long_description_content_type='text/markdown',
    author="Chris Ochoa, Kuan Lee",
    author_email='charles.ochoa@woven-planet.global, kuan-hui.lee@woven-planet.global',
    url="https://github.com/TRI-ML/dgp/tree/master/dgp/contribs/dgp2wicker",
    packages=packages,
    entry_points={'console_scripts': [
        'dgp2wicker=dgp2wicker.cli:cli',
    ]},
    include_package_data=True,
    setup_requires=['cython==0.29.21', 'grpcio==1.41.0', 'grpcio-tools==1.41.0'],
    install_requires=requirements,
    zip_safe=False,
    python_requires='>=3.7',
    cmdclass={
        'install': CustomInstallCommand,
        'develop': CustomDevelopCommand,
    }
)
