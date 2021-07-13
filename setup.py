#!/usr/bin/env python
from setuptools import find_packages
from setuptools import setup

VERSION = '0.1.0'

INSTALL_REQ = [
''
]

with open('README.md', 'r') as rmf:
    readme = rmf.read()

setup(
    version=VERSION,
    name='analyze2p',
    author='Juliana Rhee',
    packages=find_packages(),
    author_email='juliana.rhee@gmail.com',
    url="https://github.com/julianarhee/rat-2p-area-characterizations",
    description="Analysis for rat imaging data",
    long_description=readme,
    # Installation requirements
    install_requires= INSTALL_REQ,
    data_files=[('', ['README.md'])]
)



