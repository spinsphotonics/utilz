#!/usr/bin/env python

import codecs
import os

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

HERE = os.path.dirname(os.path.realpath(__file__))


def read(*parts):
  with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
    return f.read()


setup(
    name="spins-utilz",
    author="Jesse Lu",
    author_email="jesselu@spinsphotonics.com",
    url="https://github.com/spinsphotonics/utilz",
    license="MIT",
    description=("Utilities for differentiable computational photonics"),
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    install_requires=["jax", "jaxlib"],
    setup_requires=["setuptools_scm"],
    use_scm_version=True,
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    extras_require={"test": "pytest"},
)
