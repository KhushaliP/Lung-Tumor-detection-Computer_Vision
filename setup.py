#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='lung-tumor-segmentation',
    version='0.1.1',
    description='Lung tumor segmentation project',
    author='khushali',
    author_email='khushalibpatel@gmail.com',
    install_requires=['pytorch-lightning'],
    packages=find_packages(),
)

