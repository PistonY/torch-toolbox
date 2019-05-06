# -*- coding: utf-8 -*-
# Author: pistonyang@gmail.com

from setuptools import setup, find_packages

VERSION = '0.0.1'

setup(
    name='torchtools',
    version=VERSION,
    author='X.Yang',
    author_email='pistonyang@gmail.com',
    url='https://github.com/deeplearningforfun/torch-tools',
    description='Collect and write useful tools for Pytorch.',
    long_description=open('README.md').read(),
    license='MIT',
    packages=find_packages(exclude=('*test*',)),
    zip_safe=True,
    classifiers=[
        'Programming Language :: Python :: 3',
    ]
)
