# -*- coding: utf-8 -*-
# Author: pistonyang@gmail.com

from setuptools import setup, find_packages
from torchtoolbox import VERSION

setup(
    name='torchtoolbox',
    version=VERSION,
    author='X.Yang',
    author_email='pistonyang@gmail.com',
    url='https://https://github.com/PistonY/torch-toolbox',
    description='ToolBox to make using Pytorch much easier.',
    long_description_content_type="text/markdown",
    long_description=open('README.md').read(),
    license='BSD 3-Clause',
    packages=find_packages(exclude=('*tests*',)),
    zip_safe=True,
    classifiers=[
        'Programming Language :: Python :: 3',
    ], install_requires=['numpy', 'tqdm', 'pyarrow', 'six', 'lmdb', 'scikit-learn', 'scipy']
)
