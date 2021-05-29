#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
import sys
from setuptools import setup, find_packages
from codecs import open
from os import path
root_dir = path.abspath(path.dirname(__file__))

with open('README.md', encoding='utf-8') as readme_file:
    readme = readme_file.read()

info = sys.version_info

def _requirements():
    return [name.rstrip() for name in open(path.join(root_dir, 'requirements.txt')).readlines()]

setup(
    name='jel',
    version='0.1.3',
    description='Japanese Entity Linker.',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='izuna385',
    author_email='izuna385@gmail.com',
    url='https://github.com/izuna385/jel',
    packages=find_packages(),
    include_package_data=True,
    keywords='jel',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Natural Language :: English',
        "License :: OSI Approved :: Apache Software License",
        'Programming Language :: Python :: 3.7',
        "Operating System :: OS Independent",
    ],
    install_requires=_requirements(),
    test_suite="test",
)