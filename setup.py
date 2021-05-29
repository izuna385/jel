#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import sys
from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as readme_file:
    readme = readme_file.read()

info = sys.version_info

setup(
    name='jel',
    version='0.1.1',
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
    test_suite="test",
)