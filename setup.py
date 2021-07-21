#!/usr/bin/env python

import setuptools

setuptools.setup(
    name             = 'toytools',
    version          = '0.0.1',
    author           = 'LS4GAN Group',
    author_email     = 'dtorbunov@bnl.gov',
    classifiers      = [
        'Programming Language :: Python :: 3 :: Only',
    ],
    description      = 'A collection of functions to handle toyzero dataset',
    packages         = setuptools.find_packages(
        include = [ 'toytools', 'toytools.*' ]
    ),
    scripts          = [ 'scripts/preprocess', 'scripts/view_dataset', ],
    install_requires = [ 'numpy', 'pandas', 'matplotlib' ],
)

