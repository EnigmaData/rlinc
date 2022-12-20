"""Setup file for rlinc package, added because of stupid pylint"""
import pathlib
from setuptools import setup, find_packages

from rlinc import VERSION, AUTHOR

HERE = pathlib.Path(__file__).parent

setup(
    name='rlinc',
    version=VERSION,
    description='A python package for RL bandits, looking forward to expanding it',
    long_description=(HERE / "README.md").read_text(),
    license="MIT",
    author=AUTHOR['name'],
    author_email=AUTHOR['email'],
    install_requires=[
        'numpy>=1.17.0',
        'matplotlib>=3.2.0'
    ],
    packages=find_packages(),
    url='https://github.com/EnigmaData/rlinc/',
    classifiers=[
        "Development Status :: 1 - Planning",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: Implementation :: CPython"
    ],
    python_requires='>3.9.0'
)
