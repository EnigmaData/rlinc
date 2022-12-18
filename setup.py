from setuptools import setup, find_packages
import pathlib

HERE = pathlib.Path(__file__).parent

setup(
    name='rlinc',
    version='0.0.1',
    description='A python package for RL bandits, looking forward to expanding it',
    long_description=(HERE / "README.md").read_text(),
    license="MIT",
    author='Shubh Agarwal',
    author_email='shubhag3110@gmail.com',
    install_requires=[
        'numpy>=1.17.0',
        'wheel'
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
