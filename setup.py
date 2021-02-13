#import setuptools
from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="paldnn", 
    version="0.1.0",
    author="Eric Bruckner",
    author_email="eric.p.bruckner@gmail.com",
    description="A package for building a neural network to predict Peptide Amphiphile nanostructures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(include = 'paldnn'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3',
)