#import setuptools
from distutils.core import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="paldnn", 
    packages=['paldnn'],
    version="0.0.1",
    author="Eric Bruckner",
    author_email="ericbruckner2016@u.northwestern.edu",
    description="A package for building a neural network to predict Peptide Amphiphile nanostructures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stupplab/paldnn.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3',
)
