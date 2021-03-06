#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "joblib==1.0.1",
    "spacy==3.0.3",
    "torch==1.8.0",
    "spacy-transformers==1.0.1",
    "transformers==3.1.0",
    "danlp",
    "gensim==3.8.1",
    "pip==19.2.3",
    "Click==7.0",
    "pytest==4.6.5",
    "pytest-runner==5.1",
]

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Martin Closter Jespersen",
    author_email="martincjespersen@gmail.com",
    python_requires=">=3.5",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache-2.0 License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Python Boilerplate contains all the boilerplate you need to create a Python package.",
    entry_points={
        "console_scripts": [
            "textanonymization=textanonymization.cli:main",
        ],
    },
    install_requires=requirements,
    license="Apache license Version 2.0",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="DaAnonymization",
    name="DaAnonymization",
    packages=find_packages(include=["textanonymization", "textanonymization.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/martincjespersen/DaAnonymization",
    version="0.1.0",
    zip_safe=False,
)
