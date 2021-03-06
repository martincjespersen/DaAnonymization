====================================================
DaAnonymization - Anonymization tool for Danish text
====================================================


.. image:: https://img.shields.io/pypi/v/textanonymization.svg
        :target: https://pypi.python.org/pypi/textanonymization

.. image:: https://img.shields.io/travis/martincjespersen/textanonymization.svg
        :target: https://travis-ci.com/martincjespersen/textanonymization

.. image:: https://readthedocs.org/projects/textanonymization/badge/?version=latest
        :target: https://textanonymization.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/martincjespersen/textanonymization/shield.svg
     :target: https://pyup.io/repos/github/martincjespersen/textanonymization/
     :alt: Updates

Description
-----------
A simple pipeline wrapped around SpaCy, DaNLP and DaCy for anonymizing danish corpora. The pipeline allows for custom functions to be implemented and piped in combination with custom functions.

* Disclaimer: As the pipeline utilizes predictive models and regex function to identify entities, there is no guarantee that all sensitive information have been remove.

* Free software: Apache license Version 2.0

Installation
------------
To install from source:

.. code-block:: bash
    git clone https://github.com/martincjespersen/DaAnonymization.git
    cd DaAnonymization
    python setup.py install

A prerequisite for the SpaCy nlp to run, the Danish version has to be installed running the following command:

.. code-block:: bash

    python -m spacy download da_core_news_sm


**Note:**
To enable DaCy as a NER model you need to download the **large model** folder and place it within the root of the repository. Follow the instructions here:
https://github.com/KennethEnevoldsen/DaCy

Features
--------

- Regex for  CPRs, telephone numbers, emails
- Integration of custom functions as part of the pipeline
- Named Entity Models for Danish language implemented (PER, LOC, ORG):
    - DaCy: https://github.com/KennethEnevoldsen/DaCy
    - DaNLP: https://github.com/alexandrainst/danlp

Next up
--------

* Add tests for integration of custom functions
* More comprehensive tests on larger corpus'
* Evaluate performance of the various pipelines on larger (synthetic?) corpus'
* Test NER models for possible bias with person entities
* Make DaCy model path flexible (use environment variable instead of fixed path)


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
