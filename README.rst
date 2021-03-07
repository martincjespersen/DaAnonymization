.. figure:: docs/imgs/header.png
    :width: 150px
    :align: center

==================================
Anonymization tool for Danish text
==================================

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

* Free software: Apache-2.0 license

**Disclaimer:** As the pipeline utilizes predictive models and regex function to identify entities, there is no guarantee that all sensitive information have been remove.

Features
--------

- Regex for  CPRs, telephone numbers, emails
- Integration of custom functions as part of the pipeline
- Named Entity Models for Danish language implemented (PER, LOC, ORG):
    - DaCy: https://github.com/KennethEnevoldsen/DaCy
    - DaNLP: https://github.com/alexandrainst/danlp

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


Quickstart
----------
DaAnonymization's main component **TextAnonymizer** uses its ``mask_corpus`` function to anonymize text by removing person, location, organization, email, telephone number and CPR. The order of these masking methods are by default CPR, telephone number, email and NER (PER,LOC,ORG) as NER will identify names in the emails.

.. code-block:: python

    from textanonymization import textanonymization

    # list of texts
    corpus = [
        "Hej, jeg hedder Martin Jespersen og er fra Danmark og arbejder i "
        "Deloitte, mit cpr er 010203-2010, telefon: +4545454545 "
        "og email: martin.martin@gmail.com",
    ]

    Anonymizer = textanonymization.TextAnonymizer(corpus)

    # load danlp as NER model
    Anonymizer._load_NER_model("danlp")

    # Anonymize person, location, organization, emails, CPR and telephone numbers
    anonymized_corpus = Anonymizer.mask_corpus()

    for text in anonymized_corpus:
        print(text)


Running this script outputs the following:

.. code-block:: console

    Hej, jeg hedder [PERSON] og er fra [LOKATION] og arbejder i [ORGANISATION], mit cpr er [CPR],
    telefon: [TELEFON] og email: [EMAIL]

Using custom masking functions
------------------------------
As projects are very different on needs, DaAnonymization supports adding custom functions for masking additional features which are not implemented by default.

.. code-block:: python

    from textanonymization import textanonymization

    # Takes string as input and returns a masked version of the string
    example_custom_function = lambda x: x.replace('20 år', '[ALDER]')

    # list of texts
    corpus = [
        "Hej, jeg hedder Martin Jespersen, er 20 år, er fra Danmark og arbejder i "
        "Deloitte, mit cpr er 010203-2010, telefon: +4545454545 "
        "og email: martin.martin@gmail.com",
    ]

    Anonymizer = textanonymization.TextAnonymizer(corpus)

    # load danlp as NER model
    Anonymizer._load_NER_model("danlp")

    # add the name to masking_methods in the desired order
    # add custom function to custom_functions to update pool of possible masking functions
    anonymized_corpus = Anonymizer.mask_corpus(
        masking_methods=["cpr", "telefon", "email", "NER", "alder"],
        custom_functions={"alder": example_custom_function},
    )

    for text in anonymized_corpus:
        print(text)

.. code-block:: console

    Hej, jeg hedder [PERSON], er [ALDER], er fra [LOKATION] og arbejder i [ORGANISATION],
    mit cpr er [CPR], telefon: [TELEFON] og email: [EMAIL]

Next up
--------

* More comprehensive tests on larger corpus'
* Evaluate performance of the various pipelines on larger (synthetic?) corpus'
* Test NER models for possible bias with person entities
* Make DaCy model path flexible (use environment variable instead of fixed path)


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
