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
- Named Entity Models for Danish language implemented (PER, LOC, ORG, MISC):
    - DaCy: https://github.com/KennethEnevoldsen/DaCy
        - DaCy is built on multilingual RoBERTa, enabling support for other languages as well as Danish.
    - DaNLP: https://github.com/alexandrainst/danlp
    - Default entities to mask: PER, LOC and ORG (MISC can be specified but covers many different entitites)
- Batch mode for DaCy, **highly recommended** if predicting a lot of documents and it is robust to language changes as it is fine tuned from a **multilingual model**

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
To enable DaCy as a NER model you need to download the **large model** folder and place it within the root of this repository or setting the path in an environmental variable ``DACY = path_to_dacyfolder``.
To download follow the instructions here: `KennethEnevoldsen GitHub <https://github.com/KennethEnevoldsen/DaCy>`_
Or alternatively here: `MartinCJ Google Drive <https://drive.google.com/file/d/1fHyYGG01pFdMpynerxl_JaX_XZh_z0kl/view?usp=sharing>`_


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
As each project can have specific needs, DaAnonymization supports adding custom functions to the pipeline for masking additional features which are not implemented by default.

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

Fairness evaluations
--------------------
**Disclaimer:** Working progress on: `Benchmark Fairness <https://colab.research.google.com/drive/1qVdP99ZSqROfalUh63DVJ-5A6MhWrT3_?usp=sharing>`_

In the table 1, the DaNE dataset had all first name resampled (female only due to data) first a sanity check using female first names with danish origin (Sampled Danish names) and following other female names with origins than Danish (Sampled Other names). Both F1 scores are sampled by reproducing it 100 times and averaging. A small drop is found using female danish names only and further dropped using names from different origins.


.. list-table:: Table 1: Fairness of names of other origin than Danish (**F1 scores**)
   :widths: 10 15 15 15
   :header-rows: 1

   * - Model
     - DaNE (original)
     - Sampled Danish names
     - Sampled Other names
   * - DaNLP NER
     - 92.8
     - 90.6
     - 89.0
   * - DaCy
     - 95.5
     - TBD
     - TBD


Next up
--------

* Add test on >512 tokens sentence
* Test NER models for possible bias with person entities
* Optimize predicting with DaNLP creating a modified prediction function
* Implement pseudonymization module (Person 1, Person 2 etc.)
* When SpaCy fixed multiprocessing in nlp.pipe, remove current hack
