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
A simple pipeline wrapped around SpaCy and DaCy for anonymizing danish corpora. The pipeline allows for custom functions to be implemented and piped in combination with custom functions.

The **DaCy model** is built on **multilingual RoBERTa** which enables **cross-lingual transfer** for other languagues ultimately providing a robust named entity recognition model for anonymization that is able to handle noisy Danish text data which could include other languages.

Languages used in the multilingual RoBERTa can be found in appendix A of XLM-RoBERTa paper: `Unsupervised Cross lingual Representation Learning at Scale <https://arxiv.org/pdf/1911.02116.pdf>`_

* Free software: Apache-2.0 license

**Disclaimer:** As the pipeline utilizes predictive models and regex function to identify entities, there is no guarantee that all sensitive information have been remove.

Features
--------

- Regex for  CPRs, telephone numbers, emails
- Integration of custom functions as part of the pipeline
- Named Entity Models for Danish language implemented (PER, LOC, ORG, MISC):
    - DaCy: https://github.com/KennethEnevoldsen/DaCy
    - Default entities to mask: PER, LOC and ORG (MISC can be specified but covers many different entitites)
    - Batch mode and multiprocessing
    - DaCy is robust to language changes as it is fine tuned from a **multilingual RoBERTa model**
- Allow anonymizing using suppression
- Allow masking to be aware of prior knowledge about individuals occuring in the texts
- Pseudonymization module (Person 1, Person 2 etc.)
- Logging to masked_corpus function, enabling tracking of warning if no person was found in a text
- Beta version: Masking or noising with laplace epsilon noise of numbers


Installation
------------
**1. First install from source**

.. code-block:: bash

    git clone https://github.com/martincjespersen/DaAnonymization.git
    cd DaAnonymization
    python setup.py install

**2. Install DaCy**

A requirement for DaAnonymization is DaCy, which can be installed following the instructions here: `Dacy GitHub <https://github.com/KennethEnevoldsen/DaCy>`_

Quickstart
----------
DaAnonymization's two main components are:

- TextAnonymizer
- TextPseudonymizer

Both components uses their ``mask_corpus`` function to anonymize/pseudonymize text by removing person, location, organization, email, telephone number and CPR. The order of these masking methods are by default CPR, telephone number, email and NER (PER,LOC,ORG) as NER will identify names in the emails. The following example shows an example of applying default anonymization and how it also **cross-lingual transfer to english**.

.. code-block:: python

    from textprivacy import TextAnonymizer

    # list of texts (example with cross-lingual transfer to english)
    corpus = [
        "Hej, jeg hedder Martin Jespersen og er fra Danmark og arbejder i "
        "Deloitte, mit cpr er 010203-2010, telefon: +4545454545 "
        "og email: martin.martin@gmail.com",
        "Hi, my name is Martin Jespersen and work in Deloitte. "
        "I used to be a PhD. at DTU in Machine Learning and B-cell immunoinformatics "
        "at Anker Engelunds Vej 1 Bygning 101A, 2800 Kgs. Lyngby.",
    ]

    Anonymizer = TextAnonymizer(corpus)

    # Anonymize person, location, organization, emails, CPR and telephone numbers
    anonymized_corpus = Anonymizer.mask_corpus()

    for text in anonymized_corpus:
        print(text)


Running this script outputs the following:

.. code-block:: console

    Hej, jeg hedder [PERSON] og er fra [LOKATION] og arbejder i [ORGANISATION], mit cpr er [CPR],
    telefon: [TELEFON] og email: [EMAIL]

    Hi, my name is [PERSON] and work in [ORGANISATION]. I used to be a PhD. at [ORGANISATION]
    in Machine Learning and B-cell immunoinformatics at [LOKATION].


Using custom masking functions
------------------------------
As each project can have specific needs, DaAnonymization supports adding custom functions to the pipeline for masking additional features which are not implemented by default.

.. code-block:: python

    from textprivacy import TextAnonymizer
    import re

    # Takes string as input and returns a set of all occurences
    example_custom_function = lambda x: set(list(re.findall(r"\d+ år", x)))

    # list of texts
    corpus = [
        "Hej, jeg hedder Martin Jespersen, er 20 år, er fra Danmark og arbejder i "
        "Deloitte, mit cpr er 010203-2010, telefon: +4545454545 "
        "og email: martin.martin@gmail.com",
    ]

    Anonymizer = TextAnonymizer(corpus)

    # update the mapping to include new custom function entity finder and replacement placeholder
    Anonymizer.mapping.update({"ALDER": "[ALDER]"})

    # add the name to masking_order in the desired order
    # add custom function to custom_functions to update pool of possible masking functions
    anonymized_corpus = Anonymizer.mask_corpus(
        masking_order=["CPR", "TELEFON", "EMAIL", "NER", "ALDER"],
        custom_functions={"ALDER": example_custom_function},
    )

    for text in anonymized_corpus:
        print(text)

.. code-block:: console

    Hej, jeg hedder [PERSON], er [ALDER], er fra [LOKATION] og arbejder i [ORGANISATION],
    mit cpr er [CPR], telefon: [TELEFON] og email: [EMAIL]



Pseudonymization with prior knowledge
-------------------------------------
Sometimes it can be useful to maintain some context regarding sensitive information within the text. Pseudonymization allows for maintaining the connection between entities while masking them. Essentially this means adding a unique identifier for each individual and their information in the text.

By using the **optional** input argument ``individuals``, you can add prior information about known individuals in the text you want to mask. The structure of individuals needs to be as shown below. The first dictionary provides a key for index of the text in the corpus, the next the unique identifier (integer) of the individuals and finally a dictionary of entities known prior for each individual.

.. code-block:: python

    from textprivacy import TextPseudonymizer

    # prior information about the text
    individuals = {1:
                    {1:
                        {'PER': set(['Martin Jespersen', 'Martin', 'Jespersen, Martin']),
                         'CPR': set(['010203-2010']),
                         'EMAIL': set(['martin.martin@gmail.com']),
                         'LOC': set(['Danmark']),
                         'ORG': set(['Deloitte'])
                         },
                    2:
                        {'PER': set(['Kristina']),
                         'ORG': set(['Novo Nordisk'])
                         }
                     }

                  }

    # list of texts
    corpus = [
        "Første tekst om intet, blot Martin",
        "Hej, jeg hedder Martin Jespersen og er fra Danmark og arbejder i "
        "Deloitte, mit cpr er 010203-2010, telefon: +4545454545 "
        "og email: martin.martin@gmail.com. Martin er en 20 årig mand. "
        "Kristina er en person som arbejder i Novo Nordisk. "
        "Frank er en mand som bor i Danmark og arbejder i Netto",
    ]

    Pseudonymizer = TextPseudonymizer(corpus, individuals=individuals)

    # Pseudonymize person, location, organization, emails, CPR and telephone numbers
    pseudonymized_corpus = Pseudonymizer.mask_corpus()

    for text in pseudonymized_corpus:
        print(text)


.. code-block:: console

    Første tekst om intet, blot Person 1

    Hej, jeg hedder Person 1 og er fra Lokation 1 og arbejder i Organisation 1, mit cpr er CPR 1,
    telefon: Telefon 5 og email: Email 1. Person 1 er en 20 årig mand. Person 2 er en person som
    arbejder i Organisation 2. Person 3 er en mand som bor i Lokation 1 og arbejder i Organisation 4



Fairness evaluations
--------------------
**Disclaimer:** Work in progress on: `Benchmark Fairness <https://colab.research.google.com/drive/1qVdP99ZSqROfalUh63DVJ-5A6MhWrT3_?usp=sharing>`_

In the table 1, the DaNE dataset had all first name resampled (female only due to data) first a sanity check using female first names with danish origin (Sampled Danish names) and following other female names with origins than Danish (Sampled Other names). Both F1 scores are sampled by reproducing it 100 times and average is reported together with standard deviation in the parentheses. The stars * on columns means tested on goldstandard tokens, where without means it own internal tokenizer was used (more realistic).

A drop is observed using female danish names only and further dropped using names from different origins for the DaNLP model, whereas the DaCy model seems to be more robust to foreign names.


.. list-table:: Table 1: Fairness of names of other origin than Danish (**F1 scores**)
   :widths: 10 15 15 15 15
   :header-rows: 1

   * - Model
     - DaNE
     - DaNE *
     - Sampled Danish names *
     - Sampled Other names *
   * - DaNLP NER
     - NA
     - 92.8
     - 90.5 (.013)
     - 88.8 (.016)
   * - DaCy
     - 95.5
     - 94.1
     - 94.2 (.008)
     - 93.8 (.009)


Next up
--------

* Test DaCy for possible bias with person entities (compared to DaNLP)
* When SpaCy fixed multiprocessing in nlp.pipe, remove current hack
