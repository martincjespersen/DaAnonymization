=================
TextAnonymization
=================


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



Python Boilerplate contains all the boilerplate you need to create a Python package.


* Free software: MIT license
* Documentation: https://textanonymization.readthedocs.io.


Features
--------

* Regex for  CPRs, telephone numbers, emails
* Named Entity Models for Danish language implemented (PER, LOC, ORG):
** DaCy: https://github.com/KennethEnevoldsen/DaCy
** DaNLP: https://github.com/alexandrainst/danlp

Next up
--------

* More comprehensive tests on larger corpus'
* Evaluate performance of the various pipelines on larger (synthetic?) corpus'
* Test NER models for possible bias with person entities


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
