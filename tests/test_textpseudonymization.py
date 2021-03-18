#!/usr/bin/env python

"""Tests for `textanonymization` package."""

import pytest

from click.testing import CliRunner

from textprivacy import TextPseudonymizer

import re


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_corpus_mask(response):
    """Tests all masks on pre-defined corpus"""

    test_corpus = [
        "Hej, jeg hedder Martin Jespersen, er 20 år, mit cpr er 010203-2010,"
        "telefon: +4545454545 og email: martin.martin@gmail.com",
        "Hej, jeg hedder Martin Jespersen og er fra Danmark og arbejder i "
        "Deloitte, mit cpr er 010203-2010, telefon: +4545454545 "
        "og email: martin.martin@gmail.com. Martin er en 20 årig mand.",
    ]
    test_output = [
        "Hej, jeg hedder Person 1, er 20 år, mit cpr er CPR 2,"
        "telefon: Telefon 4 og email: Email 3",
        "Hej, jeg hedder Person 1 og er fra Lokation 4 og arbejder i "
        "Organisation 5, mit cpr er CPR 2, telefon: Telefon 6 "
        "og email: Email 3. Person 1 er en 20 årig mand.",
    ]
    CorpusObj = TextPseudonymizer(test_corpus)
    masked_corpus = CorpusObj.mask_corpus(loglevel="CRITICAL")

    assert masked_corpus == test_output, "{}\nvs.\n{}".format(
        masked_corpus[0], test_output[0]
    )


def test_prior_individuals(response):
    """Tests adding a prior known individuals to pseudonymize"""

    individuals = {
        1: {
            1: {
                "PER": set(["Martin Jespersen", "Martin", "Jespersen, Martin"]),
                "CPR": set(["010203-2010"]),
                "EMAIL": set(["martin.martin@gmail.com"]),
                "LOC": set(["Danmark"]),
                "ORG": set(["Deloitte"]),
            },
            2: {"PER": set(["Kristina"]), "ORG": set(["Novo Nordisk"])},
        }
    }

    test_corpus = [
        "Hej, jeg hedder Martin Jespersen, er 20 år, mit cpr er 010203-2010,"
        "telefon: +4545454545 og email: martin.martin@gmail.com.",
        "Hej, jeg hedder Martin Jespersen og er fra Danmark og arbejder i "
        "Deloitte, mit cpr er 010203-2010, telefon: +4545454545 "
        "og email: martin.martin@gmail.com. Martin er en 20 årig mand. "
        "Kristina er en person som arbejder i Novo Nordisk. "
        "Frank er en mand som bor i Danmark og arbejder i Netto",
    ]
    test_output = [
        "Hej, jeg hedder Person 1, er 20 år, mit cpr er CPR 2,"
        "telefon: Telefon 4 og email: Email 3.",
        "Hej, jeg hedder Person 1 og er fra Lokation 1 og arbejder i "
        "Organisation 1, mit cpr er CPR 1, telefon: Telefon 5 "
        "og email: Email 1. Person 1 er en 20 årig mand. "
        "Person 2 er en person som arbejder i Organisation 2. "
        "Person 3 er en mand som bor i Lokation 1 og arbejder i Organisation 4",
    ]
    CorpusObj = TextPseudonymizer(test_corpus, individuals=individuals)
    masked_corpus = CorpusObj.mask_corpus(loglevel="CRITICAL")

    assert masked_corpus == test_output, "{}\nvs.\n{}".format(
        masked_corpus[0], test_output[0]
    )
