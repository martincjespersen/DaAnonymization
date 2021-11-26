#!/usr/bin/env python

"""Tests for `textanonymization` package."""

import pytest

from click.testing import CliRunner

from textprivacy import TextAnonymizer
from textprivacy import cli

import re


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_cpr_mask(response):
    """Tests CPR mask on pre-defined text"""

    test_string = "Hej med dig, mit CPR nr er 010203-2010"
    test_output = "Hej med dig, mit CPR nr er [CPR]"
    CorpusObj = TextAnonymizer()
    cpr_nr = CorpusObj.find_cpr(test_string)
    output = CorpusObj.mask_entities(test_string, cpr_nr, "CPR")

    assert output == test_output


def test_tlf_mask(response):
    """Tests telephone mask on pre-defined text"""

    test_string = "Hej med dig, mit telefon nr er +4545454545"
    test_output = "Hej med dig, mit telefon nr er [TELEFON]"
    CorpusObj = TextAnonymizer()
    telefon_nr = CorpusObj.find_telefon_nr(test_string)
    output = CorpusObj.mask_entities(test_string, telefon_nr, "TELEFON")

    assert output == test_output


def test_email_mask(response):
    """Tests email mask on pre-defined text"""

    test_string = "Hej med dig, min email er jakob.jakobsen@gmail.com"
    test_output = "Hej med dig, min email er [EMAIL]"
    CorpusObj = TextAnonymizer()
    emails = CorpusObj.find_email(test_string)
    output = CorpusObj.mask_entities(test_string, emails, "EMAIL")

    assert output == test_output


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
        "Hej, jeg hedder [PERSON], er 20 år, mit cpr er [CPR],"
        "telefon: [TELEFON] og email: [EMAIL]",
        "Hej, jeg hedder [PERSON] og er fra [LOKATION] og arbejder i "
        "[ORGANISATION], mit cpr er [CPR], telefon: [TELEFON] "
        "og email: [EMAIL]. [PERSON] er en 20 årig mand.",
    ]
    CorpusObj = TextAnonymizer(test_corpus)
    masked_corpus = CorpusObj.mask_corpus(loglevel="CRITICAL")

    assert masked_corpus == test_output, "{}\nvs.\n{}".format(
        masked_corpus[0], test_output[0]
    )


def test_custom_mask(response):
    """Tests adding a custom function to all masks on a pre-defined corpus"""

    def custom_mask_age(text):
        """
        Masks age from a text

        Args:
            text: Text to remove age from

        Returns:
            Text with [ALDER] instead of the numbers

        """
        number_pattern = r"\d+ år"
        numbers = set(re.findall(number_pattern, text))
        return numbers

    test_corpus = [
        "Hej, jeg hedder Martin Jespersen, er 20 år, mit cpr er 010203-2010,"
        "telefon: +4545454545 og email: martin.martin@gmail.com.",
        "Hej, jeg hedder Martin Jespersen og er fra Danmark og arbejder i "
        "Deloitte, mit cpr er 010203-2010, telefon: +4545454545 "
        "og email: martin.martin@gmail.com",
    ]
    test_output = [
        "Hej, jeg hedder [PERSON], er [ALDER], mit cpr er [CPR],"
        "telefon: [TELEFON] og email: [EMAIL].",
        "Hej, jeg hedder [PERSON] og er fra [LOKATION] og arbejder i "
        "[ORGANISATION], mit cpr er [CPR], telefon: [TELEFON] "
        "og email: [EMAIL]",
    ]
    CorpusObj = TextAnonymizer(test_corpus)
    CorpusObj.mapping.update({"ALDER": "[ALDER]"})
    masked_corpus = CorpusObj.mask_corpus(
        masking_order=["CPR", "TELEFON", "EMAIL", "NER", "ALDER"],
        custom_functions={"ALDER": custom_mask_age},
        loglevel="CRITICAL",
    )

    assert masked_corpus == test_output, "{}\nvs.\n{}".format(
        masked_corpus[0], test_output[0]
    )
