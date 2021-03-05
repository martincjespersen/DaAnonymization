#!/usr/bin/env python

"""Tests for `textanonymization` package."""

import pytest

from click.testing import CliRunner

from textanonymization import textanonymization
from textanonymization import cli


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

    assert textanonymization.mask_cpr(test_string) == test_output


def test_tlf_mask(response):
    """Tests CPR mask on pre-defined text"""

    test_string = "Hej med dig, mit telefon nr er +4545454545"
    test_output = "Hej med dig, mit telefon nr er [TELEFON]"

    assert textanonymization.mask_telefon_nr(test_string) == test_output


def test_email_mask(response):
    """Tests CPR mask on pre-defined text"""

    test_string = "Hej med dig, min email er jakob.jakobsen@gmail.com"
    test_output = "Hej med dig, min email er [EMAIL]"

    assert textanonymization.mask_email(test_string) == test_output


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert "textanonymization.cli.main" in result.output
    help_result = runner.invoke(cli.main, ["--help"])
    assert help_result.exit_code == 0
    assert "--help  Show this message and exit." in help_result.output
