"""Main module."""

from typing import List

import re


class TextAnonymizer(object):
    """docstring for TextAnonymizer"""

    def __init__(self, corpus: List[str]):
        super(TextAnonymizer, self).__init__()
        self.corpus = corpus

    @classmethod
    def mask_cpr(self, text: str) -> str:
        cpr_pattern = "|".join(
            [r"[0-3]\d{1}[0-1]\d{3}-\d{4}", r"[0-3]\d{1}[0-1]\d{3} \d{4}"]
        )
        cprs = re.findall(cpr_pattern, text)
        for cpr in cprs:
            text = text.replace(cpr, "[CPR]")

        return text

    @classmethod
    def mask_telefon_nr(self, text: str) -> str:
        tlf_pattern = "|".join(
            [
                r"\+\d{10}",
                r"\+\d{4} \d{2}​ \d{2}​ \d{2}",
                r"\+\d{2} \d{8}",
                r"\+\d{2} \d{2}​ \d{2}​ \d{2}​ \d{2}",
                r"\d{8}",
                r"\d{2}​ \d{2}​ \d{2}​​ \d{2}",
            ]
        )
        tlf_nrs = re.findall(tlf_pattern, text)
        for tlf_nr in tlf_nrs:
            print(tlf_nr)
            text = text.replace(tlf_nr, "[TELEFON]")

        return text

    @classmethod
    def mask_email(self, text: str) -> str:
        mail_pattern = r"[\w\.-]+@[\w\.-]+(?:\.[\w]+)+"
        emails = re.findall(mail_pattern, text)
        for email in emails:
            text = text.replace(email, "[EMAIL]")

        return text

    def mask_corpus(
        self, masking_methods: List[str] = ["cpr", "telefon", "email"]
    ) -> List[str]:
        methods = {
            "cpr": self.mask_cpr,
            "telefon": self.mask_telefon_nr,
            "email": self.mask_email,
        }

        for method in methods:
            self.corpus = list(map(methods[method], self.corpus))

        return self.corpus
