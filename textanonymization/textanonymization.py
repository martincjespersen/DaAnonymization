"""Main module."""
import re


class TextAnonymizer(object):
    """docstring for TextAnonymizer"""

    def __init__(self, corpus):
        super(TextAnonymizer, self).__init__()
        self.corpus = corpus

    @classmethod
    def mask_cpr(self, text: str) -> str:
        cpr_pattern = "|".join(
            [r"[0-3]\d{1}[0-1]\d{3}-\d{4}", r"[0-3]\d{1}[0-1]\d{3} \d{4}"]
        )
        cprs = re.findall(cpr_pattern, text)
        for cpr in cprs:
            text.replace(cpr, "[CPR]")

        return text

    @classmethod
    def mask_telefon_nr(self, text: str) -> str:
        tlf_pattern = "|".join(
            [
                r"\+45\d{​8}​",
                r"\+45\d{​2}​ \d{​2}​ \d{​2}​ \d{​2}​",
                r"\+45 \d{​8}​",
                r"\+45 \d{​2}​ \d{​2}​ \d{​2}​ \d{​2}​",
                r"\d{​8}​",
                r"\d{​2}​ \d{​2}​ \d{​2}​ \d{​2}​",
            ]
        )
        tlf_nrs = re.findall(tlf_pattern, text)
        for tlf_nr in tlf_nrs:
            text.replace(tlf_nr, "[TELEFON]")

        return text

    @classmethod
    def mask_email(self, text: str) -> str:
        mail_pattern = r"[\w\.-]+@[\w\.-]+(?:\.[\w]+)+"
        emails = re.findall(mail_pattern, text)
        for email in emails:
            text.replace(email, "[EMAIL]")

        return text
